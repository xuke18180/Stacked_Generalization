import os
import logging
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from accelerate import Accelerator
import wandb
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.pipeline.operation import Operation
import torchvision
import numpy as np

from models import create_model_from_config

def create_optimizer(cfg: DictConfig, model_params):
    """Create optimizer based on config."""
    opt_cfg = cfg.training.optimizer
    opt_name = opt_cfg.name.lower()
    
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps
        )
    elif opt_name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
            nesterov=opt_cfg.nesterov
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
    
class TriangularLRSchedule:
    """Custom triangular learning rate schedule implementation."""
    def __init__(self, initial_lr: float, peak_lr: float, final_lr: float, 
                 total_steps: int, peak_step: int):
        self.schedule = np.interp(
            np.arange(total_steps),
            [0, peak_step, total_steps],
            [initial_lr, peak_lr, final_lr]
        )
        self.total_steps = total_steps
        
    def __call__(self, step):
        if step >= self.total_steps:
            return float(self.schedule[-1])
        return float(self.schedule[step])
    
    def __getstate__(self):
        """Support proper serialization for state_dict."""
        return self.__dict__

def create_scheduler(cfg: DictConfig, optimizer, steps_per_epoch: int):
    """Create learning rate scheduler based on config."""
    sched_cfg = cfg.training.scheduler
    sched_name = sched_cfg.name.lower()
    
    if sched_name == "custom_triangular":
        total_steps = cfg.training.epochs * steps_per_epoch
        peak_step = sched_cfg.peak_epoch * steps_per_epoch
        schedule = TriangularLRSchedule(
            initial_lr=sched_cfg.initial_lr,
            peak_lr=sched_cfg.peak_lr,
            final_lr=sched_cfg.final_lr,
            total_steps=total_steps,
            peak_step=peak_step
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    elif sched_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sched_cfg.max_lr,
            epochs=cfg.training.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=sched_cfg.pct_start,
            div_factor=sched_cfg.div_factor,
            final_div_factor=sched_cfg.final_div_factor
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.T_max,
            eta_min=sched_cfg.eta_min
        )
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_accelerator()
        self.setup_wandb()
        self.setup_dataloaders()
        
    def setup_accelerator(self):
        self.accelerator = Accelerator(
            mixed_precision='fp16' if self.cfg.training.mixed_precision else 'no',
            gradient_accumulation_steps=1,
        )
        self.device = self.accelerator.device
        
    def setup_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.cfg.project.name,
                tags=self.cfg.project.tags,
                notes=self.cfg.project.notes,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
    
    def setup_dataloaders(self):
        # Create pipelines for FFCV
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze()
        ]
        
        image_pipeline_train: List[Operation] = [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(),
            RandomTranslate(padding=2, fill=tuple(map(int, self.cfg.dataset.mean))),
            Cutout(4, tuple(map(int, self.cfg.dataset.mean))),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float16 if self.cfg.training.mixed_precision else torch.float32),
            torchvision.transforms.Normalize(self.cfg.dataset.mean, self.cfg.dataset.std),
        ]
        
        image_pipeline_val = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float16 if self.cfg.training.mixed_precision else torch.float32),
            torchvision.transforms.Normalize(self.cfg.dataset.mean, self.cfg.dataset.std),
        ]
        
        self.train_loader = Loader(
            self.cfg.dataset.train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            order=OrderOption.RANDOM,
            drop_last=True,
            pipelines={
                'image': image_pipeline_train,
                'label': label_pipeline
            }
        )
        
        self.val_loader = Loader(
            self.cfg.dataset.val_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={
                'image': image_pipeline_val,
                'label': label_pipeline
            }
        )
    
    def train_epoch(self, model, optimizer, scheduler, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            with autocast('cuda', enabled=self.cfg.training.mixed_precision):
                outputs = model(images)
                losses = model.compute_losses(outputs, targets)
                loss = losses['total_loss']
                            
            self.accelerator.backward(loss)
            if self.cfg.training.gradient_clip > 0:
                self.accelerator.clip_grad_norm_(model.parameters(), self.cfg.training.gradient_clip)
                
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            _, predicted = outputs["meta_output"].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0 and self.accelerator.is_main_process:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/acc': 100. * correct / total,
                    'train/lr': scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
        
        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self, model, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                with autocast('cuda', enabled=self.cfg.training.mixed_precision):
                    outputs = model(images)
                    losses = model.compute_losses(outputs, targets)
                    loss = losses['total_loss']

                    if self.cfg.training.lr_tta:
                        outputs_flip = model(torch.flip(images, dims=[-1]))
                        losses_flip = model.compute_losses(outputs_flip, targets)
                        loss += losses_flip['total_loss']
                        loss /= 2
                
                total_loss += loss.item()
                _, predicted = outputs["meta_output"].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total

    def train(self, model, callback=None):
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        optimizer = create_optimizer(self.cfg, model.parameters())
        scheduler = create_scheduler(self.cfg, optimizer, len(self.train_loader))
        
        # Prepare for distributed training
        model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            model, optimizer, self.train_loader, scheduler
        )
        
        best_acc = 0
        for epoch in range(self.cfg.training.epochs):
            train_loss, train_acc = self.train_epoch(
                model, optimizer, scheduler, epoch
            )
            val_loss, val_acc = self.validate(model, criterion)

            if callback is not None:
                callback(epoch, val_acc)
            
            if self.accelerator.is_main_process:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_acc': train_acc,
                    'val/loss': val_loss,
                    'val/acc': val_acc
                })
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.accelerator.save_state('./best_model_checkpoint/')
                    
                logging.info(
                    f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | '
                    f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.3f} | '
                    f'Val Acc: {val_acc:.2f}%'
                )
        
        return best_acc

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Set random seed
    torch.manual_seed(cfg.training.seed)
    
    # Initialize trainer
    trainer = Trainer(cfg)
    
    # Create model
    model = create_model_from_config(cfg)
    
    # Train the model
    trainer.train(model)

if __name__ == "__main__":
    main()