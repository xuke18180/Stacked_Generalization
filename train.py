import os
import logging
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import accelerate
from accelerate import Accelerator
import wandb
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.pipeline.operation import Operation
import torchvision
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from hydra.utils import get_original_cwd

from models import BaseLearner, BaseModelConfig, create_model_from_config, StackedEnsemble, create_base_models_from_config, create_meta_learner_from_config

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
    

def create_scheduler(cfg: DictConfig, optimizer, steps_per_epoch: int):
    """Create learning rate scheduler based on config."""
    sched_cfg = cfg.training.scheduler
    sched_name = sched_cfg.name.lower()
    total_steps = cfg.training.epochs * steps_per_epoch
    
    if sched_name == "linear":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.training.optimizer.lr,
            epochs=cfg.training.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=sched_cfg.pct_start,
            anneal_strategy='linear',
            div_factor=sched_cfg.div_factor,
            final_div_factor=sched_cfg.final_div_factor
        )
    elif sched_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.training.optimizer.lr,
            epochs=cfg.training.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=sched_cfg.pct_start,
            div_factor=sched_cfg.div_factor,
            final_div_factor=sched_cfg.final_div_factor
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
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
        wandb.finish() 
        return best_acc

class StackedTrainer(Trainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.n_folds = cfg.training.n_folds
        self.fold_models = {i: [] for i in range(len(cfg.model.base_learners))}
        
        # Initialize metric tracking for each model
        self.model_metrics = {}
        
        if self.cfg.training.save_fold_models:
            os.makedirs(self.cfg.training.fold_model_dir, exist_ok=True)
        
    def _update_model_metrics(self, model_idx: int, epoch: int, metrics: dict):
        """Update running metrics for a model across folds"""
        if model_idx not in self.model_metrics:
            self.model_metrics[model_idx] = {}
        
        epoch_metrics = self.model_metrics[model_idx]
        if epoch not in epoch_metrics:
            epoch_metrics[epoch] = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'prediction_stability': [],
                'confidence': [],
                'learning_rate': None  # Single value, not averaged across folds
            }
        
        # Add metrics for this fold
        for key, value in metrics.items():
            if key == 'learning_rate':
                epoch_metrics[epoch][key] = value  # Don't average learning rate
            else:
                epoch_metrics[epoch][key].append(value)

    def _log_aggregated_metrics(self):
        """Log aggregated metrics across folds for a model"""
        if not self.accelerator.is_main_process:
            return
            
        for model_idx, metrics in self.model_metrics.items():
            for epoch, epoch_metrics in metrics.items():
                epoch_metrics = self.model_metrics[model_idx][epoch]
        
                # Log learning rate separately (not averaged)
                if epoch_metrics['learning_rate'] is not None:
                    wandb.log({
                        f'model_{model_idx}/learning_rate': epoch_metrics['learning_rate'],
                        'epoch': epoch
                    })
                
                # Calculate mean and std for other metrics
                for metric_name, values in epoch_metrics.items():
                    if metric_name == 'learning_rate':
                        continue
                    
                    values_tensor = torch.tensor(values)
                    mean = values_tensor.mean().item()
                    if len(values) > 1:
                        std = values_tensor.std().item()
                    else:
                        std = 0.0
                    
                    # Log mean and bounds
                    wandb.log({
                        f'model_{model_idx}/{metric_name}/mean': mean,
                        f'model_{model_idx}/{metric_name}/upper': mean + std,
                        f'model_{model_idx}/{metric_name}/lower': mean - std,
                        'epoch': epoch
                    })

    def setup_dataloaders(self):
        """Override to create FFCV dataloaders for each fold"""
        # Create pipelines similar to parent class
        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze()
        ]
        
        image_pipeline = [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, self.cfg.dataset.mean))),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float16 if self.cfg.training.mixed_precision else torch.float32),
            torchvision.transforms.Normalize(self.cfg.dataset.mean, self.cfg.dataset.std),
        ]

        # Create fold indices
        n_samples = self._get_dataset_size(self.cfg.dataset.train_dataset)
        fold_size = n_samples // self.cfg.training.n_folds
        indices = torch.randperm(n_samples)
        
        # Create loaders for each fold
        self.fold_loaders = []
        for i in range(self.cfg.training.n_folds):
            val_indices = indices[i*fold_size:(i+1)*fold_size]
            train_indices = torch.cat([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            
            train_loader = Loader(
                self.cfg.dataset.train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_workers,
                indices=train_indices,
                order=OrderOption.RANDOM,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                }
            )
            
            val_loader = Loader(
                self.cfg.dataset.train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_workers,
                indices=val_indices,
                order=OrderOption.SEQUENTIAL,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                }
            )
            
            self.fold_loaders.append((train_loader, val_loader))
            
        # Keep original validation loader for final evaluation
        self.test_loader = Loader(
            self.cfg.dataset.val_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            order=OrderOption.SEQUENTIAL,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )

    def train(self, callback=None):
        """Train using k-fold cross validation with multiple level 0 models"""
        # Reset metrics for new training run
        self.model_metrics = {}
        
        level0_predictions = []  # List of predictions for each fold
        level0_targets = []
        level0_images = []
        # For each fold
        for fold_idx, (train_loader, val_loader) in enumerate(self.fold_loaders):
            logging.info(f"Training fold {fold_idx+1}/{self.cfg.training.n_folds}")
            
            fold_predictions = []  # Predictions from all base models for this fold
            # Create base learners for this fold
            base_learners = create_base_models_from_config(self.cfg)
            
            # Train each base learner
            for model_idx, base_model in enumerate(base_learners):
                logging.info(f"Training base model {model_idx+1}/{len(base_learners)}")
                
                # Train on this fold
                optimizer = create_optimizer(self.cfg, base_model.parameters())
                scheduler = create_scheduler(self.cfg, optimizer, len(train_loader))
                
                base_model, optimizer, train_loader, scheduler = self.accelerator.prepare(
                    base_model, optimizer, train_loader, scheduler
                )
                
                # Train base model
                best_acc = self._train_fold(base_model, optimizer, scheduler, train_loader, val_loader, 
                                          fold_idx, model_idx)
                
                # Get predictions on validation fold
                unwrapped_model = self.accelerator.unwrap_model(base_model)

                # saving of base models are handled in the StackedEnsemble class
                # if self.cfg.training.save_fold_models:
                #     self.accelerator.wait_for_everyone()
                #     self.accelerator.save({
                #         'accuracy': best_acc,
                #         'model': unwrapped_model.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'scheduler': scheduler.state_dict(),
                #     }, f'{self.cfg.training.fold_model_dir}/model_{model_idx}_fold_{fold_idx}.pth')

                self.fold_models[model_idx].append(unwrapped_model)
            
            fold_predictions, fold_targets, fold_images = self._get_fold_predictions([self.fold_models[model_idx][fold_idx] for model_idx in range(len(self.cfg.model.base_learners))], val_loader)
            
            # Concatenate predictions from all base models for this fold
            level0_predictions.append(fold_predictions)
            level0_targets.append(fold_targets)
            level0_images.append(fold_images)
            self._log_aggregated_metrics()
            
        # Prepare level1 data
        level1_data = self._prepare_level1_data(level0_predictions, level0_targets, level0_images)

        # save fold models
        data_dir = os.path.join(get_original_cwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        for model_idx, fold_models in self.fold_models.items():
            for fold_idx, model in enumerate(fold_models):
                model_path = os.path.join(data_dir, f"model_{model_idx}_fold_{fold_idx}.pt")
                torch.save(model, model_path)
        
        # # Train level 1 model using collected predictions
        # meta_learner = create_meta_learner_from_config(self.cfg)
        # best_acc = self._train_level1(meta_learner, level1_data)
        
        wandb.finish()
        # return best_acc
        return

    def train_level1(self):
        # Load existing level1 data
        data_dir = os.path.join(get_original_cwd(), "data")
        level1_data_path = os.path.join(data_dir, 'level1_data.pt')
        level1_data = self.load_level1_data(level1_data_path)
        logging.info(f"Loaded existing level1 data from {level1_data_path}")

        # load fold models
        for model_idx in range(len(self.cfg.model.base_learners)):
            for fold_idx in range(self.cfg.training.n_folds):
                model_path = os.path.join(data_dir, f"model_{model_idx}_fold_{fold_idx}.pt")
                model = torch.load(model_path, weights_only=False)
                model = self.accelerator.prepare(model)
                self.fold_models[model_idx].append(model)

        meta_learner = create_meta_learner_from_config(self.cfg)
        best_acc = self._train_level1(meta_learner, level1_data)
        
        wandb.finish()
        return best_acc

    def _create_base_model(self, model_idx: int, base_model_cfg: DictConfig):
        """Create a single base model instance"""
        # Create base model config
        base_config = BaseModelConfig(
            architecture=base_model_cfg.architecture,
            pretrained=base_model_cfg.pretrained,
            init_method=base_model_cfg.get('init_method'),
            init_params=base_model_cfg.get('init_params')
        )
        
        return BaseLearner(
            config=base_config,
            num_classes=self.cfg.dataset.num_classes
        )

    def _get_fold_accuracies(self):
        """Get accuracies for each model type across folds"""
        accuracies = {}
        for model_idx, fold_models in self.fold_models.items():
            accuracies[f'model_{model_idx}'] = [
                getattr(model, 'best_acc', 0) for model in fold_models
            ]
        return accuracies

    def _prepare_level1_data(self, level0_predictions, level0_targets, level0_images):
        """Prepare training data for level 1 model"""
        
        # Concatenate all data
        X_preds = torch.cat(level0_predictions, dim=0)
        X_images = torch.cat(level0_images, dim=0)
        y = torch.cat(level0_targets, dim=0)
        
        # Save the level1 data if configured
        if self.cfg.training.save_level1_data:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(get_original_cwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            save_path = os.path.join(data_dir, 'level1_data.pt')
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                torch.save({
                    'predictions': X_preds,
                    'images': X_images,
                    'targets': y
                }, save_path)
                logging.info(f"Saved level1 data to {save_path}")
        
        return DataLoader(
            TensorDataset(X_preds, X_images, y),
            batch_size=self.cfg.training.batch_size,
            shuffle=True
        )

    def load_level1_data(self, data_path: str) -> DataLoader:
        """Load previously saved level1 data and return a DataLoader"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Level1 data not found at {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        return DataLoader(
            TensorDataset(data['predictions'], data['images'], data['targets']),
            batch_size=self.cfg.training.batch_size,
            shuffle=True
        )

    def _train_fold(self, model, optimizer, scheduler, train_loader, val_loader, fold_idx, model_idx):
        """Train a single model on one fold"""
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        best_acc = 0
        patience_counter = 0
        
        # Track predictions for diversity metrics
        prev_predictions = None
        
        for epoch in range(self.cfg.training.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training loop
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Optimization step
                self.accelerator.backward(loss)
                if self.cfg.training.gradient_clip > 0:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.cfg.training.gradient_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Validation step
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_predictions = []
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
                    all_predictions.append(outputs.softmax(dim=1))
            
            # Compute epoch metrics
            train_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Compute diversity metrics
            all_predictions_tensor = torch.cat(all_predictions)
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            if prev_predictions is not None:
                # Prediction stability (correlation with previous epoch)
                prediction_stability = torch.corrcoef(
                    torch.stack([all_predictions_tensor.flatten(), 
                               prev_predictions.flatten()])
                )[0,1].item()
                
                # Confidence calibration
                confidence = all_predictions_tensor.max(dim=1)[0].mean().item()
                
                metrics.update({
                    'prediction_stability': prediction_stability,
                    'confidence': confidence
                })
            
            # Update and log aggregated metrics
            self._update_model_metrics(model_idx, epoch, metrics)
            
            # Store predictions for next epoch
            prev_predictions = all_predictions_tensor
            
            # Log progress more concisely
            if self.accelerator.is_main_process:
                logging.info(
                    f'Model {model_idx}, Epoch {epoch}: '
                    f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
                    f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%'
                )
            
            # Early stopping check
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= self.cfg.training.patience:
            #         logging.info(f"Early stopping triggered for model {model_idx} fold {fold_idx}")
            #         break
        
        # Store best accuracy for this model/fold combination
        setattr(model, 'best_acc', best_acc)
        return best_acc

    def _train_level1(self, meta_learner, level1_loader):
        """Train the level 1 (meta) model using predictions from level 0 models"""
        logging.info("Training level 1 model...")
        
        # Use meta-learner specific settings
        meta_cfg = self.cfg.training.meta_learner
        
        # Create optimizer and scheduler for level 1
        optimizer = create_optimizer(
            DictConfig({'training': {'optimizer': meta_cfg.optimizer}}), 
            meta_learner.parameters()
        )
        scheduler = create_scheduler(
            DictConfig({'training': {
                'scheduler': meta_cfg.scheduler, 
                'epochs': meta_cfg.epochs,
                'optimizer': meta_cfg.optimizer
            }}),
            optimizer, 
            len(level1_loader)
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        
        # Create ensemble for validation
        ensemble = StackedEnsemble(
            fold_models=self.fold_models,
            level1_model=meta_learner,
            fold_accuracies=self._get_fold_accuracies(),
            strategy=self.cfg.training.test_ensemble_method,
            device=self.device
        )
        
        # Prepare for distributed training
        meta_learner, optimizer, level1_loader, scheduler = self.accelerator.prepare(
            meta_learner, optimizer, level1_loader, scheduler
        )
        
        best_acc = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(meta_cfg.epochs):  # Use meta-learner specific epochs
            meta_learner.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training step
            for batch_idx, (level0_preds, images, targets) in enumerate(level1_loader):
                level0_preds = level0_preds.to(self.device)
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass through meta learner
                meta_output = meta_learner(level0_preds, images)
                loss = criterion(meta_output, targets)
                
                # Optimization step
                self.accelerator.backward(loss)
                if meta_cfg.gradient_clip > 0:  # Use meta-learner specific gradient clip
                    self.accelerator.clip_grad_norm_(meta_learner.parameters(), meta_cfg.gradient_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = meta_output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Log batch metrics
                if batch_idx % self.cfg.training.log_batch_frequency == 0 and self.accelerator.is_main_process:
                    wandb.log({
                        'meta/train_batch_loss': loss.item(),
                        'meta/train_batch_acc': 100. * correct / total,
                        'meta/learning_rate': scheduler.get_last_lr()[0],
                        'meta_epoch': epoch
                    })
            
            # Validation step
            meta_learner.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, targets in self.test_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = ensemble(images)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Compute epoch metrics
            train_loss = total_loss / len(level1_loader)
            train_acc = 100. * correct / total
            val_loss = val_loss / len(self.test_loader)
            val_acc = 100. * val_correct / val_total
            
            # Log epoch metrics
            if self.accelerator.is_main_process:
                wandb.log({
                    'meta/train_epoch_loss': train_loss,
                    'meta/train_epoch_acc': train_acc,
                    'meta/val_loss': val_loss,
                    'meta/val_acc': val_acc,
                    'meta_epoch': epoch
                })
                
                # Log progress
                logging.info(
                    f'Meta Epoch: {epoch} | '
                    f'Train Loss: {train_loss:.3f} | '
                    f'Train Acc: {train_acc:.2f}% | '
                    f'Val Loss: {val_loss:.3f} | '
                    f'Val Acc: {val_acc:.2f}%'
                )
            
            # Save best model and check early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= meta_cfg.patience:  # Use meta-learner specific patience
                    logging.info("Early stopping triggered for meta learner")
                    break
        

        if self.cfg.training.save_fold_models:
            ensemble.save(self.cfg.training.fold_model_dir)
        return best_acc

    def _get_dataset_size(self, dataset_path: str) -> int:
        """Get total number of samples in FFCV dataset"""
        # Create minimal pipelines just to count samples
        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze()
        ]
        
        image_pipeline = [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage()
        ]
        
        # Create a loader with batch size 1 just to count samples
        loader = Loader(
            dataset_path,
            batch_size=1,
            num_workers=1,
            order=OrderOption.SEQUENTIAL,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )
        
        return len(loader) * loader.batch_size

    def _get_fold_predictions(self, model_list, val_loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get predictions from a model on validation fold"""
        all_predictions = []
        all_targets = []
        all_images = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                batch_predictions = []
                for model in model_list:
                    model.eval()
                    outputs = model(images)  # [batch_size, n_classes]
                    batch_predictions.append(outputs)
                
                # Stack predictions from all models for this batch
                # [batch_size, n_models, n_classes]
                batch_predictions = torch.stack(batch_predictions, dim=1)
                # Reshape to [batch_size, n_models * n_classes]
                batch_predictions = batch_predictions.reshape(batch_predictions.shape[0], -1)
                
                all_predictions.append(batch_predictions)
                all_targets.append(targets)
                all_images.append(images.cpu())
        
        # Concatenate along batch dimension
        return (
            torch.cat(all_predictions, dim=0),  # [n_samples, n_models * n_classes]
            torch.cat(all_targets, dim=0),      # [n_samples]
            torch.cat(all_images, dim=0)        # [n_samples, channels, height, width]
        )


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Set random seed
    torch.manual_seed(cfg.training.seed)
    
    # Previous version
    # Initialize trainer
    # trainer = Trainer(cfg)
    # Create model
    # model = create_model_from_config(cfg)
    
    # New version
    trainer = StackedTrainer(cfg)    
    # Train the model
    # trainer.train()
    trainer.train_level1()
    
if __name__ == "__main__":
    main()