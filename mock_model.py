import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Callable, Dict
import wandb
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
from accelerate import Accelerator

class MockModel(nn.Module):
    """Mock model that returns fake predictions"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Generate random logits
        base_outputs = [
            torch.randn(batch_size, self.num_classes) 
            for _ in range(3)  # Simulate 3 base learners
        ]
        meta_output = torch.randn(batch_size, self.num_classes)
        
        return {
            'base_outputs': base_outputs,
            'meta_output': meta_output
        }
    
    def compute_losses(self, outputs: Dict, targets: torch.Tensor) -> Dict:
        # Return mock losses
        return {
            'total_loss': torch.tensor(1.0),
            'meta_loss': torch.tensor(0.5),
            'base_loss': torch.tensor(0.5),
            'individual_base_losses': [torch.tensor(0.2) for _ in range(3)]
        }

class MockTrainer:
    """Mock trainer that simulates training behavior"""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accelerator = Accelerator(
            mixed_precision='no',
            gradient_accumulation_steps=1
        )
        
        # Parameters for generating mock validation accuracy
        self.accuracy_ceiling = 95.0  # Maximum accuracy to reach
        self.learning_rate = 0.1  # Controls how fast accuracy increases
        self.noise_std = 0.5  # Standard deviation of random fluctuations
        self.plateau_start = 0.7  # When to start plateauing (fraction of epochs)
    
    def generate_accuracy(self, epoch: int) -> float:
        """Generate a realistic validation accuracy for given epoch"""
        max_epochs = self.cfg.training.epochs
        progress = epoch / max_epochs
        
        # Base accuracy trend (sigmoid curve)
        if progress < self.plateau_start:
            # Initial growth phase
            x = (progress / self.plateau_start - 0.5) * 10
            base_acc = self.accuracy_ceiling / (1 + np.exp(-x))
        else:
            # Plateau phase with slight improvements
            plateau_progress = (progress - self.plateau_start) / (1 - self.plateau_start)
            base_acc = self.accuracy_ceiling - (self.accuracy_ceiling * 0.1 * np.exp(-3 * plateau_progress))
        
        # Add noise that decreases over time
        noise_factor = 1 - (progress ** 2)  # Noise reduces as training progresses
        noise = np.random.normal(0, self.noise_std * noise_factor)
        
        # Ensure accuracy stays within reasonable bounds
        accuracy = base_acc + noise
        accuracy = min(max(accuracy, 0), 100)
        
        return accuracy
    
    def train(self, model: nn.Module, callback: Optional[Callable] = None) -> float:
        """Simulate training process"""
        best_acc = 0
        
        # Initialize wandb if needed
        if self.accelerator.is_main_process and wandb.run is None:
            wandb.init(
                project=self.cfg.project.name,
                tags=self.cfg.project.tags,
                notes=self.cfg.project.notes,
                config=OmegaConf.to_container(self.cfg, resolve=True)
            )
        
        for epoch in range(self.cfg.training.epochs):
            # Generate training metrics
            train_loss = 2.0 / (epoch + 1) + 0.1 * np.random.randn()
            train_acc = self.generate_accuracy(epoch) - 5  # Training accuracy slightly lower than validation
            
            # Generate validation metrics
            val_acc = self.generate_accuracy(epoch)
            val_loss = 1.5 / (epoch + 1) + 0.1 * np.random.randn()
            
            # Update best accuracy
            best_acc = max(best_acc, val_acc)
            
            # Log metrics
            if self.accelerator.is_main_process:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_acc': train_acc,
                    'val/loss': val_loss,
                    'val/acc': val_acc
                })
                
                logging.info(
                    f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | '
                    f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.3f} | '
                    f'Val Acc: {val_acc:.2f}%'
                )
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, val_acc)

        wandb.finish()
        
        return best_acc