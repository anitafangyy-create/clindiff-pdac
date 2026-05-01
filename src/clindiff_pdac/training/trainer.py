"""
Training Module for ClinDiff-PDAC
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path


class TrainingConfig:
    """Configuration for training"""
    
    def __init__(
        self,
        # Model
        data_dim: int = 20,
        kg_dim: int = 256,
        hidden_dims: list = None,
        num_timesteps: int = 1000,
        
        # Training
        batch_size: int = 64,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        
        # Optimization
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10,
        
        # Logging
        log_interval: int = 100,
        save_interval: int = 10,
        
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.data_dim = data_dim
        self.kg_dim = kg_dim
        self.hidden_dims = hidden_dims or [512, 512, 256]
        self.num_timesteps = num_timesteps
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.device = device
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ClinDiffTrainer:
    """
    Trainer for ClinDiff-PDAC model
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        else:
            self.scheduler = scheduler
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move to device
            values = batch['values'].to(self.config.device)
            mask = batch['mask'].to(self.config.device)
            
            # Create knowledge context (simplified - would come from KG)
            kg_context = torch.randn(values.shape[0], self.config.kg_dim, 
                                     device=self.config.device)
            
            # Forward pass
            loss_dict = self.model.compute_loss(values, kg_context, mask)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-self.config.log_interval:])
                print(f"Step {self.global_step}: Loss = {avg_loss:.4f}")
        
        return {'train_loss': np.mean(epoch_losses)}
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                values = batch['values'].to(self.config.device)
                mask = batch['mask'].to(self.config.device)
                
                kg_context = torch.randn(values.shape[0], self.config.kg_dim,
                                         device=self.config.device)
                
                loss_dict = self.model.compute_loss(values, kg_context, mask)
                val_losses.append(loss_dict['total_loss'].item())
        
        return {'val_loss': np.mean(val_losses)}
    
    def train(self) -> Dict[str, list]:
        """
        Main training loop
        
        Returns:
            Dictionary of training history
        """
        print(f"Training on device: {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])
                print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Early stopping
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        save_path = Path('checkpoints') / filename
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded: {filename}")


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
