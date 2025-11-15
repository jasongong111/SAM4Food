"""
Training Pipeline for SAM with LoRA Fine-tuning

This module implements the training loop for fine-tuning SAM on FoodSeg103
with LoRA adapters and specified hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tqdm
from datetime import datetime
import json

from ..models.sam_lora import SAMLoRAModel
from ..data.foodseg_dataset import create_data_loaders
from ..configs.config import Config
from ..utils.metrics import calculate_miou, calculate_dice


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for mask segmentation
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss
        
        Args:
            pred: Predicted masks (B, H, W)
            target: Ground truth masks (B, H, W)
        
        Returns:
            Dice loss value
        """
        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten for calculation
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice)
        return 1.0 - dice


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy loss with logits (for training stability)
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            pred, target.float(), pos_weight=self.pos_weight
        )


class CombinedLoss(nn.Module):
    """
    Combined loss function with Dice and BCE
    """
    
    def __init__(self, dice_weight: float = 0.5, use_pos_weight: bool = True):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = BCEWithLogitsLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        combined_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return {
            'total_loss': combined_loss,
            'dice_loss': dice,
            'bce_loss': bce
        }


class Trainer:
    """
    Trainer class for SAM LoRA fine-tuning
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # Initialize model
        self.model = SAMLoRAModel(config).to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(dice_weight=config.training.dice_weight)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler(enabled=config.training.use_mixed_precision)
        
        # Initialize data loaders
        self.train_loader, self.val_loader = create_data_loaders(config)
        
        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.train_history = []
        self.val_history = []
        
        # Initialize logging
        self._setup_logging()
        
        # Print model summary
        self._print_model_summary()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer for trainable parameters"""
        trainable_params = self.model.get_trainable_parameters()
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")
        
        # Group parameters for differential learning rates if needed
        params = [p for group in trainable_params for p in group['params']]
        
        if self.config.training.optimizer.lower() == 'adam':
            return optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.training.scheduler is None:
            return None
        elif self.config.training.scheduler.lower() == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler.lower() == 'linear':
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.config.training.num_epochs
            )
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging directories and files"""
        self.log_dir = Path(self.config.system.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file for this training session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.json"
        
    def _print_model_summary(self):
        """Print model summary with parameter counts"""
        total_params, trainable_params = self.model.count_parameters()
        
        print(f"Model Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Print LoRA specific info
        if hasattr(self.model.mask_decoder, 'peft_config'):
            print(f"LoRA rank: {self.config.model.lora_rank}")
            print(f"LoRA alpha: {self.config.model.lora_alpha}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_bce_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            image = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.training.use_mixed_precision):
                # Get image features
                image_features = self.model.image_encoder(image)
                
                # Prepare prompts
                prompts = {
                    'points': batch['prompts']['points'].to(self.device),
                    'point_labels': batch['prompts']['point_labels'].to(self.device)
                }
                
                # Get predictions
                pred_masks = self.model.sam_model(image_features, prompts)
                
                # Calculate loss
                losses = self.criterion(pred_masks.squeeze(1), mask)
            
            # Backward pass
            self.scaler.scale(losses['total_loss']).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update running statistics
            epoch_loss += losses['total_loss'].item()
            epoch_dice_loss += losses['dice_loss'].item()
            epoch_bce_loss += losses['bce_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Dice': f"{losses['dice_loss'].item():.4f}",
                'BCE': f"{losses['bce_loss'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log at specified frequency
            if (batch_idx + 1) % self.config.training.log_frequency == 0:
                self._log_batch(batch_idx, losses, num_batches)
        
        # Calculate epoch averages
        avg_loss = epoch_loss / num_batches
        avg_dice_loss = epoch_dice_loss / num_batches
        avg_bce_loss = epoch_bce_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_dice_loss': avg_dice_loss,
            'train_bce_loss': avg_bce_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_miou = 0.0
        total_dice = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(self.val_loader, desc="Validation")):
                # Move batch to device
                image = batch['image'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Get image features
                image_features = self.model.image_encoder(image)
                
                # Prepare prompts
                prompts = {
                    'points': batch['prompts']['points'].to(self.device),
                    'point_labels': batch['prompts']['point_labels'].to(self.device)
                }
                
                # Get predictions
                pred_masks = self.model.sam_model(image_features, prompts)
                pred_masks_sigmoid = torch.sigmoid(pred_masks.squeeze(1))
                
                # Calculate metrics
                miou = calculate_miou(pred_masks_sigmoid, mask)
                dice = calculate_dice(pred_masks_sigmoid, mask)
                
                total_miou += miou.item() * image.size(0)
                total_dice += dice.item() * image.size(0)
                num_samples += image.size(0)
        
        # Calculate epoch averages
        avg_miou = total_miou / num_samples
        avg_dice = total_dice / num_samples
        
        return {
            'val_miou': avg_miou,
            'val_dice': avg_dice
        }
    
    def _log_batch(self, batch_idx: int, losses: Dict[str, torch.Tensor], num_batches: int):
        """Log batch statistics"""
        if self.config.system.use_wandb:
            import wandb
            wandb.log({
                'train/batch_loss': losses['total_loss'].item(),
                'train/batch_dice_loss': losses['dice_loss'].item(),
                'train/batch_bce_loss': losses['bce_loss'].item(),
                'train/learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def _log_epoch(self, epoch_stats: Dict[str, float]):
        """Log epoch statistics"""
        # Update history
        self.train_history.append(epoch_stats)
        
        if 'val_miou' in epoch_stats:
            self.val_history.append(epoch_stats)
            epoch_stats['epoch'] = self.current_epoch
            
            # Log to wandb if enabled
            if self.config.system.use_wandb:
                import wandb
                wandb.log(epoch_stats)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.system.model_save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        
        self.model.save_checkpoint(
            checkpoint_path, 
            self.current_epoch, 
            {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict()
            }
        )
        
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            self.model.save_checkpoint(
                best_path,
                self.current_epoch,
                {
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict()
                }
            )
            print(f"Best model saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = self.model.load_checkpoint(checkpoint_path)
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        
        # Load optimizer state if present
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if present
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training for {self.config.training.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.config.training.use_mixed_precision}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_stats = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_stats['train_loss']:.4f}")
            
            # Validate epoch
            val_stats = {}
            if (epoch + 1) % self.config.training.eval_frequency == 0:
                val_stats = self.validate_epoch()
                print(f"Epoch {epoch}: Val mIoU = {val_stats['val_miou']:.4f}, Val Dice = {val_stats['val_dice']:.4f}")
                
                # Update best mIoU
                if val_stats['val_miou'] > self.best_miou:
                    self.best_miou = val_stats['val_miou']
                    self.save_checkpoint(is_best=True)
            
            # Combine statistics
            epoch_stats = {**train_stats, **val_stats}
            
            # Log epoch statistics
            self._log_epoch(epoch_stats)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self.save_checkpoint()
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
        
        # Final validation
        if self.val_history:
            final_stats = self.val_history[-1]
            print(f"\nTraining completed!")
            print(f"Best mIoU: {self.best_miou:.4f}")
            print(f"Final mIoU: {final_stats.get('val_miou', 0.0):.4f}")
            print(f"Final Dice: {final_stats.get('val_dice', 0.0):.4f}")
        
        # Save training history
        self._save_training_history()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def _save_training_history(self):
        """Save training history to JSON file"""
        history = {
            'config': self.config.__dict__,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_miou': self.best_miou,
            'final_epoch': self.current_epoch
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {self.log_file}")