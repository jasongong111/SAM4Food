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
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tqdm
from datetime import datetime
import json

from ..models.sam_lora import SAMLoRAModel
from ..data.foodseg_dataset import create_data_loaders
from configs.config import Config
from ..utils.metrics import (
    calculate_miou,
    calculate_dice,
    calculate_classification_accuracy,
    calculate_instance_iou,
    calculate_instance_dice,
)
from .losses import JointIngredientLoss


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
        ingredient_mode = config.model.use_ingredient_head
        foodinsseg_mode = config.data.dataset_name == "FoodInsSeg"
        if ingredient_mode != foodinsseg_mode:
            raise ValueError(
                "use_ingredient_head and dataset_name='FoodInsSeg' must be enabled together. "
                f"Got use_ingredient_head={ingredient_mode}, dataset_name={config.data.dataset_name!r}."
            )

        self.config = config
        self.device = torch.device(config.system.device)
        
        # Initialize model
        self.model = SAMLoRAModel(config).to(self.device)
        
        # Initialize loss function
        if config.model.use_ingredient_head:
            self.criterion = JointIngredientLoss(
                mask_weight=config.training.mask_loss_weight,
                class_weight=config.training.classification_loss_weight,
            )
        else:
            self.criterion = CombinedLoss(dice_weight=config.training.dice_weight)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Initialize scaler for mixed precision
        if config.training.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler('cuda', enabled=True)
        else:
            self.scaler = GradScaler('cpu', enabled=False)
        
        # Initialize data loaders
        if config.data.dataset_name == "FoodInsSeg":
            from ..data.foodinsseg_dataset import FoodInsSegDataset
            train_dataset = FoodInsSegDataset(config, split="train")
            val_dataset = FoodInsSegDataset(config, split="val")
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.system.num_workers,
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.system.num_workers,
                pin_memory=True,
            )
        else:
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
        
        # Filter parameters - ensure they're leaf tensors for optimizer
        # PEFT models should have leaf tensors, but we check to be safe
        params = []
        non_leaf_params = []
        
        for p in trainable_params:
            if p.requires_grad:
                if p.is_leaf:
                    params.append(p)
                else:
                    non_leaf_params.append(p)
        
        # If we have non-leaf parameters, we need to handle them differently
        # The issue is that PyTorch optimizers require leaf tensors
        if non_leaf_params:
            print(f"⚠️  Warning: Found {len(non_leaf_params)} non-leaf parameters")
            print("  Attempting to extract leaf parameters from PEFT model...")
            
            # Try to get the actual trainable parameters from PEFT adapters
            # PEFT stores LoRA weights in the model's state_dict
            try:
                # Get state dict and recreate parameters
                for name, param in self.model.mask_decoder.named_parameters():
                    if param.requires_grad and 'lora' in name.lower():
                        # LoRA parameters should be leaf tensors
                        if param.is_leaf:
                            if param not in params:
                                params.append(param)
                
                # Also check prompt encoder if it has LoRA
                if hasattr(self.model.prompt_encoder, 'peft_config'):
                    for name, param in self.model.prompt_encoder.named_parameters():
                        if param.requires_grad and 'lora' in name.lower():
                            if param.is_leaf:
                                if param not in params:
                                    params.append(param)
            except Exception as e:
                print(f"  Could not extract leaf parameters: {e}")
        
        if len(params) == 0:
            raise ValueError(
                "No valid trainable parameters found! "
                "All parameters are non-leaf tensors. "
                "This may indicate an issue with PEFT/LoRA setup."
            )
        
        print(f"Setting up optimizer with {len(params)} parameter groups")
        print(f"Total trainable parameters: {sum(p.numel() for p in params):,}")
        if non_leaf_params:
            print(f"  (Excluded {len(non_leaf_params)} non-leaf parameters)")
        
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
        if self.config.model.use_ingredient_head:
            return self._train_epoch_ingredient()
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
            with autocast(device_type=self.device.type, enabled=self.config.training.use_mixed_precision):
                # Get image features
                image_features = self.model.image_encoder(image)
                
                # Prepare prompts
                prompts = {
                    'points': batch['prompts']['points'].to(self.device),
                    'point_labels': batch['prompts']['point_labels'].to(self.device)
                }
                
                # Get predictions (upsampled to ground-truth resolution)
                pred_masks = self.model.sam_model(image_features, prompts)
                pred_masks = self.model.resize_predictions(pred_masks, mask.shape[-2:])
                
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
        if self.config.model.use_ingredient_head:
            return self._validate_epoch_ingredient()
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
                
                # Get predictions and resize to ground-truth resolution
                pred_masks = self.model.sam_model(image_features, prompts)
                pred_masks = self.model.resize_predictions(pred_masks, mask.shape[-2:])
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
    
    def _train_epoch_ingredient(self) -> Dict[str, float]:
        """Train for one epoch using joint instance mask + ingredient classification loss."""
        self.model.train()
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        epoch_acc = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm.tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch} [ingredient]",
        )

        for batch_idx, batch in enumerate(progress_bar):
            image = batch['image'].to(self.device)
            instance_mask = batch['instance_mask'].to(self.device)
            class_id = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            with autocast(
                device_type=self.device.type,
                enabled=self.config.training.use_mixed_precision,
            ):
                prompts = {
                    'points': batch['prompts']['points'].to(self.device),
                    'point_labels': batch['prompts']['point_labels'].to(self.device),
                }
                outputs = self.model.forward_instance(image, prompts)
                mask_logits = self.model.resize_predictions(
                    outputs["mask_logits"], instance_mask.shape[-2:]
                )
                losses = self.criterion(
                    mask_logits, instance_mask, outputs["class_logits"], class_id
                )

            self.scaler.scale(losses['total_loss']).backward()

            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            with torch.no_grad():
                # Head-only ablation (disable LoRA optimizer, keep only ingredient-head params) is deferred to a future task.
                batch_iou = sum(
                    calculate_instance_iou(mask_logits.squeeze(1)[i], instance_mask[i])
                    for i in range(image.size(0))
                ) / image.size(0)
                batch_dice = sum(
                    calculate_instance_dice(mask_logits.squeeze(1)[i], instance_mask[i])
                    for i in range(image.size(0))
                ) / image.size(0)
                acc = calculate_classification_accuracy(
                    outputs["class_logits"], class_id
                )

            epoch_loss += losses['total_loss'].item()
            epoch_mask_loss += losses['mask_loss'].item()
            epoch_cls_loss += losses['classification_loss'].item()
            epoch_iou += batch_iou
            epoch_dice += batch_dice
            epoch_acc += acc

            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'MaskL': f"{losses['mask_loss'].item():.4f}",
                'ClsL': f"{losses['classification_loss'].item():.4f}",
                'IoU': f"{batch_iou:.4f}",
                'Dice': f"{batch_dice:.4f}",
                'Acc': f"{acc:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

            if (batch_idx + 1) % self.config.training.log_frequency == 0:
                self._log_batch(batch_idx, losses, num_batches)

        return {
            'train_loss': epoch_loss / num_batches,
            'train_mask_loss': epoch_mask_loss / num_batches,
            'train_cls_loss': epoch_cls_loss / num_batches,
            'train_iou': epoch_iou / num_batches,
            'train_dice': epoch_dice / num_batches,
            'train_acc': epoch_acc / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    def _validate_epoch_ingredient(self) -> Dict[str, float]:
        """Validate for one epoch in ingredient mode."""
        self.model.eval()
        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader, desc="Validation [ingredient]"):
                image = batch['image'].to(self.device)
                instance_mask = batch['instance_mask'].to(self.device)
                class_id = batch['class_id'].to(self.device)

                prompts = {
                    'points': batch['prompts']['points'].to(self.device),
                    'point_labels': batch['prompts']['point_labels'].to(self.device),
                }
                outputs = self.model.forward_instance(image, prompts)
                mask_logits = self.model.resize_predictions(
                    outputs["mask_logits"], instance_mask.shape[-2:]
                )

                batch_size = image.size(0)
                for i in range(batch_size):
                    iou = calculate_instance_iou(
                        mask_logits.squeeze(1)[i], instance_mask[i]
                    )
                    dice = calculate_instance_dice(
                        mask_logits.squeeze(1)[i], instance_mask[i]
                    )
                    total_iou += iou
                    total_dice += dice

                acc = calculate_classification_accuracy(
                    outputs["class_logits"], class_id
                )
                total_acc += acc * batch_size
                num_samples += batch_size

        return {
            'val_miou': total_iou / num_samples,
            'val_dice': total_dice / num_samples,
            'val_acc': total_acc / num_samples,
        }

    def _log_batch(self, batch_idx: int, losses: Dict[str, torch.Tensor], num_batches: int):
        """Log batch statistics"""
        if self.config.system.use_wandb:
            import wandb
            log_dict = {'train/batch_loss': losses['total_loss'].item()}
            if 'dice_loss' in losses:
                log_dict['train/batch_dice_loss'] = losses['dice_loss'].item()
                log_dict['train/batch_bce_loss'] = losses['bce_loss'].item()
            if 'mask_loss' in losses:
                log_dict['train/batch_mask_loss'] = losses['mask_loss'].item()
            if 'classification_loss' in losses:
                log_dict['train/batch_classification_loss'] = losses['classification_loss'].item()
            log_dict['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
    
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