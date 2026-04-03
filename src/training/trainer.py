"""
Training Pipeline for SAM with LoRA Fine-tuning

This module implements the training loop for fine-tuning SAM on FoodInsSeg
with LoRA adapters and specified hyperparameters. Optional FoodSeg103 validation
uses binary mask metrics only (no ingredient labels).
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import tqdm
from datetime import datetime
import json

from ..models.sam_lora import SAMLoRAModel
from configs.config import Config
from ..utils.metrics import (
    calculate_miou,
    calculate_dice,
    calculate_classification_accuracy,
    calculate_instance_iou,
    calculate_instance_dice,
)
from .losses import JointIngredientLoss


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
        
        self.criterion = JointIngredientLoss(
            mask_weight=config.training.mask_loss_weight,
            class_weight=config.training.classification_loss_weight,
        )
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Initialize scaler for mixed precision
        if config.training.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler('cuda', enabled=True)
        else:
            self.scaler = GradScaler('cpu', enabled=False)
        
        # Initialize data loaders (training: FoodInsSeg only)
        if config.data.dataset_name != "FoodInsSeg":
            raise ValueError(
                "Training targets FoodInsSeg only. "
                f"Got dataset_name={config.data.dataset_name!r}. "
                "Use --foodseg103_validation_path to validate on FoodSeg103, or eval with --dataset_name FoodSeg103."
            )

        from ..data.foodinsseg_dataset import FoodInsSegDataset
        from ..data.foodseg_dataset import FoodSeg103Dataset

        self._val_uses_foodseg103_binary = False
        train_dataset = FoodInsSegDataset(config, split="train")
        fv = getattr(config.data, "foodseg103_validation_path", None)
        if fv and str(fv).strip():
            p = Path(fv)
            if p.is_dir() and (p / "ImageSets" / "test.txt").is_file():
                val_dataset = FoodSeg103Dataset(config, split="val")
                self._val_uses_foodseg103_binary = True
            else:
                print(
                    f"Warning: foodseg103_validation_path={fv} is missing ImageSets/test.txt; "
                    "using FoodInsSeg validation split."
                )
                val_dataset = FoodInsSegDataset(config, split="val")
        else:
            val_dataset = FoodInsSegDataset(config, split="val")

        pin = config.system.device == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.system.num_workers,
            pin_memory=pin,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.system.num_workers,
            pin_memory=pin,
            drop_last=False,
        )

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
        """Train for one epoch (FoodInsSeg + ingredient head)."""
        return self._train_epoch_ingredient()

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch (FoodInsSeg metrics, or FoodSeg103 binary if configured)."""
        return self._validate_epoch_ingredient()

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

    def _validate_epoch_binary_foodseg103(self) -> Dict[str, float]:
        """Validation on FoodSeg103 (binary masks); mask IoU/Dice only — no ingredient accuracy."""
        self.model.eval()
        total_miou = 0.0
        total_dice = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.val_loader, desc="Validation [FoodSeg103 binary]"
            ):
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)

                image_features = self.model.image_encoder(image)
                prompts = {
                    "points": batch["prompts"]["points"].to(self.device),
                    "point_labels": batch["prompts"]["point_labels"].to(self.device),
                }
                pred_masks = self.model.sam_model(image_features, prompts)
                pred_masks = self.model.resize_predictions(pred_masks, mask.shape[-2:])
                pred_masks_sigmoid = torch.sigmoid(pred_masks.squeeze(1))

                miou = calculate_miou(pred_masks_sigmoid, mask)
                dice = calculate_dice(pred_masks_sigmoid, mask)

                total_miou += miou.item() * image.size(0)
                total_dice += dice.item() * image.size(0)
                num_samples += image.size(0)

        return {
            "val_miou": total_miou / num_samples,
            "val_dice": total_dice / num_samples,
            "val_acc": 0.0,
        }

    def _validate_epoch_ingredient(self) -> Dict[str, float]:
        """Validate for one epoch in ingredient mode."""
        if getattr(self, "_val_uses_foodseg103_binary", False):
            return self._validate_epoch_binary_foodseg103()

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