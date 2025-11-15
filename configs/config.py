"""
Configuration file for SAM Food Segmentation Project
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for SAM model with LoRA adaptation"""
    # SAM model settings
    sam_model_name: str = "vit_b"  # Options: vit_b, vit_l, vit_h
    sam_checkpoint_path: Optional[str] = None  # Will be downloaded automatically
    
    # LoRA settings
    lora_rank: int = 8  # Rank for LoRA adapters
    lora_alpha: int = 32  # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.1  # Dropout rate for LoRA layers
    target_modules: list = None  # Target modules for LoRA injection
    
    def __post_init__(self):
        if self.target_modules is None:
            # Target the mask decoder and prompt encoder components
            self.target_modules = [
                "mask_decoder.transformer.layers",
                "prompt_encoder.mask_tokens",
                "prompt_encoder.output_tokens",
                "mask_decoder.output_upscaling"
            ]

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Basic training settings
    num_epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Optimizer settings
    optimizer: str = "adam"
    scheduler: str = "cosine"  # Options: cosine, linear, step, None
    
    # Loss function settings
    dice_weight: float = 0.5  # Weight for Dice loss (BCE weight will be 1-dice_weight)
    
    # Checkpoint and logging
    save_frequency: int = 5  # Save checkpoint every N epochs
    eval_frequency: int = 2  # Evaluate every N epochs
    log_frequency: int = 100  # Log every N iterations
    
    # Mixed precision training
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

@dataclass
class DataConfig:
    """Configuration for dataset and data loading"""
    # Dataset paths
    dataset_name: str = "FoodSeg103"
    dataset_path: Optional[str] = None  # Will be downloaded automatically
    
    # Split information (as specified in the project)
    train_size: int = 4983
    val_size: int = 2135
    test_size: int = 0  # Not specified in the proposal
    
    # Data preprocessing
    image_size: int = 1024  # SAM's default image size
    input_size: int = 1024  # Resize images to this size
    mean: tuple = (0.485, 0.456, 0.406)  # ImageNet statistics
    std: tuple = (0.229, 0.224, 0.225)  # ImageNet statistics
    
    # Data augmentation
    use_augmentation: bool = True
    color_jitter: float = 0.1
    horizontal_flip: float = 0.5
    
    # Prompt generation
    num_point_prompts: int = 5  # Number of point prompts per mask
    num_boxes_per_mask: int = 1  # Number of box prompts per mask

@dataclass
class SystemConfig:
    """Configuration for system settings"""
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Output directories
    output_dir: str = "outputs"
    model_save_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    visualization_dir: str = "visualizations"
    
    # Random seed
    seed: int = 42
    
    # Experiment tracking
    use_wandb: bool = True
    experiment_name: str = "sam_food_segmentation"
    
    # Debug settings
    debug: bool = False
    max_samples_for_debug: int = 100

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Metrics to compute
    compute_miou: bool = True
    compute_dice: bool = True
    compute_f1: bool = True
    
    # Comparison models
    compare_with_baseline_sam: bool = True
    compare_with_foodsam: bool = True
    
    # Visualization settings
    save_predictions: bool = True
    save_visualizations: bool = True
    num_visualizations: int = 50  # Save visualization for first N samples
    
    # IoU thresholds for evaluation
    iou_threshold_range: tuple = (0.5, 0.95)  # mIoU from 0.5 to 0.95
    iou_threshold_step: float = 0.05

class Config:
    """Main configuration class that combines all configs"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        self.evaluation = EvaluationConfig()
        
        # Create output directories
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.model_save_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
        os.makedirs(self.system.results_dir, exist_ok=True)
        os.makedirs(self.system.visualization_dir, exist_ok=True)

# Import torch at module level for device detection
import torch