"""
Main Execution Script for SAM Food Segmentation Project

This script provides the complete pipeline for training and evaluating
the SAM model with LoRA adaptation on the FoodSeg103 dataset.
"""

import argparse
import torch
import json
import sys
from pathlib import Path

# Ensure root directory is in path for imports
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from configs.config import Config
from src.training.trainer import Trainer
from src.utils.visualization import Visualizer
from src.utils.metrics import evaluate_model, compare_with_baselines
from src.data.foodseg_dataset import download_foodseg103_dataset


def _resolve_latest_trained_checkpoint(config: Config) -> str | None:
    """Find the most relevant trained checkpoint for evaluation flows."""
    checkpoint_dir = Path(config.system.model_save_dir)
    best_checkpoint = checkpoint_dir / "best_model.pth"
    if best_checkpoint.exists():
        return str(best_checkpoint)

    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if epoch_checkpoints:
        return str(epoch_checkpoints[-1])

    return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM with LoRA for Food Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python main.py --mode train --model_name vit_b --sam_checkpoint sam_vit_b.pth --epochs 50

  # Continue training from checkpoint
  python main.py --mode train --sam_checkpoint sam_vit_b.pth --resume checkpoints/best_model.pth

  # Evaluate trained model
  python main.py --mode eval --sam_checkpoint sam_vit_b.pth --trained_checkpoint checkpoints/best_model.pth

  # Generate visualizations
  python main.py --mode visualize --sam_checkpoint sam_vit_b.pth --trained_checkpoint checkpoints/best_model.pth

  # Complete pipeline (train + eval + visualize)
  python main.py --mode full --model_name vit_b --sam_checkpoint sam_vit_b.pth --epochs 50
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['train', 'eval', 'visualize', 'full'],
        default='train',
        help='Execution mode'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_name',
        choices=['vit_b', 'vit_l', 'vit_h'],
        default='vit_b',
        help='SAM model architecture'
    )
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        help='Path to the base SAM checkpoint. Download from: https://dl.fbaipublicfiles.com/segment_anything/'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--trained_checkpoint',
        type=str,
        help='Path to a trained task checkpoint for evaluation or visualization'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to resume training from checkpoint'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for training'
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=8,
        help='LoRA rank for parameter efficient fine-tuning'
    )
    parser.add_argument(
        '--use_ingredient_head',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable ingredient classification head'
    )
    parser.add_argument(
        '--num_ingredient_classes',
        type=int,
        default=None,
        help='Number of ingredient classes for ingredient mode'
    )
    parser.add_argument(
        '--ingredient_head_hidden_dim',
        type=int,
        default=None,
        help='Hidden dimension for the ingredient classification head'
    )
    parser.add_argument(
        '--classification_loss_weight',
        type=float,
        default=None,
        help='Loss weight for ingredient classification'
    )
    parser.add_argument(
        '--mask_loss_weight',
        type=float,
        default=None,
        help='Loss weight for prompted mask supervision'
    )
    
    # Data configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Dataset identifier to use (e.g. FoodSeg103 or FoodInsSeg)'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to dataset root'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='Input image size for SAM'
    )
    
    # System configuration
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto/cuda/cpu)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with limited data'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    # Output configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--use_prompted_aggregation',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable prompted aggregation for ingredient-mode inference'
    )
    parser.add_argument(
        '--aggregation_score_threshold',
        type=float,
        default=None,
        help='Minimum score threshold for aggregation candidates'
    )
    parser.add_argument(
        '--aggregation_iou_threshold',
        type=float,
        default=None,
        help='IoU threshold used to merge prompted predictions'
    )
    parser.add_argument(
        '--max_prompts_per_image',
        type=int,
        default=None,
        help='Maximum prompted regions to aggregate per image'
    )
    parser.add_argument(
        '--visualization_samples',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )

    args = parser.parse_args()
    if args.sam_checkpoint is None and args.model_path is not None:
        args.sam_checkpoint = args.model_path

    if (
        args.trained_checkpoint is None
        and args.model_path is not None
        and args.mode in {'eval', 'visualize'}
    ):
        args.trained_checkpoint = args.model_path

    if not args.sam_checkpoint:
        parser.error('--sam_checkpoint is required (or use legacy --model_path)')
    if args.mode in {'eval', 'visualize'} and not args.trained_checkpoint:
        parser.error('--trained_checkpoint is required for eval and visualize modes')

    return args


def setup_config(args) -> Config:
    """Set up configuration from arguments"""
    config = Config()
    
    # Update model configuration
    config.model.sam_model_name = args.model_name
    if args.sam_checkpoint:
        config.model.sam_checkpoint_path = args.sam_checkpoint
    if args.trained_checkpoint:
        config.model.trained_checkpoint_path = args.trained_checkpoint
    config.model.lora_rank = args.lora_rank
    if args.use_ingredient_head is not None:
        config.model.use_ingredient_head = args.use_ingredient_head
    if args.num_ingredient_classes is not None:
        config.model.num_ingredient_classes = args.num_ingredient_classes
    if args.ingredient_head_hidden_dim is not None:
        config.model.ingredient_head_hidden_dim = args.ingredient_head_hidden_dim
    
    # Update training configuration
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    if args.classification_loss_weight is not None:
        config.training.classification_loss_weight = args.classification_loss_weight
    if args.mask_loss_weight is not None:
        config.training.mask_loss_weight = args.mask_loss_weight
    
    # Update data configuration
    if args.dataset_name:
        config.data.dataset_name = args.dataset_name
    config.data.dataset_path = args.dataset_path
    config.data.input_size = args.image_size
    
    # Update system configuration
    if args.device == 'auto':
        config.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.system.device = args.device
    
    config.system.num_workers = args.num_workers
    config.system.debug = args.debug
    config.system.use_wandb = not args.no_wandb
    config.system.output_dir = args.output_dir
    
    # Update evaluation configuration
    config.evaluation.num_visualizations = args.visualization_samples

    # Update inference configuration
    if args.use_prompted_aggregation is not None:
        config.inference.use_prompted_aggregation = args.use_prompted_aggregation
    if args.aggregation_score_threshold is not None:
        config.inference.aggregation_score_threshold = args.aggregation_score_threshold
    if args.aggregation_iou_threshold is not None:
        config.inference.aggregation_iou_threshold = args.aggregation_iou_threshold
    if args.max_prompts_per_image is not None:
        config.inference.max_prompts_per_image = args.max_prompts_per_image
    
    return config


def setup_wandb(config, args):
    """Setup Weights & Biases logging"""
    if not config.system.use_wandb:
        return
    
    try:
        import wandb
        wandb.init(
            project=config.system.experiment_name,
            config={
                'model_name': config.model.sam_model_name,
                'lora_rank': config.model.lora_rank,
                'epochs': config.training.num_epochs,
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'device': config.system.device,
                'debug': config.system.debug
            },
            tags=['sam', 'food-segmentation', 'lora', 'fine-tuning']
        )
        print("W&B logging initialized")
    except ImportError:
        print("W&B not available, skipping logging")
    except Exception as e:
        print(f"Failed to initialize W&B: {e}")
        config.system.use_wandb = False


def download_dataset_if_needed(config, args):
    """Download dataset if not available"""
    if config.data.dataset_path is None:
        if config.data.dataset_name == "FoodSeg103":
            dataset_path = download_foodseg103_dataset()
            config.data.dataset_path = str(dataset_path)
        elif config.data.dataset_name == "FoodInsSeg":
            config.data.dataset_path = "data/FoodInsSeg"
        else:
            raise ValueError(
                f"Dataset path must be provided for dataset '{config.data.dataset_name}'"
            )


def train_model(config, args):
    """Train the model"""
    print("Starting training...")
    
    # Download dataset if needed
    download_dataset_if_needed(config, args)
    
    # Setup training
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    history = trainer.train()
    
    # Save training history
    history_path = Path(config.system.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training completed! History saved to {history_path}")
    
    return trainer, history


def evaluate_model_script(config, args):
    """Evaluate a trained model"""
    print("Starting evaluation...")
    
    # Load model
    from src.models.sam_lora import SAMLoRAModel
    model = SAMLoRAModel(config).to(config.system.device)
    
    if args.trained_checkpoint:
        print(f"Loading trained model from: {args.trained_checkpoint}")
        checkpoint = torch.load(args.trained_checkpoint, map_location=config.system.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    from src.data.foodseg_dataset import create_data_loaders
    _, val_loader = create_data_loaders(config)
    
    # Evaluate model
    results = evaluate_model(model, val_loader, config, config.system.device)
    
    # Save results
    results_path = Path(config.system.output_dir) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"Results saved to {results_path}")
    
    return results


def visualize_results(config, args):
    """Generate visualizations"""
    print("Starting visualization...")
    
    # Load model
    from src.models.sam_lora import SAMLoRAModel
    model = SAMLoRAModel(config).to(config.system.device)
    
    if args.trained_checkpoint:
        print(f"Loading trained model from: {args.trained_checkpoint}")
        checkpoint = torch.load(args.trained_checkpoint, map_location=config.system.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    from src.data.foodseg_dataset import create_data_loaders
    _, val_loader = create_data_loaders(config)
    
    # Create visualizer
    visualizer = Visualizer(config, config.system.visualization_dir)
    
    # Generate visualizations
    viz_dir = visualizer.visualize_predictions(
        model, val_loader, config.system.device, 
        num_samples=config.evaluation.num_visualizations
    )
    
    print(f"Visualizations saved to: {viz_dir}")
    
    return viz_dir


def full_pipeline(config, args):
    """Run the complete pipeline"""
    print("Starting full pipeline: Train + Evaluate + Visualize")
    
    # Train model
    trainer, history = train_model(config, args)

    trained_checkpoint = args.trained_checkpoint or _resolve_latest_trained_checkpoint(config)
    if trained_checkpoint is None:
        raise ValueError("Full mode requires a trained checkpoint after training, but none was found")

    args.trained_checkpoint = trained_checkpoint
    config.model.trained_checkpoint_path = trained_checkpoint
    
    # Evaluate model
    results = evaluate_model_script(config, args)
    
    # Generate visualizations
    viz_dir = visualize_results(config, args)
    
    # Generate comprehensive report
    visualizer = Visualizer(config, config.system.visualization_dir)
    
    # Load training history for analysis
    history_path = Path(config.system.output_dir) / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        visualizer.create_training_analysis_plots(history)
    
    # Generate evaluation report
    comparison = {
        'current_model': results,
        'model_info': {
            'trainable_params': sum(p.numel() for p in trainer.model.get_trainable_parameters()),
            'total_params': sum(p.numel() for p in trainer.model.parameters()),
            'trainable_percentage': (sum(p.numel() for p in trainer.model.get_trainable_parameters()) / 
                                   sum(p.numel() for p in trainer.model.parameters())) * 100
        }
    }
    
    report = visualizer.generate_evaluation_report({'current_model': results}, comparison)
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print(f"✅ Training completed: {len(history['train_history'])} epochs")
    print(f"✅ Best mIoU: {trainer.best_miou:.4f}")
    print(f"✅ Target achieved: {'Yes' if trainer.best_miou >= 0.5 else 'No'}")
    print(f"✅ Visualizations: {viz_dir}")
    print(f"✅ Report: {config.system.visualization_dir}/evaluation_report.md")
    
    return {
        'trainer': trainer,
        'history': history,
        'results': results,
        'visualizations': viz_dir
    }


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Set up configuration
    config = setup_config(args)
    
    print("="*60)
    print("SAM FOOD SEGMENTATION - FINE-TUNING WITH LORA")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Model: {config.model.sam_model_name}")
    print(f"Device: {config.system.device}")
    print(f"Output directory: {config.system.output_dir}")
    print("="*60)
    
    # Setup W&B logging
    setup_wandb(config, args)
    
    try:
        if args.mode == 'train':
            train_model(config, args)
        elif args.mode == 'eval':
            evaluate_model_script(config, args)
        elif args.mode == 'visualize':
            visualize_results(config, args)
        elif args.mode == 'full':
            full_pipeline(config, args)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if config.system.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    main()