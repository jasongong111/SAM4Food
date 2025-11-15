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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM with LoRA for Food Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python main.py --mode train --model_name vit_b --epochs 50

  # Continue training from checkpoint
  python main.py --mode train --resume checkpoints/best_model.pth

  # Evaluate trained model
  python main.py --mode eval --model_path checkpoints/best_model.pth

  # Generate visualizations
  python main.py --mode visualize --model_path checkpoints/best_model.pth

  # Complete pipeline (train + eval + visualize)
  python main.py --mode full --model_name vit_b --epochs 50
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
        '--model_path',
        type=str,
        required=True,
        help='Path to SAM model checkpoint (required). Download from: https://dl.fbaipublicfiles.com/segment_anything/'
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
    
    # Data configuration
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to FoodSeg103 dataset'
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
        '--visualization_samples',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )
    
    return parser.parse_args()


def setup_config(args) -> Config:
    """Set up configuration from arguments"""
    config = Config()
    
    # Update model configuration
    config.model.sam_model_name = args.model_name
    if args.model_path:
        config.model.sam_checkpoint_path = args.model_path
    config.model.lora_rank = args.lora_rank
    
    # Update training configuration
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    # Update data configuration
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
        dataset_path = download_foodseg103_dataset()
        config.data.dataset_path = str(dataset_path)


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
    
    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=config.system.device)
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
    
    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=config.system.device)
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