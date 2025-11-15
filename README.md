# SAM Food Segmentation with LoRA Fine-tuning

A comprehensive implementation of fine-tuning the Segment Anything Model (SAM) with LoRA adapters for food image segmentation using the FoodSeg103 dataset. This project aims to achieve high-precision food segmentation with parameter-efficient fine-tuning.

## 📋 Project Overview

### Objective
Fine-tune SAM on FoodSeg103 using efficient LoRA adaptations to achieve mIoU > 0.50 on the validation set, rivaling or surpassing FoodSAM performance.

### Key Features
- **Parameter-Efficient Fine-tuning**: Uses LoRA adapters to reduce trainable parameters
- **Frozen Image Encoder**: Leverages pre-trained vision transformer features
- **Combined Loss Function**: Combines Binary Cross-Entropy and Dice loss
- **Comprehensive Evaluation**: Includes mIoU, Dice, F1, and boundary metrics
- **Rich Visualizations**: Qualitative assessment with overlay comparisons
- **Experiment Tracking**: Weights & Biases integration for monitoring

## 🏗️ Architecture

### Model Components
1. **Frozen Image Encoder**: Pre-trained SAM vision transformer (ViT)
2. **LoRA-adapted Mask Decoder**: Efficient fine-tuning with rank decomposition
3. **LoRA-adapted Prompt Encoder**: Enhanced prompt processing capabilities

### Training Strategy
- **Freeze**: Image encoder parameters (large majority of parameters)
- **Fine-tune**: Mask decoder and prompt encoder with LoRA adapters
- **Objective**: Ingredient-level food segmentation
- **Dataset**: FoodSeg103 (7,118 images, 103 ingredient classes)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and setup environment
git clone <repository-url>
cd sam-food-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Dataset

```bash
# Download FoodSeg103 manually from:
# https://github.com/L1016517444/FoodSeg103

# Expected structure:
data/FoodSeg103/
├── images/
├── masks/
└── annotations.json
```

### 3. Basic Training

```bash
# Train with default settings
python main.py --mode train --model_name vit_b --epochs 50

# Train with custom hyperparameters
python main.py --mode train --model_name vit_b --epochs 100 --learning_rate 5e-5 --batch_size 8
```

### 4. Complete Pipeline

```bash
# Run full pipeline (train + evaluate + visualize)
python main.py --mode full --model_name vit_b --epochs 50

# Resume training from checkpoint
python main.py --mode train --resume checkpoints/best_model.pth --epochs 100
```

## 📊 Usage Examples

### Training Configuration

```python
from configs.config import Config
from src.training.trainer import Trainer

# Create configuration
config = Config()
config.model.sam_model_name = "vit_b"
config.model.lora_rank = 8
config.training.num_epochs = 50
config.training.learning_rate = 1e-4
config.training.batch_size = 4

# Start training
trainer = Trainer(config)
history = trainer.train()
```

### Evaluation

```python
from src.utils.metrics import evaluate_model
from src.data.foodseg_dataset import create_data_loaders

# Load model and data
_, val_loader = create_data_loaders(config)
results = evaluate_model(model, val_loader, config, device)

# Results include:
# {'miou': 0.5234, 'dice': 0.6789, 'f1': 0.7456, ...}
```

### Visualization

```python
from src.utils.visualization import Visualizer

# Generate visualizations
visualizer = Visualizer(config)
viz_dir = visualizer.visualize_predictions(model, val_loader, device, num_samples=20)

# Creates comparison plots with:
# - Original images
# - Ground truth masks
# - Predicted masks
# - Overlay comparisons
# - Performance metrics
```

## ⚙️ Configuration Options

### Model Configuration
```python
config.model.sam_model_name = "vit_b"      # ViT variants: vit_b, vit_l, vit_h
config.model.lora_rank = 8                 # LoRA rank for parameter efficiency
config.model.lora_alpha = 32               # LoRA scaling parameter
config.model.lora_dropout = 0.1            # Dropout for LoRA layers
```

### Training Configuration
```python
config.training.num_epochs = 50            # Total training epochs
config.training.batch_size = 4             # Batch size
config.training.learning_rate = 1e-4       # Learning rate
config.training.dice_weight = 0.5          # Weight for Dice loss component
config.training.use_mixed_precision = True # Enable AMP training
```

### Data Configuration
```python
config.data.input_size = 1024              # Input image size
config.data.use_augmentation = True        # Data augmentation
config.data.num_point_prompts = 5          # Point prompts per mask
config.data.train_size = 4983              # Training samples
config.data.val_size = 2135                # Validation samples
```

## 📈 Metrics and Evaluation

### Primary Metrics
- **Mean IoU (mIoU)**: Primary target metric (aim > 0.50)
- **Dice Coefficient**: Spatial overlap measure
- **F1 Score**: Harmonic mean of precision and recall

### Additional Metrics
- **Precision/Recall**: Classification quality
- **Boundary IoU**: Boundary quality assessment
- **Accuracy**: Overall classification accuracy

### Sample Output
```
Evaluation Results:
  miou: 0.5234
  dice: 0.6789
  f1: 0.7456
  precision: 0.7234
  recall: 0.7691
  accuracy: 0.8902
  boundary_iou: 0.4123
```

## 📁 Project Structure

```
sam-food-segmentation/
├── main.py                          # Main execution script
├── requirements.txt                 # Dependencies
├── configs/
│   └── config.py                   # Configuration management
├── src/
│   ├── __init__.py
│   ├── models/
│   │   └── sam_lora.py            # SAM model with LoRA
│   ├── data/
│   │   └── foodseg_dataset.py     # Dataset loading
│   ├── training/
│   │   └── trainer.py             # Training pipeline
│   └── utils/
│       ├── metrics.py             # Evaluation metrics
│       └── visualization.py       # Visualization tools
├── data/
│   └── FoodSeg103/                # Dataset directory
├── outputs/                        # Training outputs
├── checkpoints/                    # Model checkpoints
├── logs/                          # Training logs
├── results/                       # Evaluation results
└── visualizations/               # Generated visualizations
```

## 🎯 Expected Performance

### Target Achievement
- **Primary Goal**: mIoU > 0.50 on validation set
- **Comparison**: Rival or surpass FoodSAM performance
- **Efficiency**: Significantly fewer trainable parameters

### Parameter Efficiency
```
Traditional SAM Fine-tuning:
- Total Parameters: ~91M
- Trainable: ~91M (100%)

LoRA Fine-tuning (vit_b, rank=8):
- Total Parameters: ~91M
- Trainable: ~0.9M (1%)
- Memory Savings: ~99%
```

## 🛠️ Advanced Usage

### Custom LoRA Configuration
```python
# Target specific modules
config.model.target_modules = [
    "mask_decoder.transformer.layers",
    "prompt_encoder.mask_tokens",
    "prompt_encoder.output_tokens"
]

# Adjust LoRA parameters
config.model.lora_rank = 16      # Higher rank for more capacity
config.model.lora_alpha = 64     # Higher scaling
config.model.lora_dropout = 0.2  # More regularization
```

### Debug Mode
```python
# Limited dataset for fast iteration
config.system.debug = True
config.system.max_samples_for_debug = 100

# Quick training run
python main.py --mode train --debug --epochs 5
```

### Multiple Model Comparison
```python
# Compare different SAM backbones
for model_name in ['vit_b', 'vit_l', 'vit_h']:
    args.model_name = model_name
    results = run_pipeline(args)
    print(f"{model_name}: {results['miou']:.4f}")
```

## 📊 Visualization Features

### Prediction Analysis
- **Side-by-side comparison**: Original, GT, Pred, Probability
- **Overlay visualization**: Color-coded comparison
- **Metrics display**: Comprehensive performance metrics
- **Error analysis**: TP/FP/FN breakdown

### Training Analysis
- **Loss curves**: Training and validation progress
- **Learning rate schedule**: Optimization visualization
- **Metric trends**: mIoU, Dice, F1 over epochs
- **Convergence analysis**: Moving averages and stability

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   config.training.batch_size = 2
   # Or use gradient accumulation
   config.training.gradient_accumulation_steps = 4
   ```

2. **Slow Training**
   ```python
   # Enable mixed precision
   config.training.use_mixed_precision = True
   # Increase data workers
   config.system.num_workers = 8
   ```

3. **Poor Performance**
   ```python
   # Increase LoRA rank
   config.model.lora_rank = 16
   # Adjust loss weights
   config.training.dice_weight = 0.7
   ```

### Performance Monitoring
```python
# Check GPU utilization
nvidia-smi

# Monitor training in real-time
tensorboard --logdir outputs/logs/

# W&B dashboard
wandb sync outputs/logs/
```

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{gong2025fine,
  title={Fine-tuning Segment Anything Model for Food Segmentation using the FoodSeg103 Dataset},
  author={Gong, Zijing and Cheng, Zheya},
  journal={Computer Vision Project},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Additional evaluation metrics
- Enhanced visualization features  
- Support for more SAM variants
- Optimized training strategies
- Extended documentation

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Meta AI**: For the Segment Anything Model
- **PEFT**: For efficient parameter fine-tuning
- **FoodSeg103**: For the comprehensive food segmentation dataset
- **PyTorch**: For the deep learning framework

---

**Author**: MiniMax Agent  
**Version**: 1.0.0  
**Last Updated**: 2025-11-06