# SAM4Food: Food Segmentation with LoRA Fine-tuned SAM

Fine-tuning the Segment Anything Model (SAM) with LoRA adapters for food image segmentation using the FoodSeg103 dataset.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Quick Start](#quick-start)
5. [Training](#training)
6. [Inference](#inference)

---

## Prerequisites

- **Python**: 3.10+
- **CUDA**: 11.7+ (for GPU training, optional but recommended)
- **Conda**: Miniconda or Anaconda
- **GPU Memory**: 16GB+ recommended for training

---

## Installation

### Option 1: Automated Setup (Recommended)

Run the setup script which handles environment creation, checkpoint downloads, and directory setup:

```bash
git clone <repository-url>
cd SAM4Food

# Run automated setup
bash setup.sh
```

The script will:
- Create conda environment `sam4food`
- Download SAM ViT-B checkpoint (~375MB)
- Download pre-trained LoRA weights from HuggingFace
- Create required directories

After setup, activate the environment:
```bash
conda activate sam4food
```

### Option 2: Manual Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate sam4food

# Or use pip
pip install -r requirements.txt
```

### Download Model Checkpoints

**SAM Base Model** (required):
```bash
# ViT-B (~375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**Pre-trained LoRA Weights** (For inference without training):
```bash
mkdir -p checkpoints
curl -L -o checkpoints/best_model.pth \
  "https://huggingface.co/JasonGong111/SAM4Food/resolve/main/best_model.pth?download=true"
```

---

## Dataset Setup

### Download FoodSeg103

Download the dataset from: https://github.com/L1016517444/FoodSeg103

Extract and organize with this structure:

```
data/FoodSeg103/
├── Images/
│   ├── img_dir/
│   │   ├── train/        # Training images (.jpg)
│   │   └── test/         # Test images (.jpg)
│   └── ann_dir/
│       ├── train/        # Training masks (.png)
│       └── test/         # Test masks (.png)
├── ImageSets/
│   ├── train.txt         # Training image IDs
│   └── test.txt          # Test image IDs
└── category_id.txt       # Category definitions
```

---

## Quick Start

### Run Inference with Pre-trained Model

Use the pre-trained LoRA weights to segment food images immediately:

```bash
# Ensure you have both checkpoints:
# - sam_vit_b_01ec64.pth (SAM base model)
# - checkpoints/best_model.pth (LoRA weights)

python inference.py
```

Or use the Jupyter notebook for interactive inference:
```bash
jupyter notebook inference.ipynb
```

### Train a New Model

```bash
python main.py --mode train \
  --model_path sam_vit_b_01ec64.pth \
  --model_name vit_b \
  --epochs 50
```

---

## Training

### Basic Training Command

```bash
python main.py --mode train \
  --model_path sam_vit_b_01ec64.pth \
  --model_name vit_b \
  --epochs 50 \
  --batch_size 4 \
  --learning_rate 1e-4
```

### Training with Custom Settings

```bash
python main.py --mode train \
  --model_path sam_vit_b_01ec64.pth \
  --model_name vit_b \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --dataset_path /path/to/FoodSeg103
```

### Resume Training from Checkpoint

```bash
python main.py --mode train \
  --model_path sam_vit_b_01ec64.pth \
  --resume checkpoints/best_model.pth \
  --epochs 100
```

### Full Pipeline (Train + Evaluate + Visualize)

```bash
python main.py --mode full \
  --model_path sam_vit_b_01ec64.pth \
  --epochs 50
```

### All Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | Execution mode: `train`, `eval`, `visualize`, `full` |
| `--model_name` | `vit_b` | SAM architecture: `vit_b`, `vit_l`, `vit_h` |
| `--model_path` | Required | Path to SAM checkpoint |
| `--resume` | None | Path to resume training from |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--lora_rank` | 8 | LoRA adapter rank |
| `--dataset_path` | None | Custom dataset path |
| `--image_size` | 1024 | Input image size |
| `--device` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `--num_workers` | 4 | Data loading workers |
| `--debug` | False | Enable debug mode |
| `--no_wandb` | False | Disable W&B logging |
| `--output_dir` | `outputs` | Output directory |

---

## Inference

### Using Python Script

Edit `inference.py` to set your checkpoint paths:

```python
# Set the path to SAM base checkpoint
config.model.sam_checkpoint_path = "sam_vit_b_01ec64.pth"

# Set the path to trained LoRA checkpoint
lora_checkpoint_path = "checkpoints/best_model.pth"
```

Then run:
```bash
python inference.py
```

The notebook provides interactive inference with visualization.

---

## Results

### Training Curves

![Training Loss](assets/training.png)

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [FoodSeg103 Dataset](https://github.com/L1016517444/FoodSeg103)
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
