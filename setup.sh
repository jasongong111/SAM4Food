#!/bin/bash

# Setup script for SAM Food Segmentation Project (Poetry-managed environment)

set -e

echo "Setting up SAM Food Segmentation Project..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed."
    echo "   Install: curl -sSL https://install.python-poetry.org | python3 -"
    echo "   Docs: https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "✅ Poetry found: $(poetry --version)"

# Install dependencies from pyproject.toml / poetry.lock
echo ""
echo "📦 Installing dependencies (Poetry)..."
poetry install

echo ""
echo "✅ Virtual environment ready at .venv (in-project)"
echo "   Run commands with: poetry run python main.py ..."
echo "   Or open a shell:   poetry shell"

# SAM checkpoint download
echo ""
echo "📥 Downloading SAM ViT-B Model Checkpoint..."
if [ ! -f "sam_vit_b_01ec64.pth" ]; then
    curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo "✅ Download complete: sam_vit_b_01ec64.pth"
else
    echo "✅ Checkpoint already exists: sam_vit_b_01ec64.pth"
fi
echo ""
echo "   Using default path: ./sam_vit_b_01ec64.pth"
echo "   Example: poetry run python main.py --mode train --model_path sam_vit_b_01ec64.pth"
echo ""

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p visualizations
mkdir -p data

# Download LoRA checkpoint
echo ""
echo "📥 Downloading LoRA Best Model Checkpoint..."
if [ ! -f "checkpoints/best_model.pth" ]; then
    curl -L -o checkpoints/best_model.pth "https://huggingface.co/JasonGong111/SAM4Food/resolve/main/best_model.pth?download=true"
    echo "✅ Download complete: checkpoints/best_model.pth"
else
    echo "✅ Checkpoint already exists: checkpoints/best_model.pth"
fi
echo ""

# Set up environment variables (optional; poetry run inherits cwd)
echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Activate the venv in new shells:"
echo "  poetry shell"
echo "  # or: source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Training uses FoodInsSeg (auto-download if missing). Optional: clone FoodSeg103 to data/FoodSeg103 for validation (--foodseg103_validation_path)."
echo "2. Run: poetry run python main.py --mode train --sam_checkpoint sam_vit_b_01ec64.pth --epochs 50"
echo ""
echo "For help: poetry run python main.py --help"
