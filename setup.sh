#!/bin/bash

# Setup script for SAM Food Segmentation Project

echo "Setting up SAM Food Segmentation Project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python version $python_version is compatible"
else
    echo "❌ Python version $python_version is not compatible. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download SAM checkpoint if needed
echo "Checking SAM checkpoints..."
if [ ! -f "models/sam_vit_b_01ec64.pth" ]; then
    echo "Downloading SAM ViT-B checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/ || \
    curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o models/sam_vit_b_01ec64.pth
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p visualizations
mkdir -p data

# Set up environment variables
echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download FoodSeg103 dataset to data/FoodSeg103/"
echo "2. Run: python main.py --mode train --model_name vit_b --epochs 50"
echo ""
echo "For help: python main.py --help"