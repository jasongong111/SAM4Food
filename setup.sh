#!/bin/bash

# Setup script for SAM Food Segmentation Project

echo "Setting up SAM Food Segmentation Project..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Initialize conda for bash shell (if not already initialized)
eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -q "^sam4food "; then
    echo "⚠️  Conda environment 'sam4food' already exists."
    echo "Activating existing environment..."
    conda activate sam4food
    echo "✅ Environment activated"
    echo ""
    echo "To recreate the environment, run: conda env remove -n sam4food -y && bash setup.sh"
    echo ""
    echo "Continuing with existing environment..."
    # Continue with rest of setup (checkpoints, directories, etc.)
else
    # Create conda environment from environment.yml
    if [ -f "environment.yml" ]; then
        echo "Creating conda environment from environment.yml..."
        conda env create -f environment.yml
    else
        echo "❌ environment.yml not found. Creating environment manually..."
        conda create -n sam4food python=3.10 -y
        conda activate sam4food
        echo "Installing requirements..."
        pip install -r requirements.txt
    fi
    
    # Activate the environment
    echo "Activating conda environment..."
    conda activate sam4food
    
    echo "✅ Conda environment 'sam4food' is ready!"
fi

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
echo "To activate the conda environment in future sessions, run:"
echo "  conda activate sam4food"
echo ""
echo "Next steps:"
echo "1. Download FoodSeg103 dataset to data/FoodSeg103/"
echo "2. Run: python main.py --mode train --model_name vit_b --epochs 50"
echo ""
echo "For help: python main.py --help"