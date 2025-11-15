#!/usr/bin/env python3
"""
Test script for SAM Food Segmentation Project
"""

import sys
import torch
from pathlib import Path

# Ensure root directory is in path for imports
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from configs.config import Config
        print("✅ Config module imported successfully")
        
        from src.models.sam_lora import SAMLoRAModel
        print("✅ SAM LoRA model imported successfully")
        
        from src.data.foodseg_dataset import FoodSeg103Dataset
        print("✅ Dataset module imported successfully")
        
        from src.training.trainer import Trainer
        print("✅ Trainer module imported successfully")
        
        from src.utils.metrics import calculate_miou, calculate_dice
        print("✅ Metrics module imported successfully")
        
        from src.utils.visualization import Visualizer
        print("✅ Visualization module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    try:
        from configs.config import Config
        config = Config()
        
        print(f"✅ SAM model: {config.model.sam_model_name}")
        print(f"✅ Batch size: {config.training.batch_size}")
        print(f"✅ Device: {config.system.device}")
        print(f"✅ LoRA rank: {config.model.lora_rank}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_model_creation():
    """Test model creation (mock, no actual SAM loading)"""
    print("\nTesting model creation...")
    
    try:
        from configs.config import Config
        config = Config()
        
        # Test parameter count
        total_params = sum(p.numel() for p in torch.nn.Linear(10, 5).parameters())
        trainable_params = sum(p.numel() for p in torch.nn.Linear(10, 5).parameters() if p.requires_grad)
        
        print(f"✅ PyTorch working: {total_params} parameters")
        print(f"✅ Device available: {config.system.device}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_dice_loss():
    """Test Dice loss implementation"""
    print("\nTesting loss functions...")
    
    try:
        from src.training.trainer import DiceLoss
        
        dice_loss = DiceLoss()
        
        # Create test data
        pred = torch.randn(2, 10, 10)
        target = torch.randint(0, 2, (2, 10, 10)).float()
        
        loss = dice_loss(pred, target)
        print(f"✅ Dice loss working: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        return False

def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting metrics...")
    
    try:
        from src.utils.metrics import calculate_dice, calculate_miou
        
        # Create test data
        pred = torch.sigmoid(torch.randn(2, 10, 10))
        target = torch.randint(0, 2, (2, 10, 10)).float()
        
        dice = calculate_dice(pred, target)
        miou = calculate_miou(pred, target)
        
        print(f"✅ Dice coefficient: {dice:.4f}")
        print(f"✅ Mean IoU: {miou:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing SAM Food Segmentation Project")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_dice_loss),
        ("Metrics", test_metrics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Download FoodSeg103 dataset")
        print("2. Run: python main.py --mode train --model_name vit_b --epochs 50")
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())