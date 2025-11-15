"""
FoodSeg103 Dataset and Data Loading Pipeline

This module implements the data loading pipeline for the FoodSeg103 dataset
including preprocessing, data augmentation, and prompt generation.
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..configs.config import Config


class FoodSeg103Dataset(Dataset):
    """
    Dataset class for FoodSeg103 with support for SAM prompts
    """
    
    def __init__(self, config: Config, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Set up dataset paths
        self.dataset_path = self._setup_dataset_path()
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Set up data transforms
        self.transforms = self._setup_transforms()
        
        # Prepare prompt generation settings
        self.prompt_config = {
            'num_point_prompts': self.config.data.num_point_prompts,
            'num_boxes_per_mask': self.config.data.num_boxes_per_mask,
            'input_size': self.config.data.input_size
        }
        
        print(f"Loaded {len(self.annotations)} samples for {split} split")
        
    def _setup_dataset_path(self) -> Path:
        """Setup dataset download and path"""
        if self.config.data.dataset_path:
            dataset_path = Path(self.config.data.dataset_path)
        else:
            # Use default path
            dataset_path = Path("data/FoodSeg103")
            
        return dataset_path
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from the dataset"""
        annotations = []
        
        # For FoodSeg103, annotations are typically in JSON format
        annotation_file = self.dataset_path / "annotations.json"
        
        if not annotation_file.exists():
            print(f"Annotations not found at {annotation_file}")
            print("Please download the FoodSeg103 dataset manually.")
            print("Dataset can be found at: https://github.com/L1016517444/FoodSeg103")
            
            # Create a simple example for demonstration
            if self.config.system.debug:
                print("Creating mock annotations for debugging...")
                return self._create_mock_annotations()
            else:
                raise FileNotFoundError(f"Dataset annotations not found at {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Process annotations based on the split
        for item in data:
            if item.get('split') == self.split or (self.split == 'all' and item.get('split') in ['train', 'val']):
                annotations.append({
                    'image_path': self.dataset_path / 'images' / item['image'],
                    'mask_path': self.dataset_path / 'masks' / item['mask'],
                    'ingredients': item.get('ingredients', []),
                    'labels': item.get('labels', [])
                })
        
        return annotations
    
    def _create_mock_annotations(self) -> List[Dict]:
        """Create mock annotations for debugging purposes"""
        mock_annotations = []
        
        # Create simple examples with placeholder data
        for i in range(min(10, self.config.data.train_size if self.split == 'train' else self.config.data.val_size)):
            mock_annotations.append({
                'image_path': self.dataset_path / f'images' / f'sample_{i}.jpg',
                'mask_path': self.dataset_path / f'masks' / f'sample_{i}.png',
                'ingredients': [f'ingredient_{j}' for j in range(np.random.randint(1, 5))],
                'labels': list(range(np.random.randint(1, 5)))
            })
        
        return mock_annotations
    
    def _setup_transforms(self):
        """Setup data augmentation transforms"""
        if self.split == 'train' and self.config.data.use_augmentation:
            transforms = A.Compose([
                # Geometric transformations
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=self.config.data.horizontal_flip),
                A.VerticalFlip(p=0.3),
                
                # Color transformations
                A.ColorJitter(brightness=self.config.data.color_jitter,
                             contrast=self.config.data.color_jitter,
                             saturation=self.config.data.color_jitter,
                             hue=self.config.data.color_jitter/4,
                             p=0.7),
                
                # Blur and noise
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0)
                ], p=0.3),
                
                # Random resize and crop
                A.RandomResizedCrop(height=self.config.data.input_size,
                                  width=self.config.data.input_size,
                                  scale=(0.8, 1.0),
                                  p=0.5),
                
                # Normalization
                A.Normalize(mean=self.config.data.mean, std=self.config.data.std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            # Validation transforms (only resize and normalization)
            transforms = A.Compose([
                A.Resize(height=self.config.data.input_size, width=self.config.data.input_size),
                A.Normalize(mean=self.config.data.mean, std=self.config.data.std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        
        return transforms
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset"""
        annotation = self.annotations[idx]
        
        # Load image and mask
        image, mask = self._load_image_and_mask(annotation)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Generate prompts from mask
        prompts = self._generate_prompts(mask)
        
        # Prepare output
        sample = {
            'image': image,
            'mask': mask,
            'prompts': prompts,
            'metadata': {
                'image_path': str(annotation['image_path']),
                'ingredients': annotation.get('ingredients', []),
                'labels': annotation.get('labels', [])
            }
        }
        
        return sample
    
    def _load_image_and_mask(self, annotation: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask from file paths"""
        # Load image
        image_path = annotation['image_path']
        if not os.path.exists(image_path):
            # For debugging, create a simple colored image
            if self.config.system.debug:
                image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = annotation['mask_path']
        if not os.path.exists(mask_path):
            # For debugging, create a simple mask
            if self.config.system.debug:
                mask = np.random.randint(0, 2, (image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        
        return image, mask
    
    def _generate_prompts(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate prompts (points and boxes) from ground truth mask
        for SAM training
        """
        # Convert mask to numpy for processing
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        
        # Get binary mask (assuming multi-class mask, use first class for now)
        if len(mask_np.shape) == 3:
            binary_mask = (mask_np[0] > 0).astype(np.uint8)
        else:
            binary_mask = mask_np
        
        # Generate point prompts
        point_coords, point_labels = self._sample_points_from_mask(binary_mask)
        
        # Generate box prompt from mask bounding box
        bbox = self._get_mask_bbox(binary_mask)
        
        prompts = {
            'points': torch.tensor(point_coords, dtype=torch.float32),
            'point_labels': torch.tensor(point_labels, dtype=torch.int32),
            'bbox': torch.tensor(bbox, dtype=torch.float32) if bbox else None
        }
        
        return prompts
    
    def _sample_points_from_mask(self, mask: np.ndarray) -> Tuple[List[List[float]], List[int]]:
        """Sample positive and negative points from the mask"""
        num_points = self.prompt_config['num_point_prompts']
        
        # Find positive points (inside mask)
        positive_coords = np.column_stack(np.where(mask > 0))
        
        if len(positive_coords) == 0:
            # No mask pixels found, return empty prompts
            return [], []
        
        # Sample positive points
        num_positive = min(num_points // 2, len(positive_coords))
        if num_positive > 0:
            positive_indices = np.random.choice(len(positive_coords), num_positive, replace=False)
            positive_points = positive_coords[positive_indices]
        else:
            positive_points = np.array([]).reshape(0, 2)
        
        # Find negative points (outside mask)
        negative_coords = np.column_stack(np.where(mask == 0))
        
        # Sample negative points
        num_negative = num_points - num_positive
        if num_negative > 0 and len(negative_coords) > 0:
            negative_indices = np.random.choice(len(negative_coords), num_negative, replace=False)
            negative_points = negative_coords[negative_indices]
        else:
            negative_points = np.array([]).reshape(0, 2)
        
        # Combine points and labels
        all_points = []
        all_labels = []
        
        # Add positive points (label = 1)
        for coord in positive_points:
            all_points.append([coord[1], coord[0]])  # SAM expects (x, y) format
            all_labels.append(1)
        
        # Add negative points (label = 0)
        for coord in negative_points:
            all_points.append([coord[1], coord[0]])
            all_labels.append(0)
        
        return all_points, all_labels
    
    def _get_mask_bbox(self, mask: np.ndarray) -> Optional[List[float]]:
        """Get bounding box of the mask"""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Return in [x1, y1, x2, y2] format
        return [x, y, x + w, y + h]


def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Create datasets
    train_dataset = FoodSeg103Dataset(config, split='train')
    val_dataset = FoodSeg103Dataset(config, split='val')
    
    # Apply debugging limit if specified
    if config.system.debug and config.system.max_samples_for_debug > 0:
        train_dataset.annotations = train_dataset.annotations[:config.system.max_samples_for_debug]
        val_dataset.annotations = val_dataset.annotations[:min(len(val_dataset.annotations), 
                                                            config.system.max_samples_for_debug // 2)]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True if config.system.debug else True,  # Always shuffle training
        num_workers=config.system.num_workers,
        pin_memory=True if config.system.device == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True if config.system.device == 'cuda' else False,
        drop_last=False
    )
    
    return train_loader, val_loader


def download_foodseg103_dataset() -> Path:
    """
    Download the FoodSeg103 dataset
    This is a placeholder for the actual download process
    """
    import subprocess
    
    dataset_url = "https://github.com/L1016517444/FoodSeg103.git"
    dataset_path = Path("data/FoodSeg103")
    
    print("Downloading FoodSeg103 dataset...")
    print("Note: Please download the dataset manually from:")
    print("https://github.com/L1016517444/FoodSeg103")
    print(f"Expected location: {dataset_path}")
    
    # Try to clone if git repo
    try:
        subprocess.run(["git", "clone", dataset_url, str(dataset_path)], 
                      check=True, capture_output=True)
        print("Dataset downloaded successfully!")
    except subprocess.CalledProcessError:
        print("Automatic download failed. Please download manually.")
        
    return dataset_path