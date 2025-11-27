import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from configs.config import Config
from src.models.sam_lora import SAMLoRAModel
from segment_anything import SamPredictor

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ## 1. Configuration and Model Setup

# Initialize config
config = Config()

# Set model paths
# TODO: Set the path to your base SAM checkpoint (required)
# Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
config.model.sam_checkpoint_path = "sam_vit_b_01ec64.pth" 

# TODO: Set the path to your trained LoRA checkpoint (optional)
lora_checkpoint_path = "checkpoints/best_model.pth"

config.system.device = device

def load_model(config, lora_checkpoint_path=None):
    # Initialize model structure
    model = SAMLoRAModel(config)
    model.to(device)
    
    # Load LoRA weights if provided
    if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
        print(f"Loading LoRA checkpoint from {lora_checkpoint_path}...")
        checkpoint = torch.load(lora_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("LoRA checkpoint loaded successfully.")
    else:
        print("No LoRA checkpoint found or provided. Using base SAM model.")
    
    # PATCH: Replace the original SAM components with the LoRA-adapted ones
    # This allows us to use the standard SamPredictor with our LoRA model
    model._original_sam_model.mask_decoder = model.mask_decoder
    model._original_sam_model.prompt_encoder = model.prompt_encoder
    
    return model

# Create model instance
try:
    model_wrapper = load_model(config, lora_checkpoint_path)
    
    # Use the standard SamPredictor with the patched internal model
    predictor = SamPredictor(model_wrapper._original_sam_model)
    print("Model and Predictor initialized successfully!")
except Exception as e:
    print(f"Error initializing model: {e}")
    print("Please ensure you have downloaded the base SAM checkpoint.")
    predictor = None

# ## 2. Load and Prepare Image

def show_image(image, points=None, labels=None, box=None, mask=None, ax=None, title=None):
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    
    ax.imshow(image)
    
    if mask is not None:
        show_mask(mask, ax)
        
    if points is not None:
        pos_points = points[labels==1]
        neg_points = points[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=200, edgecolor='white', linewidth=1.25, label='Positive')
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=200, edgecolor='white', linewidth=1.25, label='Negative')
        
    if box is not None:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2, label='Box'))
        
    if title:
        ax.set_title(title)
    ax.axis('on')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# ## 3. Run Inference on Randomly Selected Images from UECFOODPIX Dataset

# Dataset configuration
dataset_root = "UECFOODPIX/data/UECFoodPIX"
train_img_dir = os.path.join(dataset_root, "train", "img")
test_img_dir = os.path.join(dataset_root, "test", "img")

# Create output directory for inference results
output_dir = "inference_outputs"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Number of random images to select
num_random_images = 5

# Collect all image paths from train and test directories
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
all_image_paths = []

print("Collecting images from UECFOODPIX dataset...")

# Collect from train directory
if os.path.exists(train_img_dir):
    train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) 
                    if os.path.splitext(f)[1].lower() in valid_extensions]
    all_image_paths.extend(train_images)
    print(f"Found {len(train_images)} images in train set")

# Collect from test directory
if os.path.exists(test_img_dir):
    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    all_image_paths.extend(test_images)
    print(f"Found {len(test_images)} images in test set")

if not all_image_paths:
    print(f"No images found in dataset directories.")
    print(f"Train directory: {train_img_dir}")
    print(f"Test directory: {test_img_dir}")
    selected_images = []
else:
    print(f"Total images available: {len(all_image_paths)}")
    
    # Randomly select images
    num_to_select = min(num_random_images, len(all_image_paths))
    selected_images = random.sample(all_image_paths, num_to_select)
    print(f"Randomly selected {num_to_select} images for inference")

# Run inference for each selected image
if predictor is not None and selected_images:
    for image_path in selected_images:
        print(f"\nProcessing: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image filename for output naming
        image_filename = os.path.basename(image_path)
        file_stem = os.path.splitext(image_filename)[0]
        
        # Save a copy of the original image
        original_output_path = os.path.join(output_dir, f"original_{image_filename}")
        # Convert back to BGR for saving with cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_output_path, image_bgr)
        print(f"Saved original image to {original_output_path}")
        
        predictor.set_image(image)
        
        # Define prompts (Center point)
        h, w = image.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1]) # 1 indicates a foreground point
        
        print(f"Prompts: Point {input_point}, Label {input_label}")
        
        # Predict
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False # SAM can output multiple masks for a single prompt
        )
        
        print(f"Generated {len(masks)} masks for {image_filename}")
        
        # ## 4. Visualize Results
        
        # Visualize masks
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            show_image(image, input_point, input_label, title=f"{image_filename} - Mask {i+1}, Score: {score:.3f}")
            show_mask(mask, plt.gca())
            
            # Save predicted result with mask overlay
            output_filename = f"prediction_{file_stem}_{i+1}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            plt.savefig(output_path)
            print(f"Saved prediction result to {output_path}")
            plt.close() 
elif predictor is None:
    print("Predictor not initialized, skipping inference.")
else:
    print("No images selected for inference.")

