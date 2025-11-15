"""
SAM Model with LoRA Adaptation for Food Segmentation

This module implements the SAM model with LoRA adapters for efficient fine-tuning
on the FoodSeg103 dataset. The approach freezes the pre-trained image encoder
and adds LoRA adapters to the mask decoder and prompt encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

# Import SAM components
try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import PromptedSAM
    from segment_anything.predictor import SamPredictor
except ImportError:
    print("Installing SAM...")
    import subprocess
    subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"], 
                   check=True)
    from segment_anything import sam_model_registry
    from segment_anything.modeling import PromptedSAM
    from segment_anything.predictor import SamPredictor

# Import LoRA components
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Installing PEFT for LoRA...")
    import subprocess
    subprocess.run(["pip", "install", "peft"], check=True)
    from peft import LoraConfig, get_peft_model

from ..configs.config import Config


class SAMLoRAModel(nn.Module):
    """
    SAM model with LoRA adaptation for food segmentation.
    
    This implementation:
    1. Freezes the pre-trained image encoder
    2. Adds LoRA adapters to mask decoder and prompt encoder
    3. Maintains compatibility with original SAM API
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load base SAM model
        self._load_sam_model()
        
        # Configure LoRA for target modules
        self._setup_lora_adapters()
        
        # Freeze the image encoder
        self._freeze_image_encoder()
        
    def _load_sam_model(self):
        """Load the base SAM model"""
        model_type = self.config.model.sam_model_name
        
        # Load SAM model checkpoint
        sam_checkpoint = self.config.model.sam_checkpoint_path
        if sam_checkpoint is None:
            # Use the default SAM model from the registry
            if model_type == "vit_b":
                sam_checkpoint = "sam_vit_b_01ec64.pth"
            elif model_type == "vit_l":
                sam_checkpoint = "sam_vit_l_0b3195.pth"
            elif model_type == "vit_h":
                sam_checkpoint = "sam_vit_h_4b8939.pth"
            else:
                raise ValueError(f"Unsupported SAM model type: {model_type}")
        
        # Register and load the model
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        
        # Store components for later use
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        
        # Store original forward methods for potential reference
        self._original_forward = sam_model.forward
        
    def _setup_lora_adapters(self):
        """Set up LoRA adapters for the target modules"""
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config.model.lora_rank,
            lora_alpha=self.config.model.lora_alpha,
            target_modules=self.config.model.target_modules,
            lora_dropout=self.config.model.lora_dropout,
            bias="none",  # No bias modifications
        )
        
        # Apply LoRA to mask decoder
        self.mask_decoder = get_peft_model(self.mask_decoder, lora_config)
        
        # Apply LoRA to prompt encoder
        self.prompt_encoder = get_peft_model(self.prompt_encoder, lora_config)
        
        # Set the combined model for inference
        self._setup_combined_model()
        
    def _setup_combined_model(self):
        """Set up the combined SAM model with LoRA adapters"""
        # Create a combined model that uses the LoRA-adapted components
        class CombinedSAM(nn.Module):
            def __init__(self, image_encoder, prompt_encoder, mask_decoder):
                super().__init__()
                self.image_encoder = image_encoder
                self.prompt_encoder = prompt_encoder
                self.mask_decoder = mask_decoder
                
            def forward(self, image_embeddings, prompts):
                """
                Forward pass through the combined model
                
                Args:
                    image_embeddings: Image features from the frozen encoder
                    prompts: Prompt features (points, boxes, masks)
                
                Returns:
                    Predicted masks
                """
                # Get prompt embeddings
                sparse_embeddings, dense_embeddings = self.prompt_encoder(points=prompts['points'], 
                                                                        boxes=prompts.get('boxes'),
                                                                        masks=prompts.get('masks'))
                
                # Predict masks using the decoder
                pred_masks = self.mask_decoder(image_embeddings, 
                                             sparse_prompt_embeddings=sparse_embeddings,
                                             dense_prompt_embeddings=dense_embeddings)
                
                return pred_masks
        
        self.sam_model = CombinedSAM(self.image_encoder, self.prompt_encoder, self.mask_decoder)
        
    def _freeze_image_encoder(self):
        """Freeze the image encoder parameters"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # Also freeze any batch norm layers in the image encoder
        for module in self.image_encoder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = False
                    
    def get_trainable_parameters(self) -> List[Dict]:
        """Get all trainable parameters for optimization"""
        trainable_params = []
        
        # Add LoRA parameters from mask decoder
        if hasattr(self.mask_decoder, 'peft_config'):
            for name, param in self.mask_decoder.named_parameters():
                if param.requires_grad:
                    trainable_params.append({
                        'name': f'mask_decoder.{name}',
                        'params': param
                    })
        
        # Add LoRA parameters from prompt encoder
        if hasattr(self.prompt_encoder, 'peft_config'):
            for name, param in self.prompt_encoder.named_parameters():
                if param.requires_grad:
                    trainable_params.append({
                        'name': f'prompt_encoder.{name}',
                        'params': param
                    })
        
        return trainable_params
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        return total_params, trainable_params
    
    def save_checkpoint(self, save_path: Union[str, Path], epoch: int, optimizer_state: Dict):
        """Save model checkpoint with LoRA configuration"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'config': self.config.__dict__,
            'lora_config': {
                'rank': self.config.model.lora_rank,
                'alpha': self.config.model.lora_alpha,
                'target_modules': self.config.model.target_modules,
                'dropout': self.config.model.lora_dropout
            }
        }
        
        torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint with LoRA configuration"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint


class SamPredictorLoRA:
    """SAM Predictor adapted for LoRA model"""
    
    def __init__(self, model: SAMLoRAModel, config: Config):
        self.model = model
        self.config = config
        self.original_image_size = None
        
    def set_image(self, image: np.ndarray):
        """Set the image for prediction"""
        import cv2
        
        # Convert BGR to RGB if needed
        if image.ndim == 3 and image.shape[2] == 3:
            if isinstance(image, np.ndarray) and len(image.shape) == 3:
                if image.shape[2] == 3 and image.dtype == np.uint8:
                    # Assume BGR format, convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.original_image_size = image.shape[:2]
        self.image = image
        
        # Preprocess image for SAM
        image_resized = self._preprocess_image(image)
        
        # Get image embeddings
        with torch.no_grad():
            input_image_torch = torch.from_numpy(image_resized).float()
            input_image_torch = input_image_torch.permute(2, 0, 1).unsqueeze(0)
            
            if self.config.system.device == 'cuda':
                input_image_torch = input_image_torch.cuda()
                
            self.features = self.model.image_encoder(input_image_torch)
            
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SAM input"""
        import cv2
        
        # Resize image to SAM input size
        h, w = image.shape[:2]
        new_h, new_w = self.config.data.input_size, self.config.data.input_size
        
        # Calculate scaling factor
        scale = min(new_h / h, new_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        padded_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
        start_h = (new_h - h * scale) // 2
        start_w = (new_w - w * scale) // 2
        padded_image[start_h:start_h + new_h, start_w:start_w + new_w] = image_resized
        
        return padded_image
        
    def predict(self, point_coords: np.ndarray, point_labels: np.ndarray, 
                multimask_output: bool = True):
        """Predict masks given prompts"""
        
        with torch.no_grad():
            # Prepare prompts
            prompts = {
                'points': (point_coords, point_labels)
            }
            
            # Get predictions from LoRA-adapted SAM
            predictions = self.model.sam_model(self.features, prompts)
            
            # Extract masks and scores
            if multimask_output:
                masks = predictions[0][:, :-1, :, :]
                scores = predictions[1]
                masks = torch.sigmoid(masks)
                best_mask_idx = torch.argmax(scores, dim=1)
                masks = masks[range(masks.shape[0]), best_mask_idx]
            else:
                masks = torch.sigmoid(predictions[0][:, 0])
                
            return masks, predictions[1] if multimask_output else None