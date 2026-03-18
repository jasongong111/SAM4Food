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
    from segment_anything.predictor import SamPredictor
except ImportError:
    print("Installing SAM...")
    import subprocess
    subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"], 
                   check=True)
    from segment_anything import sam_model_registry
    from segment_anything.predictor import SamPredictor

# Import LoRA components
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("Installing PEFT for LoRA...")
    import subprocess
    subprocess.run(["pip", "install", "peft"], check=True)
    from peft import LoraConfig, get_peft_model

from pathlib import Path
from configs.config import Config
from src.models.ingredient_head import IngredientClassificationHead


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

        self.ingredient_head = None
        if self.config.model.use_ingredient_head:
            self.ingredient_head = IngredientClassificationHead(
                in_dim=self._infer_region_feature_dim(),
                hidden_dim=self.config.model.ingredient_head_hidden_dim,
                num_classes=self.config.model.num_ingredient_classes,
            )
        
    def _load_sam_model(self):
        """Load the base SAM model"""
        model_type = self.config.model.sam_model_name
        
        # Load SAM model checkpoint - must be provided manually
        sam_checkpoint = self.config.model.sam_checkpoint_path
        if sam_checkpoint is None:
            raise ValueError(
                f"SAM checkpoint path is required. Please download the SAM {model_type} checkpoint manually:\n"
                f"  - ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                f"  - ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                f"  - ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                f"\nThen specify the path using --model_path argument or set config.model.sam_checkpoint_path"
            )
        
        # Check if checkpoint file exists
        checkpoint_path = Path(sam_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint file not found at: {sam_checkpoint}\n"
                f"Please download the SAM {model_type} checkpoint from:\n"
                f"  https://dl.fbaipublicfiles.com/segment_anything/"
            )
        
        # Register and load the model
        sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        
        # Store components for later use
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        
        # Store original SAM model for accessing helper methods
        self._original_sam_model = sam_model
        
        # Store original forward methods for potential reference
        self._original_forward = sam_model.forward
        
    def _get_target_modules(self, model, model_name="model"):
        """Get target module names for LoRA from the model"""
        target_modules = []
        all_module_names = []
        
        # Find all linear layers in the model - use full module paths
        for name, module in model.named_modules():
            all_module_names.append(name)
            if isinstance(module, nn.Linear):
                # Use full module path for PEFT
                target_modules.append(name)
        
        if target_modules:
            print(f"Found {len(target_modules)} linear layers in {model_name}")
            # Show first few for debugging
            if len(target_modules) > 0:
                print(f"  Example modules: {target_modules[:5]}")
        else:
            # No Linear layers found - print available modules for debugging
            print(f"⚠️  No Linear layers found in {model_name}")
            print(f"  Total modules: {len(all_module_names)}")
            print(f"  Sample module names: {all_module_names[:10]}")
            # Try to find other trainable layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    target_modules.append(name)
                    print(f"  Found Conv layer: {name}")
            # If still nothing, return empty list (will skip LoRA)
            if not target_modules:
                print(f"  ⚠️  No suitable layers found for LoRA in {model_name}")
        
        return target_modules
    
    def _setup_lora_adapters(self):
        """Set up LoRA adapters for the target modules"""
        
        # Helper function to create LoRA config
        def create_lora_config(target_modules):
            return LoraConfig(
                r=self.config.model.lora_rank,
                lora_alpha=self.config.model.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.model.lora_dropout,
                bias="none",  # No bias modifications
            )
        
        # Apply LoRA to mask decoder (required for fine-tuning)
        if self.config.model.target_modules:
            # Use configured target modules
            mask_decoder_modules = self.config.model.target_modules
        else:
            # Auto-detect target modules from mask decoder
            mask_decoder_modules = self._get_target_modules(self.mask_decoder, "mask_decoder")
        
        if not mask_decoder_modules:
            raise ValueError(
                "No suitable layers found in mask_decoder for LoRA. "
                "Mask decoder must have Linear layers for LoRA fine-tuning. "
                "Please check your SAM model installation."
            )
        
        try:
            lora_config_md = create_lora_config(mask_decoder_modules)
            self.mask_decoder = get_peft_model(self.mask_decoder, lora_config_md)
            print("✅ LoRA adapters applied to mask_decoder")
        except ValueError as e:
            print(f"⚠️  Error applying LoRA to mask_decoder: {e}")
            print("  This is a critical error - mask decoder LoRA is required for training.")
            raise
        
        # Apply LoRA to prompt encoder (only if it has suitable layers)
        if self.config.model.target_modules:
            # Use configured target modules
            prompt_encoder_modules = self.config.model.target_modules
        else:
            # Auto-detect target modules from prompt encoder
            prompt_encoder_modules = self._get_target_modules(self.prompt_encoder, "prompt_encoder")
        
        # Only apply LoRA if we found suitable modules
        if prompt_encoder_modules:
            try:
                lora_config_pe = create_lora_config(prompt_encoder_modules)
                self.prompt_encoder = get_peft_model(self.prompt_encoder, lora_config_pe)
                print("✅ LoRA adapters applied to prompt_encoder")
            except ValueError as e:
                print(f"⚠️  Error applying LoRA to prompt_encoder: {e}")
                print("  Skipping LoRA on prompt_encoder (keeping original)")
                # Keep original prompt encoder if LoRA fails
                pass
        else:
            print("ℹ️  Skipping LoRA on prompt_encoder (no suitable layers found)")
            print("  Prompt encoder will remain frozen (original weights)")
        
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
                    image_embeddings: Image features from the frozen encoder [B, C, H, W]
                    prompts: Prompt features (points, boxes, masks)
                
                Returns:
                    Predicted masks [B, 1, H, W]
                """
                batch_size = image_embeddings.shape[0]
                device = image_embeddings.device
                dtype = image_embeddings.dtype

                def _slice_tensor(value, idx):
                    if value is None:
                        return None
                    if isinstance(value, torch.Tensor):
                        if value.shape[0] == 1:
                            return value
                        return value[idx:idx+1]
                    return value

                def _prepare_points(idx):
                    if 'points' not in prompts:
                        return None
                    points_value = prompts['points']
                    labels_value = prompts.get('point_labels')

                    if isinstance(points_value, tuple):
                        # Support prompts where points are already provided as (coords, labels)
                        points_value, tuple_labels = points_value
                        labels_value = tuple_labels

                    if labels_value is None:
                        raise ValueError(
                            "points must include corresponding labels via 'point_labels' or as part of the tuple"
                        )

                    point_coords = _slice_tensor(points_value, idx)
                    point_labels = _slice_tensor(labels_value, idx)

                    if point_coords is None or point_labels is None:
                        return None

                    # Keep only the first point per image to avoid mismatched batch sizes
                    if point_coords.ndim >= 3 and point_coords.shape[1] > 1:
                        point_coords = point_coords[:, 0:1, :]
                        point_labels = point_labels[:, 0:1]

                    return (point_coords, point_labels)

                def _prepare_boxes(idx):
                    boxes = prompts.get('boxes')
                    return _slice_tensor(boxes, idx)

                def _prepare_masks(idx):
                    masks = prompts.get('masks')
                    return _slice_tensor(masks, idx)

                # Generate image positional encoding
                image_pe = None
                try:
                    if hasattr(self.prompt_encoder, 'get_dense_pe'):
                        image_pe = self.prompt_encoder.get_dense_pe()
                        image_pe = image_pe.to(device=device, dtype=dtype)
                    elif hasattr(self.mask_decoder, 'pe_layer'):
                        image_pe = self.mask_decoder.pe_layer(image_embeddings)
                    elif hasattr(self.mask_decoder, 'get_image_pe'):
                        image_pe = self.mask_decoder.get_image_pe(image_embeddings)
                    else:
                        base_model = getattr(self.mask_decoder, 'get_base_model', lambda: self.mask_decoder)()
                        if hasattr(base_model, 'pe_layer'):
                            image_pe = base_model.pe_layer(image_embeddings)
                except Exception:
                    image_pe = None

                if image_pe is None:
                    h, w = image_embeddings.shape[-2:]
                    pe_dim = 256
                    image_pe = torch.zeros((batch_size, pe_dim, h, w), device=device, dtype=dtype)

                masks_list = []

                for idx in range(batch_size):
                    points_tuple = _prepare_points(idx)
                    boxes = _prepare_boxes(idx)
                    masks = _prepare_masks(idx)

                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=points_tuple,
                        boxes=boxes,
                        masks=masks
                    )

                    curr_image_embeddings = image_embeddings[idx:idx+1]
                    curr_image_pe = image_pe if image_pe.shape[0] == 1 else image_pe[idx:idx+1]

                    pred_masks = self.mask_decoder(
                        image_embeddings=curr_image_embeddings,
                        image_pe=curr_image_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False  # single mask output during training
                    )

                    if isinstance(pred_masks, tuple):
                        masks_out = pred_masks[0]
                    else:
                        masks_out = pred_masks

                    masks_list.append(masks_out)

                # Concatenate masks from all samples
                masks = torch.cat(masks_list, dim=0)

                # Ensure we have the right shape: [B, 1, H, W]
                if len(masks.shape) == 4:
                    return masks
                elif len(masks.shape) == 3:
                    return masks.unsqueeze(1)
                else:
                    return masks
        
        self.sam_model = CombinedSAM(self.image_encoder, self.prompt_encoder, self.mask_decoder)
    
    def resize_predictions(
        self,
        pred_masks: torch.Tensor,
        target_size: Optional[Union[Tuple[int, int], torch.Size, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Resize predicted masks to match a desired spatial resolution.
        
        Args:
            pred_masks: Tensor of shape (B, C, H, W) or (B, H, W).
            target_size: Desired (H, W). Defaults to configured input_size if None.
        
        Returns:
            Tensor with the same batch/channel dims as input but spatially resized.
        """
        if pred_masks is None:
            return pred_masks
        
        if target_size is None:
            size = getattr(self.config.data, 'input_size', None)
            if size is None:
                return pred_masks
            target_size = (size, size)
        elif isinstance(target_size, torch.Size):
            target_size = tuple(int(dim) for dim in target_size[-2:])
        elif isinstance(target_size, torch.Tensor):
            if target_size.numel() < 2:
                raise ValueError("target_size tensor must contain at least two elements for (H, W)")
            target_size = tuple(int(dim) for dim in target_size[-2:].tolist())
        else:
            target_size = tuple(int(dim) for dim in target_size)
        
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
            squeeze_channel = True
        else:
            squeeze_channel = False
        
        if tuple(pred_masks.shape[-2:]) != tuple(target_size):
            pred_masks = F.interpolate(
                pred_masks,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        if squeeze_channel:
            pred_masks = pred_masks.squeeze(1)
        
        return pred_masks

    def _infer_region_feature_dim(self) -> int:
        for attr_name in ("output_dim", "out_channels", "embed_dim", "hidden_dim"):
            attr_value = getattr(self.image_encoder, attr_name, None)
            if isinstance(attr_value, int) and attr_value > 0:
                return attr_value

        for module in reversed(list(self.image_encoder.modules())):
            if isinstance(module, nn.Conv2d):
                return module.out_channels
            if isinstance(module, nn.Linear):
                return module.out_features

        return 256

    def _ensure_ingredient_head(self, feature_dim: int, device: torch.device) -> None:
        if not self.config.model.use_ingredient_head:
            return

        needs_init = self.ingredient_head is None
        if not needs_init:
            needs_init = self.ingredient_head.net[0].in_features != feature_dim

        if needs_init:
            self.ingredient_head = IngredientClassificationHead(
                in_dim=feature_dim,
                hidden_dim=self.config.model.ingredient_head_hidden_dim,
                num_classes=self.config.model.num_ingredient_classes,
            ).to(device)

    def forward_instance(self, image: torch.Tensor, prompts: Dict) -> Dict[str, Optional[torch.Tensor]]:
        image_embeddings = self.image_encoder(image)
        mask_logits = self.sam_model(image_embeddings, prompts)

        if isinstance(mask_logits, tuple):
            mask_logits = mask_logits[0]

        embedding_mask = self.resize_predictions(mask_logits, target_size=image_embeddings.shape[-2:])
        embedding_mask = torch.sigmoid(embedding_mask)
        weighted_embeddings = image_embeddings * embedding_mask
        mask_area = embedding_mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        region_features = weighted_embeddings.sum(dim=(-2, -1)) / mask_area

        mask_logits = self.resize_predictions(mask_logits, target_size=image.shape[-2:])
        class_logits = None

        if self.config.model.use_ingredient_head:
            self._ensure_ingredient_head(region_features.shape[1], region_features.device)
            class_logits = self.ingredient_head(region_features)

        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "region_features": region_features,
        }
        
    def _freeze_image_encoder(self):
        """Freeze the image encoder parameters"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # Also freeze any batch norm layers in the image encoder
        for module in self.image_encoder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = False
                    
    def get_trainable_parameters(self):
        """Get all trainable parameters for optimization
        
        Returns:
            List of parameters (tensors) that require gradients
        """
        # For PEFT models, get parameters directly
        # PEFT ensures LoRA adapter parameters are properly set up
        trainable_params = []
        
        # Get parameters from mask decoder (should be LoRA adapters)
        for name, param in self.mask_decoder.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Get parameters from prompt encoder (if LoRA was applied)
        for name, param in self.prompt_encoder.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        if self.ingredient_head is not None:
            for param in self.ingredient_head.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        
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