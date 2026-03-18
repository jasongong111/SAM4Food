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
from src.inference.aggregator import merge_instance_predictions


def load_model(config, lora_checkpoint_path=None):
    device = config.system.device
    model = SAMLoRAModel(config)
    model.to(device)

    if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
        print(f"Loading LoRA checkpoint from {lora_checkpoint_path}...")
        checkpoint = torch.load(lora_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("LoRA checkpoint loaded successfully.")
    else:
        print("No LoRA checkpoint found or provided. Using base SAM model.")

    # Patch: replace original SAM components with LoRA-adapted ones so
    # SamPredictor can be used without modification.
    model._original_sam_model.mask_decoder = model.mask_decoder
    model._original_sam_model.prompt_encoder = model.prompt_encoder

    return model


def run_ingredient_inference(model, image_tensor, prompts_list, config, class_names=None):
    """
    Run prompted ingredient inference and assemble a semantic map.

    Args:
        model: SAMLoRAModel instance (must expose forward_instance).
        image_tensor: Preprocessed image tensor (1, C, H, W) or (C, H, W).
        prompts_list: List of prompt dicts, each passed individually to
                      model.forward_instance as the `prompts` argument.
        config: Config object; uses config.inference.aggregation_score_threshold
                and config.inference.aggregation_iou_threshold.
        class_names: Optional dict mapping class_id (int) -> name (str).

    Returns:
        dict with keys:
            "semantic_map": torch.LongTensor of shape (H, W)
            "class_names":  list of accepted class name strings (if class_names
                            dict was provided, otherwise list of class_id ints)
    """
    score_threshold = config.inference.aggregation_score_threshold
    iou_threshold = config.inference.aggregation_iou_threshold

    all_masks = []
    all_class_ids = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for prompts in prompts_list:
            outputs = model.forward_instance(image_tensor, prompts)
            # mask_logits: (B, 1, H, W)
            mask_logits = outputs["mask_logits"]
            # class_logits: (B, num_classes)
            class_logits = outputs["class_logits"]

            binary_masks = (torch.sigmoid(mask_logits) > 0.5).float().squeeze(1)  # (B, H, W)
            class_probs = torch.softmax(class_logits, dim=-1)
            pred_class_ids = class_probs.argmax(dim=-1)           # (B,)
            confidence_scores = class_probs.max(dim=-1).values    # (B,)

            all_masks.append(binary_masks)
            all_class_ids.append(pred_class_ids)
            all_scores.append(confidence_scores)

    if all_masks:
        masks = torch.cat(all_masks, dim=0)
        class_ids = torch.cat(all_class_ids, dim=0)
        scores = torch.cat(all_scores, dim=0)
    else:
        # No prompts provided — return empty map inferred from image spatial size
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        return {
            "semantic_map": torch.zeros((H, W), dtype=torch.long),
            "class_names": [],
        }

    semantic_map, _ = merge_instance_predictions(
        masks, class_ids, scores,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    accepted_ids = semantic_map.unique().tolist()
    accepted_ids = [cid for cid in accepted_ids if cid != 0]

    if class_names is not None:
        accepted_names = [class_names.get(cid, str(cid)) for cid in accepted_ids]
    else:
        accepted_names = accepted_ids

    return {
        "semantic_map": semantic_map,
        "class_names": accepted_names,
    }


def show_image(image, points=None, labels=None, box=None, mask=None, ax=None, title=None):
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

    ax.imshow(image)

    if mask is not None:
        show_mask(mask, ax)

    if points is not None:
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=200, edgecolor='white', linewidth=1.25, label='Positive')
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=200, edgecolor='white', linewidth=1.25, label='Negative')

    if box is not None:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2, label='Box'))

    if title:
        ax.set_title(title)
    ax.axis('on')


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    print("inference.py — UECFOODPIX single-image inference script")
    print()
    print("Usage:")
    print("  python inference.py")
    print()
    print("Prerequisites:")
    print("  1. Download the base SAM checkpoint (ViT-B):")
    print("     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    print("     Place it at: sam_vit_b_01ec64.pth")
    print()
    print("  2. (Optional) Place a trained LoRA checkpoint at:")
    print("     checkpoints/best_model.pth")
    print()
    print("  3. Ensure the UECFOODPIX dataset is available at:")
    print("     UECFOODPIX/data/UECFoodPIX/{train,test}/img/")
    print()
    print("Ingredient-mode (programmatic API):")
    print("  Use run_ingredient_inference(model, image_tensor, prompts_list, config,")
    print("  class_names=None) to obtain a semantic map and accepted ingredient names.")
    print("  See README.md § 'SAM4Food-Sem: Ingredient-Aware Segmentation' for details.")
    print()
    print("Running UECFOODPIX inference ...")
    print()

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize config
    config = Config()

    config.model.sam_checkpoint_path = "sam_vit_b_01ec64.pth"
    lora_checkpoint_path = "checkpoints/best_model.pth"
    config.system.device = device

    # Import SamPredictor here so the file is importable without segment_anything
    from segment_anything import SamPredictor

    # Create model instance
    try:
        model_wrapper = load_model(config, lora_checkpoint_path)
        predictor = SamPredictor(model_wrapper._original_sam_model)
        print("Model and Predictor initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Please ensure you have downloaded the base SAM checkpoint.")
        predictor = None

    # ## Dataset configuration
    dataset_root = "UECFOODPIX/data/UECFoodPIX"
    train_img_dir = os.path.join(dataset_root, "train", "img")
    test_img_dir = os.path.join(dataset_root, "test", "img")

    output_dir = "inference_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    num_random_images = 5
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_image_paths = []

    print("Collecting images from UECFOODPIX dataset...")

    if os.path.exists(train_img_dir):
        train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir)
                        if os.path.splitext(f)[1].lower() in valid_extensions]
        all_image_paths.extend(train_images)
        print(f"Found {len(train_images)} images in train set")

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
        num_to_select = min(num_random_images, len(all_image_paths))
        selected_images = random.sample(all_image_paths, num_to_select)
        print(f"Randomly selected {num_to_select} images for inference")

    if predictor is not None and selected_images:
        for image_path in selected_images:
            print(f"\nProcessing: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_filename = os.path.basename(image_path)
            file_stem = os.path.splitext(image_filename)[0]

            original_output_path = os.path.join(output_dir, f"original_{image_filename}")
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(original_output_path, image_bgr)
            print(f"Saved original image to {original_output_path}")

            predictor.set_image(image)

            h, w = image.shape[:2]
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1])

            print(f"Prompts: Point {input_point}, Label {input_label}")

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            print(f"Generated {len(masks)} masks for {image_filename}")

            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                show_image(image, input_point, input_label, title=f"{image_filename} - Mask {i + 1}, Score: {score:.3f}")
                show_mask(mask, plt.gca())

                output_filename = f"prediction_{file_stem}_{i + 1}.png"
                output_path = os.path.join(output_dir, output_filename)

                plt.savefig(output_path)
                print(f"Saved prediction result to {output_path}")
                plt.close()
    elif predictor is None:
        print("Predictor not initialized, skipping inference.")
    else:
        print("No images selected for inference.")
