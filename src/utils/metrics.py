"""
Evaluation Metrics for Food Segmentation

This module implements the evaluation metrics specified in the project proposal:
- Mean Intersection over Union (mIoU)
- Dice coefficient
- Additional metrics for comprehensive evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) for binary masks
    
    Args:
        pred: Predicted mask (B, H, W) with logits or probabilities
        target: Ground truth mask (B, H, W) with binary values
        threshold: Threshold for converting predictions to binary
    
    Returns:
        IoU value
    """
    # Convert predictions to binary
    if pred.dim() == 3:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Calculate intersection and union
    intersection = torch.sum(pred_binary * target)
    union = torch.sum(pred_binary) + torch.sum(target) - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if torch.sum(target) == 0 else 0.0
    
    return intersection.float() / union.float()


def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice coefficient for binary masks
    
    Args:
        pred: Predicted mask (B, H, W) with logits or probabilities
        target: Ground truth mask (B, H, W) with binary values
        threshold: Threshold for converting predictions to binary
    
    Returns:
        Dice coefficient
    """
    # Convert predictions to binary
    if pred.dim() == 3:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Calculate Dice coefficient
    intersection = torch.sum(pred_binary * target)
    dice = (2.0 * intersection) / (torch.sum(pred_binary) + torch.sum(target))
    
    # Avoid division by zero
    if (torch.sum(pred_binary) + torch.sum(target)) == 0:
        return 1.0 if torch.sum(target) == 0 else 0.0
    
    return dice.float()


def calculate_miou(pred: torch.Tensor, target: torch.Tensor, 
                   iou_thresholds: Optional[List[float]] = None) -> float:
    """
    Calculate mean Intersection over Union (mIoU)
    
    Args:
        pred: Predicted mask (B, H, W) with logits or probabilities
        target: Ground truth mask (B, H, W) with binary values
        iou_thresholds: List of IoU thresholds for COCO-style evaluation
    
    Returns:
        mIoU value
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)
    
    ious = []
    batch_size = pred.size(0)
    
    for threshold in iou_thresholds:
        batch_ious = []
        for i in range(batch_size):
            if pred.dim() == 2:
                pred_i = pred[i:i+1]
                target_i = target[i:i+1]
            else:
                pred_i = pred[i]
                target_i = target[i]
            
            iou = calculate_iou(pred_i, target_i, threshold)
            batch_ious.append(iou.item())
        
        ious.append(np.mean(batch_ious))
    
    return np.mean(ious)


def calculate_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, 
                                 threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score
    
    Args:
        pred: Predicted mask (B, H, W) with logits or probabilities
        target: Ground truth mask (B, H, W) with binary values
        threshold: Threshold for converting predictions to binary
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Convert predictions to binary
    if pred.dim() == 3:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Calculate TP, FP, FN
    tp = torch.sum((pred_binary == 1) & (target == 1))
    fp = torch.sum((pred_binary == 1) & (target == 0))
    fn = torch.sum((pred_binary == 0) & (target == 1))
    tn = torch.sum((pred_binary == 0) & (target == 0))
    
    # Calculate metrics
    precision = tp.float() / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp.float() / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn).float() / (tp + fp + fn + tn)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item()
    }


def calculate_boundary_metrics(pred: torch.Tensor, target: torch.Tensor, 
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate boundary-related metrics (F-boundary, IoU with dilation)
    
    Args:
        pred: Predicted mask (B, H, W) with logits or probabilities
        target: Ground truth mask (B, H, W) with binary values
        threshold: Threshold for converting predictions to binary
    
    Returns:
        Dictionary with boundary metrics
    """
    import torch.nn.functional as F
    
    # Convert predictions to binary
    if pred.dim() == 3:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Calculate boundaries using dilation
    def get_boundary(mask, dilation_size=2):
        # Dilate the mask
        kernel = torch.ones(1, 1, dilation_size*2+1, dilation_size*2+1, device=mask.device)
        dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation_size)
        
        # Get boundary by subtracting original mask
        boundary = (dilated.squeeze() > 0).float() - mask
        boundary = torch.clamp(boundary, 0, 1)
        return boundary
    
    pred_boundary = get_boundary(pred_binary)
    target_boundary = get_boundary(target)
    
    # Calculate boundary IoU
    boundary_intersection = torch.sum(pred_boundary * target_boundary)
    boundary_union = torch.sum(pred_boundary) + torch.sum(target_boundary) - boundary_intersection
    
    boundary_iou = boundary_intersection.float() / boundary_union.float() if boundary_union > 0 else 0.0
    
    # Calculate boundary F1
    boundary_precision = boundary_intersection / torch.sum(pred_boundary) if torch.sum(pred_boundary) > 0 else 0.0
    boundary_recall = boundary_intersection / torch.sum(target_boundary) if torch.sum(target_boundary) > 0 else 0.0
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
    
    return {
        'boundary_iou': boundary_iou.item(),
        'boundary_f1': boundary_f1.item(),
        'boundary_precision': boundary_precision.item(),
        'boundary_recall': boundary_recall.item()
    }


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, 
                  config) -> Dict[str, float]:
    """
    Comprehensive evaluation of a batch of predictions
    
    Args:
        pred: Predicted masks (B, H, W) with logits
        target: Ground truth masks (B, H, W) with binary values
        config: Configuration object with evaluation settings
    
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {}
    
    # Ensure predictions are in correct format
    if pred.dim() == 4:  # (B, 1, H, W)
        pred = pred.squeeze(1)
    
    # Core metrics
    if config.evaluation.compute_miou:
        metrics['miou'] = calculate_miou(pred, target)
    
    if config.evaluation.compute_dice:
        metrics['dice'] = calculate_dice(pred, target)
    
    if config.evaluation.compute_f1:
        prf_metrics = calculate_precision_recall_f1(pred, target)
        metrics.update({f'prf_{k}': v for k, v in prf_metrics.items()})
    
    # Additional boundary metrics
    boundary_metrics = calculate_boundary_metrics(pred, target)
    metrics.update({f'boundary_{k}': v for k, v in boundary_metrics.items()})
    
    return metrics


def evaluate_model(model, dataloader, config, device) -> Dict[str, float]:
    """
    Evaluate model on entire dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader with evaluation data
        config: Configuration object
        device: Device to use for evaluation
    
    Returns:
        Dictionary with aggregated evaluation metrics
    """
    model.eval()
    
    # Initialize accumulators
    total_metrics = {}
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            
            # Get predictions
            image_features = model.image_encoder(image)
            
            prompts = {
                'points': batch['prompts']['points'].to(device),
                'point_labels': batch['prompts']['point_labels'].to(device)
            }
            
            pred_masks = model.sam_model(image_features, prompts)
            pred_masks = model.resize_predictions(pred_masks, mask.shape[-2:])
            
            # Evaluate batch
            batch_metrics = evaluate_batch(pred_masks, mask, config)
            
            # Accumulate metrics
            batch_size = image.size(0)
            num_samples += batch_size
            
            for metric_name, metric_value in batch_metrics.items():
                if metric_name not in total_metrics:
                    total_metrics[metric_name] = 0.0
                total_metrics[metric_name] += metric_value * batch_size
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Evaluation progress: {batch_idx+1}/{len(dataloader)}")
    
    # Average metrics over all samples
    for metric_name in total_metrics:
        total_metrics[metric_name] /= num_samples
    
    return total_metrics


def compare_with_baselines(model, dataloader, config, device, 
                          baseline_results: Optional[Dict] = None) -> Dict:
    """
    Compare model with baseline SAM and FoodSAM
    
    Args:
        model: Current model to evaluate
        dataloader: DataLoader for evaluation
        config: Configuration object
        device: Device to use
        baseline_results: Results from baseline models
    
    Returns:
        Comparison results
    """
    # Evaluate current model
    current_results = evaluate_model(model, dataloader, config, device)
    
    comparison = {
        'current_model': current_results
    }
    
    # Add baseline comparisons if provided
    if baseline_results:
        comparison['baselines'] = baseline_results
        
        # Calculate improvements
        improvements = {}
        for metric in current_results:
            if metric in baseline_results:
                if metric in ['miou', 'dice', 'f1', 'prf_f1']:
                    improvement = current_results[metric] - baseline_results[metric]
                    improvements[f'{metric}_improvement'] = improvement
                    improvements[f'{metric}_improvement_pct'] = (improvement / baseline_results[metric]) * 100 if baseline_results[metric] > 0 else 0
        
        comparison['improvements'] = improvements
    
    return comparison


# ---------------------------------------------------------------------------
# Instance-level classification and mask metrics (ingredient-aware extension)
# ---------------------------------------------------------------------------

def calculate_classification_accuracy(
    logits: torch.Tensor, target: torch.Tensor
) -> float:
    """Top-1 ingredient classification accuracy.

    Args:
        logits: Shape (B, num_classes).
        target: Ground-truth class indices, shape (B,).

    Returns:
        Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == target).float().mean().item()


def calculate_topk_accuracy(
    logits: torch.Tensor, target: torch.Tensor, k: int = 5
) -> float:
    """Top-k ingredient classification accuracy.

    Args:
        logits: Shape (B, num_classes).
        target: Ground-truth class indices, shape (B,).
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy in [0, 1].
    """
    batch_size = target.size(0)
    _, topk_preds = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(target.unsqueeze(1).expand_as(topk_preds))
    return correct.any(dim=1).float().sum().item() / batch_size


def calculate_instance_iou(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """IoU for a single prompted instance mask.

    Args:
        pred: Logits or probabilities of shape (1, H, W) or (H, W).
        target: Binary ground truth of shape (H, W).
        threshold: Binarisation threshold applied after sigmoid.

    Returns:
        IoU value in [0, 1].
    """
    if pred.dim() == 3:
        pred = torch.sigmoid(pred.squeeze(0))
    pred_binary = (pred > threshold).float()

    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return (intersection / union).item()


def calculate_instance_dice(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """Dice coefficient for a single prompted instance mask.

    Args:
        pred: Logits or probabilities of shape (1, H, W) or (H, W).
        target: Binary ground truth of shape (H, W).
        threshold: Binarisation threshold applied after sigmoid.

    Returns:
        Dice coefficient in [0, 1].
    """
    if pred.dim() == 3:
        pred = torch.sigmoid(pred.squeeze(0))
    pred_binary = (pred > threshold).float()

    denom = pred_binary.sum() + target.sum()
    if denom == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return (2.0 * (pred_binary * target).sum() / denom).item()


def create_metric_plots(history: Dict, save_dir: str):
    """
    Create plots for training and evaluation metrics
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics from history
    if 'train_history' not in history:
        print("No training history found for plotting")
        return
    
    train_history = history['train_history']
    val_history = history.get('val_history', train_history)
    
    # Plot training losses
    plt.figure(figsize=(15, 10))
    
    # Training loss plot
    plt.subplot(2, 3, 1)
    train_losses = [epoch['train_loss'] for epoch in train_history]
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Dice loss plot
    plt.subplot(2, 3, 2)
    train_dice_losses = [epoch.get('train_dice_loss', 0) for epoch in train_history]
    val_dice_losses = [epoch.get('val_dice', 0) for epoch in val_history]
    plt.plot(train_dice_losses, label='Train')
    if val_dice_losses:
        plt.plot(val_dice_losses, label='Validation')
    plt.title('Dice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # mIoU plot
    plt.subplot(2, 3, 3)
    if val_history:
        val_miou = [epoch.get('val_miou', 0) for epoch in val_history]
        plt.plot(val_miou, label='Validation mIoU')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Target (0.5)')
        plt.title('Mean IoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.legend()
        plt.grid(True)
    
    # Learning rate plot
    plt.subplot(2, 3, 4)
    learning_rates = [epoch.get('learning_rate', 0) for epoch in train_history]
    plt.plot(learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    # Accuracy metrics plot
    plt.subplot(2, 3, 5)
    if val_history:
        val_f1 = [epoch.get('prf_f1', 0) for epoch in val_history]
        val_precision = [epoch.get('prf_precision', 0) for epoch in val_history]
        val_recall = [epoch.get('prf_recall', 0) for epoch in val_history]
        
        plt.plot(val_f1, label='F1')
        plt.plot(val_precision, label='Precision')
        plt.plot(val_recall, label='Recall')
        plt.title('Accuracy Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
    
    # Target achievement
    plt.subplot(2, 3, 6)
    if val_history:
        target_achieved = [epoch.get('val_miou', 0) >= 0.5 for epoch in val_history]
        epochs_target_achieved = sum(target_achieved)
        total_epochs = len(target_achieved)
        
        plt.bar(['Target Achieved', 'Not Achieved'], 
                [epochs_target_achieved, total_epochs - epochs_target_achieved])
        plt.title(f'Target Achievement (mIoU > 0.5)')
        plt.ylabel('Number of Epochs')
        
        # Add percentage text
        pct = (epochs_target_achieved / total_epochs) * 100
        plt.text(0.5, max(epochs_target_achieved, total_epochs - epochs_target_achieved) * 0.5,
                f'{pct:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plots saved to {save_dir / 'training_metrics.png'}")