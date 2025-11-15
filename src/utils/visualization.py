"""
Visualization and Results Generation Module

This module handles visualization of predictions, training results,
and generating comprehensive analysis reports.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image
import seaborn as sns

from .metrics import calculate_miou, calculate_dice, calculate_precision_recall_f1


class Visualizer:
    """
    Class for creating visualizations and analysis reports
    """
    
    def __init__(self, config, save_dir: str = "visualizations"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def visualize_predictions(self, model, dataloader, device, 
                            num_samples: int = 20, save_prefix: str = "prediction"):
        """
        Visualize model predictions with ground truth comparisons
        
        Args:
            model: Trained model
            dataloader: DataLoader for evaluation
            device: Device to use
            num_samples: Number of samples to visualize
            save_prefix: Prefix for saved images
        """
        model.eval()
        
        # Create subdirectory for this visualization set
        viz_dir = self.save_dir / f"{save_prefix}_{num_samples}_samples"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        visualized_samples = 0
        sample_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if visualized_samples >= num_samples:
                    break
                
                # Get batch data
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                
                # Get predictions
                image_features = model.image_encoder(image)
                prompts = {
                    'points': batch['prompts']['points'].to(device),
                    'point_labels': batch['prompts']['point_labels'].to(device)
                }
                pred_masks = model.sam_model(image_features, prompts)
                
                # Process each sample in the batch
                batch_size = image.size(0)
                for i in range(batch_size):
                    if visualized_samples >= num_samples:
                        break
                    
                    # Extract individual samples
                    sample_image = image[i].cpu()
                    sample_mask = mask[i].cpu()
                    sample_pred = pred_masks[i].cpu()
                    
                    # Calculate metrics
                    metrics = {
                        'miou': calculate_miou(sample_pred.unsqueeze(0), sample_mask.unsqueeze(0)).item(),
                        'dice': calculate_dice(sample_pred.unsqueeze(0), sample_mask.unsqueeze(0)).item(),
                        **calculate_precision_recall_f1(sample_pred.unsqueeze(0), sample_mask.unsqueeze(0))
                    }
                    sample_metrics.append(metrics)
                    
                    # Create visualization
                    fig = self._create_prediction_visualization(
                        sample_image, sample_mask, sample_pred, metrics, visualized_samples
                    )
                    
                    # Save visualization
                    save_path = viz_dir / f"sample_{visualized_samples:03d}.png"
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    visualized_samples += 1
                
                print(f"Visualized {visualized_samples}/{num_samples} samples")
        
        # Save metrics summary
        self._save_metrics_summary(sample_metrics, viz_dir)
        print(f"Visualization complete! Saved to {viz_dir}")
        
        return viz_dir
    
    def _create_prediction_visualization(self, image: torch.Tensor, 
                                       mask: torch.Tensor, pred: torch.Tensor,
                                       metrics: Dict, sample_idx: int) -> plt.Figure:
        """Create a comprehensive visualization of prediction vs ground truth"""
        
        # Prepare images
        image_np = self._denormalize_image(image).transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        mask_np = mask.numpy().astype(np.float32)
        pred_np = torch.sigmoid(pred).numpy().astype(np.float32)
        pred_binary = (pred_np > 0.5).astype(np.float32)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {sample_idx} - Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        im1 = axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Predicted probability map
        im2 = axes[0, 2].imshow(pred_np, cmap='viridis')
        axes[0, 2].set_title('Predicted Probability Map', fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Binary prediction
        axes[1, 0].imshow(pred_binary, cmap='gray')
        axes[1, 0].set_title('Binary Prediction (Threshold=0.5)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Overlay comparison
        overlay = image_np.copy()
        overlay = self._create_overlay(overlay, mask_np, pred_binary)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Prediction Overlay\n(Green: GT, Red: Pred)', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Metrics text
        axes[1, 2].text(0.1, 0.9, 'Metrics:', fontweight='bold', fontsize=14, transform=axes[1, 2].transAxes)
        metrics_text = f"""
mIoU: {metrics['miou']:.3f}
Dice: {metrics['dice']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1 Score: {metrics['f1']:.3f}
Accuracy: {metrics['accuracy']:.3f}

TP: {metrics['tp']}, FP: {metrics['fp']}
FN: {metrics['fn']}, TN: {metrics['tn']}
        """
        axes[1, 2].text(0.1, 0.7, metrics_text, fontsize=12, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _denormalize_image(self, image: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor image back to numpy array"""
        # Assuming ImageNet normalization
        mean = torch.tensor(self.config.data.mean).view(-1, 1, 1)
        std = torch.tensor(self.config.data.std).view(-1, 1, 1)
        
        denormalized = image * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized.numpy()
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       prediction: np.ndarray) -> np.ndarray:
        """Create overlay image showing ground truth and predictions"""
        overlay = image.copy()
        
        # Create colored masks
        gt_color = np.array([0, 255, 0])  # Green for ground truth
        pred_color = np.array([255, 0, 0])  # Red for prediction
        
        # Apply masks with transparency
        alpha = 0.4
        
        # Add ground truth (green)
        mask_3d = np.stack([mask] * 3, axis=-1)
        overlay = overlay * (1 - alpha * mask_3d) + gt_color * (alpha * mask_3d)
        
        # Add prediction (red)
        pred_3d = np.stack([prediction] * 3, axis=-1)
        overlay = overlay * (1 - alpha * pred_3d) + pred_color * (alpha * pred_3d)
        
        return np.clip(overlay, 0, 1)
    
    def _save_metrics_summary(self, metrics_list: List[Dict], save_dir: Path):
        """Save summary of metrics for all visualized samples"""
        
        # Calculate aggregate statistics
        all_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            all_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Create visualization of metrics distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metrics Distribution Across Visualized Samples', fontsize=16, fontweight='bold')
        
        # Key metrics to plot
        metrics_to_plot = ['miou', 'dice', 'precision', 'recall', 'f1', 'accuracy']
        for idx, metric in enumerate(metrics_to_plot):
            if idx < 6:
                row = idx // 3
                col = idx % 3
                
                values = all_metrics[metric]['values']
                
                # Histogram
                axes[row, col].hist(values, bins=10, alpha=0.7, edgecolor='black')
                axes[row, col].axvline(all_metrics[metric]['mean'], color='red', linestyle='--', 
                                     label=f'Mean: {all_metrics[metric]["mean"]:.3f}')
                axes[row, col].set_title(f'{metric.upper()} Distribution')
                axes[row, col].set_xlabel(metric.upper())
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save detailed metrics to JSON
        with open(save_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"Metrics summary saved to {save_dir}")
    
    def create_training_analysis_plots(self, history: Dict):
        """Create comprehensive training analysis plots"""
        
        if 'train_history' not in history:
            print("No training history found")
            return
        
        # Extract data
        train_history = history['train_history']
        val_history = history.get('val_history', train_history)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Comprehensive Training Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(len(train_history))
        
        # 1. Loss curves
        axes[0, 0].plot(epochs, [h['train_loss'] for h in train_history], 
                        label='Training Loss', linewidth=2)
        if val_history:
            val_epochs = range(len(val_history))
            axes[0, 0].plot(val_epochs, [h.get('val_miou', 0) for h in val_history], 
                           label='Validation mIoU', linewidth=2, secondary_y=True)
        
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss', color='blue')
        if val_history:
            axes[0, 0].right_ax.set_ylabel('mIoU', color='red')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Learning rate schedule
        lrs = [h.get('learning_rate', 0) for h in train_history]
        axes[0, 1].plot(epochs, lrs, linewidth=2, color='orange')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Validation metrics
        if val_history:
            val_miou = [h.get('val_miou', 0) for h in val_history]
            val_dice = [h.get('val_dice', 0) for h in val_history]
            
            axes[1, 0].plot(val_epochs, val_miou, label='mIoU', linewidth=2)
            axes[1, 0].plot(val_epochs, val_dice, label='Dice', linewidth=2)
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Target (0.5)')
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Loss components
        if 'train_dice_loss' in train_history[0]:
            dice_losses = [h.get('train_dice_loss', 0) for h in train_history]
            bce_losses = [h.get('train_bce_loss', 0) for h in train_history]
            
            axes[1, 1].plot(epochs, dice_losses, label='Dice Loss', linewidth=2)
            axes[1, 1].plot(epochs, bce_losses, label='BCE Loss', linewidth=2)
            axes[1, 1].set_title('Loss Components')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Best model tracking
        if val_history:
            best_miou = 0
            best_epochs = []
            for i, h in enumerate(val_history):
                if h.get('val_miou', 0) > best_miou:
                    best_miou = h.get('val_miou', 0)
                    best_epochs.append(i)
            
            # Plot validation curve with best epochs marked
            axes[2, 0].plot(val_epochs, val_miou, linewidth=2, label='mIoU')
            if best_epochs:
                best_miou_values = [val_miou[i] for i in best_epochs]
                axes[2, 0].scatter(best_epochs, best_miou_values, 
                                 color='red', s=50, zorder=5, label='Best Models')
            axes[2, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Target (0.5)')
            axes[2, 0].set_title('Best Model Tracking')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('mIoU')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Convergence analysis
        if len(train_history) > 10:
            recent_loss = [h['train_loss'] for h in train_history[-10:]]
            moving_avg = np.convolve(recent_loss, np.ones(5)/5, mode='valid')
            axes[2, 1].plot(range(10), recent_loss[-10:], label='Actual Loss', linewidth=2)
            axes[2, 1].plot(range(5), moving_avg, label='Moving Average', linewidth=2, color='red')
            axes[2, 1].set_title('Convergence Analysis (Last 10 Epochs)')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Loss')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training analysis plots saved to {self.save_dir}")
    
    def generate_evaluation_report(self, results: Dict, comparison: Dict = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report_lines = []
        report_lines.append("# Food Segmentation Model Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Model performance summary
        report_lines.append("## Performance Summary")
        report_lines.append("")
        
        current_metrics = results['current_model']
        for metric, value in current_metrics.items():
            report_lines.append(f"- **{metric.upper()}**: {value:.4f}")
        
        report_lines.append("")
        
        # Target achievement
        target_miou = 0.5
        achieved_miou = current_metrics.get('miou', 0)
        target_achieved = achieved_miou >= target_miou
        
        report_lines.append("## Target Achievement")
        report_lines.append("")
        report_lines.append(f"**Target mIoU**: {target_miou:.3f}")
        report_lines.append(f"**Achieved mIoU**: {achieved_miou:.4f}")
        report_lines.append(f"**Status**: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        if target_achieved:
            improvement = achieved_miou - target_miou
            report_lines.append(f"**Improvement**: +{improvement:.4f} (+{(improvement/target_miou)*100:.2f}%)")
        
        report_lines.append("")
        
        # Comparison with baselines
        if comparison and 'baselines' in comparison:
            report_lines.append("## Baseline Comparison")
            report_lines.append("")
            
            baselines = comparison['baselines']
            improvements = comparison.get('improvements', {})
            
            for baseline_name, baseline_metrics in baselines.items():
                report_lines.append(f"### vs {baseline_name}")
                for metric in ['miou', 'dice', 'f1']:
                    if metric in baseline_metrics and metric in current_metrics:
                        improvement_key = f'{metric}_improvement'
                        if improvement_key in improvements:
                            improvement_val = improvements[improvement_key]
                            pct_key = f'{metric}_improvement_pct'
                            if pct_key in improvements:
                                pct_val = improvements[pct_key]
                                report_lines.append(f"- **{metric.upper()}**: {current_metrics[metric]:.4f} "
                                                  f"(+{improvement_val:.4f}, +{pct_val:.2f}%)")
                        else:
                            report_lines.append(f"- **{metric.upper()}**: {current_metrics[metric]:.4f}")
                report_lines.append("")
        
        # Model efficiency analysis
        report_lines.append("## Model Efficiency")
        report_lines.append("")
        
        if comparison and 'model_info' in comparison:
            model_info = comparison['model_info']
            report_lines.append(f"- **Trainable Parameters**: {model_info.get('trainable_params', 'N/A'):,}")
            report_lines.append(f"- **Total Parameters**: {model_info.get('total_params', 'N/A'):,}")
            report_lines.append(f"- **Trainable Percentage**: {model_info.get('trainable_percentage', 'N/A'):.2f}%")
            report_lines.append("")
        
        # Key findings
        report_lines.append("## Key Findings")
        report_lines.append("")
        
        if target_achieved:
            report_lines.append("✅ **Target Achievement**: The model successfully achieved the target mIoU of 0.50")
        else:
            gap = target_miou - achieved_miou
            report_lines.append(f"⚠️ **Target Gap**: The model fell short of the target by {gap:.4f} mIoU points")
        
        if 'f1' in current_metrics:
            f1_score = current_metrics['f1']
            report_lines.append(f"📊 **F1 Score**: Achieved {f1_score:.3f} F1 score for segmentation quality")
        
        if 'boundary_iou' in current_metrics:
            boundary_iou = current_metrics['boundary_iou']
            report_lines.append(f"🎯 **Boundary Quality**: Achieved {boundary_iou:.3f} boundary IoU")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("Report generated automatically by the SAM Food Segmentation pipeline")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.save_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Evaluation report saved to {report_path}")
        
        return report_text