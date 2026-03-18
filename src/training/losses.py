"""
Joint loss helpers for prompted instance mask and ingredient classification training.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined binary cross-entropy and Dice loss for a batch of masks.

    Args:
        logits: Raw mask logits with shape (B, 1, H, W) or (B, H, W).
        target: Binary ground-truth masks with shape (B, H, W).

    Returns:
        Scalar loss tensor.
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")

    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(-2, -1))
    cardinality = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    # Smooth to prevent division by zero on empty masks
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)
    dice_loss = dice_loss.mean()

    return bce + dice_loss


class JointIngredientLoss(nn.Module):
    """Joint mask and ingredient classification loss.

    Combines:
    - Weighted binary BCE + Dice for the prompted instance mask.
    - Weighted cross-entropy for the ingredient class label.

    When ``class_logits`` / ``class_target`` are ``None`` (ingredient head
    disabled), only the mask loss is computed and ``classification_loss`` is
    reported as ``0.0``.

    Args:
        mask_weight: Scalar weight applied to the mask loss component.
        class_weight: Scalar weight applied to the classification loss component.
    """

    def __init__(self, mask_weight: float = 1.0, class_weight: float = 1.0) -> None:
        super().__init__()
        self.mask_weight = mask_weight
        self.class_weight = class_weight

    def forward(
        self,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,
        class_logits: Optional[torch.Tensor] = None,
        class_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute joint loss.

        Args:
            mask_logits: Shape (B, 1, H, W) or (B, H, W).
            mask_target: Binary ground truth, shape (B, H, W).
            class_logits: Ingredient class logits, shape (B, num_classes), or
                ``None`` when the ingredient head is disabled.
            class_target: Ingredient class indices, shape (B,), or ``None``.

        Returns:
            Dict with keys ``total_loss``, ``mask_loss``, ``classification_loss``.
        """
        mask_loss = _bce_dice_loss(mask_logits, mask_target)

        if class_logits is not None and class_target is not None:
            cls_loss = F.cross_entropy(class_logits, class_target)
        else:
            cls_loss = torch.tensor(0.0, device=mask_logits.device)

        total = self.mask_weight * mask_loss + self.class_weight * cls_loss

        return {
            "total_loss": total,
            "mask_loss": mask_loss,
            "classification_loss": cls_loss,
        }
