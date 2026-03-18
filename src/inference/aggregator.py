"""
Semantic map assembly from prompted instance predictions.

Implements greedy NMS-style aggregation: predictions are sorted by score
descending, filtered by score threshold, and added to the semantic map
only when IoU with already-accepted regions is below iou_threshold.
"""

import torch


def compute_mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Compute IoU between two binary masks of shape (H, W)."""
    a = mask_a > 0.5
    b = mask_b > 0.5
    intersection = (a & b).sum().item()
    union = (a | b).sum().item()
    if union == 0:
        return 0.0
    return intersection / union


def merge_instance_predictions(
    masks: torch.Tensor,
    class_ids: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
):
    """
    Merge per-instance predictions into a single semantic map via greedy NMS.

    Args:
        masks: Float tensor of shape (N, H, W) — binary masks with values in {0.0, 1.0}.
               Callers should binarise logits (e.g. sigmoid > 0.5) before passing here.
        class_ids: Long tensor of shape (N,) — class label for each instance.
        scores: Float tensor of shape (N,) — confidence score for each instance.
        score_threshold: Discard predictions with score < this value.
        iou_threshold: Skip a candidate when its IoU with the accepted region
                       exceeds this value.

    Returns:
        semantic_map: Long tensor of shape (H, W) with class labels; background is 0.
        accepted_instances: List of dicts with keys "mask" (tensor H×W), "class_id" (int),
                            and "score" (float) for each accepted instance.
    """
    if masks.numel() == 0 or masks.shape[0] == 0:
        H, W = masks.shape[1], masks.shape[2]
        return torch.zeros((H, W), dtype=torch.long), []

    N, H, W = masks.shape
    semantic_map = torch.zeros((H, W), dtype=torch.long)

    # Step 1: filter by score threshold
    keep = scores >= score_threshold
    if keep.sum() == 0:
        return semantic_map, []

    filtered_masks = masks[keep]
    filtered_class_ids = class_ids[keep]
    filtered_scores = scores[keep]

    # Step 2: sort by score descending
    order = torch.argsort(filtered_scores, descending=True)
    filtered_masks = filtered_masks[order]
    filtered_class_ids = filtered_class_ids[order]
    filtered_scores = filtered_scores[order]

    # Accumulated accepted region (union of all accepted masks)
    accepted_region = torch.zeros((H, W), dtype=torch.bool)
    accepted_instances = []

    # Step 3 & 4: greedy acceptance
    for i in range(filtered_masks.shape[0]):
        candidate_mask = filtered_masks[i]  # (H, W)
        binary_candidate = candidate_mask > 0.5

        if accepted_region.any():
            iou = compute_mask_iou(binary_candidate.float(), accepted_region.float())
            if iou > iou_threshold:
                continue

        # Accept this prediction
        cid = filtered_class_ids[i].item()
        score = filtered_scores[i].item()
        semantic_map[binary_candidate] = cid
        accepted_region = accepted_region | binary_candidate
        accepted_instances.append({"mask": binary_candidate, "class_id": cid, "score": score})

    return semantic_map, accepted_instances
