import torch
from src.inference.aggregator import merge_instance_predictions


def test_merge_instance_predictions_returns_semantic_map():
    masks = torch.randint(0, 2, (3, 8, 8)).float()
    class_ids = torch.tensor([1, 2, 3])
    scores = torch.tensor([0.9, 0.8, 0.7])
    semantic_map, _ = merge_instance_predictions(masks, class_ids, scores, score_threshold=0.5)
    assert semantic_map.shape == (8, 8)


def test_score_threshold_filters_low_confidence():
    # Two masks: one above threshold, one below
    masks = torch.zeros(2, 8, 8)
    masks[0, :4, :4] = 1.0   # first mask: high score
    masks[1, 4:, 4:] = 1.0   # second mask: low score
    class_ids = torch.tensor([1, 2])
    scores = torch.tensor([0.9, 0.3])
    sem_map, _ = merge_instance_predictions(masks, class_ids, scores, score_threshold=0.5)
    # class 2 should be absent (filtered by threshold)
    assert (sem_map == 2).sum() == 0
    # class 1 should be present
    assert (sem_map == 1).sum() > 0


def test_overlap_suppression():
    # Two masks that heavily overlap; only higher-scored one should be kept
    masks = torch.ones(2, 8, 8)  # both cover entire image
    class_ids = torch.tensor([1, 2])
    scores = torch.tensor([0.9, 0.8])
    sem_map, _ = merge_instance_predictions(masks, class_ids, scores, score_threshold=0.5, iou_threshold=0.5)
    # class 2 should be suppressed
    assert (sem_map == 2).sum() == 0
    assert (sem_map == 1).sum() > 0


def test_non_overlapping_masks_both_kept():
    # Two non-overlapping masks: both should appear in the semantic map
    masks = torch.zeros(2, 8, 8)
    masks[0, :4, :] = 1.0
    masks[1, 4:, :] = 1.0
    class_ids = torch.tensor([1, 2])
    scores = torch.tensor([0.9, 0.8])
    sem_map, _ = merge_instance_predictions(masks, class_ids, scores, score_threshold=0.5)
    assert (sem_map == 1).sum() > 0
    assert (sem_map == 2).sum() > 0


def test_empty_predictions_returns_zero_map():
    masks = torch.zeros(0, 8, 8)
    class_ids = torch.zeros(0, dtype=torch.long)
    scores = torch.zeros(0)
    sem_map, accepted = merge_instance_predictions(masks, class_ids, scores, score_threshold=0.5)
    assert sem_map.shape == (8, 8)
    assert (sem_map == 0).all()
    assert accepted == []


def test_accepted_instances_populated():
    # Two non-overlapping masks above threshold — both should appear in accepted_instances
    masks = torch.zeros(2, 8, 8)
    masks[0, :4, :] = 1.0
    masks[1, 4:, :] = 1.0
    class_ids = torch.tensor([1, 2])
    scores = torch.tensor([0.9, 0.8])
    _, accepted = merge_instance_predictions(masks, class_ids, scores, score_threshold=0.5)
    assert len(accepted) == 2
    for inst in accepted:
        assert "mask" in inst
        assert "class_id" in inst
        assert "score" in inst
        assert inst["mask"].shape == (8, 8)
        assert inst["score"] >= 0.5
