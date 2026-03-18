import torch
from src.utils.metrics import (
    calculate_classification_accuracy,
    calculate_topk_accuracy,
    calculate_instance_iou,
    calculate_instance_dice,
)


def test_classification_accuracy_runs():
    logits = torch.randn(4, 103)
    target = torch.randint(0, 103, (4,))
    acc = calculate_classification_accuracy(logits, target)
    assert 0.0 <= acc <= 1.0


def test_classification_accuracy_perfect():
    logits = torch.zeros(3, 5)
    logits[0, 2] = 10.0
    logits[1, 0] = 10.0
    logits[2, 4] = 10.0
    target = torch.tensor([2, 0, 4])
    acc = calculate_classification_accuracy(logits, target)
    assert acc == 1.0


def test_classification_accuracy_zero():
    logits = torch.zeros(3, 5)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0
    target = torch.tensor([4, 3, 0])
    acc = calculate_classification_accuracy(logits, target)
    assert acc == 0.0


def test_topk_accuracy_k1_matches_top1():
    logits = torch.randn(8, 103)
    target = torch.randint(0, 103, (8,))
    top1 = calculate_classification_accuracy(logits, target)
    topk1 = calculate_topk_accuracy(logits, target, k=1)
    assert abs(top1 - topk1) < 1e-6


def test_topk_accuracy_k5_ge_k1():
    logits = torch.randn(8, 103)
    target = torch.randint(0, 103, (8,))
    top1 = calculate_topk_accuracy(logits, target, k=1)
    top5 = calculate_topk_accuracy(logits, target, k=5)
    assert top5 >= top1


def test_instance_iou_perfect_overlap():
    pred = torch.ones(1, 16, 16)
    target = torch.ones(16, 16)
    iou = calculate_instance_iou(pred, target)
    assert abs(iou - 1.0) < 1e-5


def test_instance_iou_no_overlap():
    pred = torch.zeros(1, 16, 16)
    pred[0, :8, :8] = 10.0
    target = torch.zeros(16, 16)
    target[8:, 8:] = 1.0
    iou = calculate_instance_iou(pred, target)
    assert iou == 0.0


def test_instance_dice_perfect_overlap():
    pred = torch.ones(1, 16, 16)
    target = torch.ones(16, 16)
    dice = calculate_instance_dice(pred, target)
    assert abs(dice - 1.0) < 1e-5
