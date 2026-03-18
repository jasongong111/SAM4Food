import torch
from src.training.losses import JointIngredientLoss


def test_joint_loss_returns_mask_and_class_components():
    criterion = JointIngredientLoss(mask_weight=1.0, class_weight=1.0)
    mask_logits = torch.randn(2, 1, 16, 16)
    mask_target = torch.randint(0, 2, (2, 16, 16)).float()
    class_logits = torch.randn(2, 103)
    class_target = torch.randint(0, 103, (2,))
    losses = criterion(mask_logits, mask_target, class_logits, class_target)
    assert "total_loss" in losses
    assert "mask_loss" in losses
    assert "classification_loss" in losses


def test_joint_loss_total_is_weighted_sum():
    criterion = JointIngredientLoss(mask_weight=0.5, class_weight=2.0)
    mask_logits = torch.randn(2, 1, 16, 16)
    mask_target = torch.randint(0, 2, (2, 16, 16)).float()
    class_logits = torch.randn(2, 103)
    class_target = torch.randint(0, 103, (2,))
    losses = criterion(mask_logits, mask_target, class_logits, class_target)
    expected = 0.5 * losses["mask_loss"] + 2.0 * losses["classification_loss"]
    assert torch.isclose(losses["total_loss"], expected, atol=1e-5)


def test_joint_loss_is_differentiable():
    criterion = JointIngredientLoss(mask_weight=1.0, class_weight=1.0)
    mask_logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    mask_target = torch.randint(0, 2, (2, 16, 16)).float()
    class_logits = torch.randn(2, 103, requires_grad=True)
    class_target = torch.randint(0, 103, (2,))
    losses = criterion(mask_logits, mask_target, class_logits, class_target)
    losses["total_loss"].backward()
    assert mask_logits.grad is not None
    assert class_logits.grad is not None


def test_joint_loss_mask_only_when_no_ingredient_head():
    criterion = JointIngredientLoss(mask_weight=1.0, class_weight=1.0)
    mask_logits = torch.randn(2, 1, 16, 16)
    mask_target = torch.randint(0, 2, (2, 16, 16)).float()
    losses = criterion(mask_logits, mask_target, class_logits=None, class_target=None)
    assert "total_loss" in losses
    assert "mask_loss" in losses
    assert losses["classification_loss"] == 0.0
