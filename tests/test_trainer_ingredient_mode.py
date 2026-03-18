from unittest.mock import patch, MagicMock

import torch

from configs.config import Config
from src.training.trainer import Trainer


def _make_mock_model():
    """Return a MagicMock that satisfies Trainer.__init__ requirements."""
    param = torch.nn.Parameter(torch.randn(10))

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.get_trainable_parameters.return_value = [param]
    mock_model.count_parameters.return_value = (1000, 10)
    # Prevent the non-leaf branch from trying to iterate real named_parameters
    mock_model.mask_decoder.named_parameters.return_value = []
    return mock_model


def _make_mock_batch(batch_size: int = 2, img_size: int = 64) -> dict:
    """Create a minimal mock batch for ingredient-mode training/validation."""
    return {
        'image': torch.randn(batch_size, 3, img_size, img_size),
        'instance_mask': torch.zeros(batch_size, img_size, img_size),
        'class_id': torch.zeros(batch_size, dtype=torch.long),
        'prompts': {
            'points': torch.zeros(batch_size, 1, 2),
            'point_labels': torch.ones(batch_size, 1, dtype=torch.long),
        },
    }


def _make_ingredient_trainer(config: Config, mock_model: MagicMock) -> Trainer:
    """Create a Trainer configured for ingredient mode using mocks."""
    with patch("src.training.trainer.SAMLoRAModel", return_value=mock_model), \
         patch("src.training.trainer.DataLoader", return_value=MagicMock()), \
         patch("src.data.foodinsseg_dataset.FoodInsSegDataset"):
        trainer = Trainer(config)
    return trainer


def test_trainer_uses_joint_loss_when_ingredient_head_enabled():
    config = Config()
    config.model.use_ingredient_head = True
    config.data.dataset_name = "FoodInsSeg"

    mock_model = _make_mock_model()

    with patch("src.training.trainer.SAMLoRAModel", return_value=mock_model), \
         patch("src.training.trainer.DataLoader", return_value=MagicMock()), \
         patch("src.data.foodinsseg_dataset.FoodInsSegDataset"):
        trainer = Trainer(config)

    assert trainer.criterion.__class__.__name__ == "JointIngredientLoss"


def test_trainer_uses_combined_loss_when_ingredient_head_disabled():
    config = Config()
    config.model.use_ingredient_head = False

    mock_model = _make_mock_model()

    with patch("src.training.trainer.SAMLoRAModel", return_value=mock_model), \
         patch("src.training.trainer.create_data_loaders",
               return_value=(MagicMock(), MagicMock())):
        trainer = Trainer(config)

    assert trainer.criterion.__class__.__name__ == "CombinedLoss"


def test_train_epoch_ingredient_stats_keys():
    """_train_epoch_ingredient returns all expected stat keys, averaged over the full batch."""
    config = Config()
    config.model.use_ingredient_head = True
    config.data.dataset_name = "FoodInsSeg"
    config.training.gradient_accumulation_steps = 1
    config.training.log_frequency = 1000

    mock_model = _make_mock_model()
    trainer = _make_ingredient_trainer(config, mock_model)

    batch = _make_mock_batch(batch_size=2)
    trainer.train_loader = [batch]

    mock_model.forward_instance.return_value = {
        "mask_logits": torch.randn(2, 1, 64, 64, requires_grad=True),
        "class_logits": torch.randn(2, 103, requires_grad=True),
        "region_features": torch.randn(2, 256),
    }
    mock_model.resize_predictions.side_effect = lambda x, size: x

    stats = trainer._train_epoch_ingredient()

    assert 'train_loss' in stats
    assert 'train_iou' in stats
    assert 'train_dice' in stats
    assert 'train_acc' in stats


def test_validate_epoch_ingredient_stats_keys():
    """_validate_epoch_ingredient returns all expected stat keys."""
    config = Config()
    config.model.use_ingredient_head = True
    config.data.dataset_name = "FoodInsSeg"

    mock_model = _make_mock_model()
    trainer = _make_ingredient_trainer(config, mock_model)

    batch = _make_mock_batch(batch_size=2)
    trainer.val_loader = [batch]

    mock_model.forward_instance.return_value = {
        "mask_logits": torch.randn(2, 1, 64, 64),
        "class_logits": torch.randn(2, 103),
        "region_features": torch.randn(2, 256),
    }
    mock_model.resize_predictions.side_effect = lambda x, size: x

    stats = trainer._validate_epoch_ingredient()

    assert 'val_miou' in stats
    assert 'val_dice' in stats
    assert 'val_acc' in stats
