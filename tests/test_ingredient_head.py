import torch
import torch.nn as nn

from configs.config import Config
from src.models.ingredient_head import IngredientClassificationHead
from src.models.sam_lora import SAMLoRAModel


def test_ingredient_head_outputs_class_logits():
    head = IngredientClassificationHead(
        in_dim=256,
        hidden_dim=128,
        num_classes=103,
    )

    features = torch.randn(2, 256)
    logits = head(features)

    assert logits.shape == (2, 103)


class _DummyImageEncoder(nn.Module):
    def forward(self, image):
        batch_size = image.shape[0]
        return torch.ones((batch_size, 256, 4, 4), dtype=image.dtype)


class _DummySAM(nn.Module):
    def forward(self, image_embeddings, prompts):
        batch_size = image_embeddings.shape[0]
        return torch.ones((batch_size, 1, 4, 4), dtype=image_embeddings.dtype)


class _SpatialDummyImageEncoder(nn.Module):
    def forward(self, image):
        batch_size = image.shape[0]
        left = torch.zeros((batch_size, 256, 4, 2), dtype=image.dtype)
        right = torch.ones((batch_size, 256, 4, 2), dtype=image.dtype)
        return torch.cat([left, right], dim=-1)


class _VariableWidthDummyImageEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.output_dim = channels

    def forward(self, image):
        batch_size = image.shape[0]
        return torch.ones((batch_size, self.channels, 4, 4), dtype=image.dtype)


class _PromptAwareDummySAM(nn.Module):
    def forward(self, image_embeddings, prompts):
        batch_size = image_embeddings.shape[0]
        mask_logits = torch.zeros((batch_size, 1, 4, 4), dtype=image_embeddings.dtype)
        point_coords = prompts["points"]

        for idx in range(batch_size):
            x_coord = point_coords[idx, 0, 0].item()
            if x_coord < 2:
                mask_logits[idx, :, :, :2] = 1.0
            else:
                mask_logits[idx, :, :, 2:] = 1.0

        return mask_logits


def test_sam_lora_forward_instance_returns_mask_and_class_logits(monkeypatch):
    def _fake_load_sam_model(self):
        self.image_encoder = _DummyImageEncoder()
        self.mask_decoder = nn.Identity()
        self.prompt_encoder = nn.Identity()
        self._original_sam_model = None
        self._original_forward = None

    def _fake_setup_lora_adapters(self):
        self.sam_model = _DummySAM()

    monkeypatch.setattr(SAMLoRAModel, "_load_sam_model", _fake_load_sam_model)
    monkeypatch.setattr(SAMLoRAModel, "_setup_lora_adapters", _fake_setup_lora_adapters)
    monkeypatch.setattr(SAMLoRAModel, "_freeze_image_encoder", lambda self: None)

    config = Config()
    config.model.use_ingredient_head = True
    config.model.num_ingredient_classes = 5
    config.model.ingredient_head_hidden_dim = 32

    model = SAMLoRAModel(config)
    outputs = model.forward_instance(
        image=torch.randn(2, 3, 8, 8),
        prompts={
            "points": torch.zeros((2, 1, 2), dtype=torch.float32),
            "point_labels": torch.ones((2, 1), dtype=torch.int32),
        },
    )

    assert outputs["mask_logits"].shape == (2, 1, 8, 8)
    assert outputs["class_logits"].shape == (2, 5)
    assert outputs["region_features"].shape == (2, 256)


def test_sam_lora_trainable_parameters_include_ingredient_head(monkeypatch):
    def _fake_load_sam_model(self):
        self.image_encoder = _DummyImageEncoder()
        self.mask_decoder = nn.Linear(4, 4)
        self.prompt_encoder = nn.Linear(4, 4)
        self._original_sam_model = None
        self._original_forward = None

    def _fake_setup_lora_adapters(self):
        self.sam_model = _DummySAM()

    monkeypatch.setattr(SAMLoRAModel, "_load_sam_model", _fake_load_sam_model)
    monkeypatch.setattr(SAMLoRAModel, "_setup_lora_adapters", _fake_setup_lora_adapters)
    monkeypatch.setattr(SAMLoRAModel, "_freeze_image_encoder", lambda self: None)

    config = Config()
    config.model.use_ingredient_head = True
    config.model.num_ingredient_classes = 5
    config.model.ingredient_head_hidden_dim = 32

    model = SAMLoRAModel(config)
    trainable_params = model.get_trainable_parameters()

    ingredient_param_ids = {id(param) for param in model.ingredient_head.parameters()}
    returned_param_ids = {id(param) for param in trainable_params}

    assert ingredient_param_ids <= returned_param_ids


def test_sam_lora_forward_instance_uses_prompt_aware_region_features(monkeypatch):
    def _fake_load_sam_model(self):
        self.image_encoder = _SpatialDummyImageEncoder()
        self.mask_decoder = nn.Identity()
        self.prompt_encoder = nn.Identity()
        self._original_sam_model = None
        self._original_forward = None

    def _fake_setup_lora_adapters(self):
        self.sam_model = _PromptAwareDummySAM()

    monkeypatch.setattr(SAMLoRAModel, "_load_sam_model", _fake_load_sam_model)
    monkeypatch.setattr(SAMLoRAModel, "_setup_lora_adapters", _fake_setup_lora_adapters)
    monkeypatch.setattr(SAMLoRAModel, "_freeze_image_encoder", lambda self: None)

    torch.manual_seed(0)
    config = Config()
    config.model.use_ingredient_head = True
    config.model.num_ingredient_classes = 5
    config.model.ingredient_head_hidden_dim = 32

    model = SAMLoRAModel(config)
    outputs = model.forward_instance(
        image=torch.randn(2, 3, 8, 8),
        prompts={
            "points": torch.tensor(
                [[[0.0, 0.0]], [[3.0, 0.0]]],
                dtype=torch.float32,
            ),
            "point_labels": torch.ones((2, 1), dtype=torch.int32),
        },
    )

    assert not torch.allclose(outputs["region_features"][0], outputs["region_features"][1])
    assert not torch.allclose(outputs["class_logits"][0], outputs["class_logits"][1])


def test_sam_lora_forward_instance_supports_non_default_feature_width(monkeypatch):
    def _fake_load_sam_model(self):
        self.image_encoder = _VariableWidthDummyImageEncoder(channels=64)
        self.mask_decoder = nn.Identity()
        self.prompt_encoder = nn.Identity()
        self._original_sam_model = None
        self._original_forward = None

    def _fake_setup_lora_adapters(self):
        self.sam_model = _DummySAM()

    monkeypatch.setattr(SAMLoRAModel, "_load_sam_model", _fake_load_sam_model)
    monkeypatch.setattr(SAMLoRAModel, "_setup_lora_adapters", _fake_setup_lora_adapters)
    monkeypatch.setattr(SAMLoRAModel, "_freeze_image_encoder", lambda self: None)

    config = Config()
    config.model.use_ingredient_head = True
    config.model.num_ingredient_classes = 5
    config.model.ingredient_head_hidden_dim = 16

    model = SAMLoRAModel(config)
    assert model.ingredient_head.net[0].in_features == 64

    outputs = model.forward_instance(
        image=torch.randn(2, 3, 8, 8),
        prompts={
            "points": torch.zeros((2, 1, 2), dtype=torch.float32),
            "point_labels": torch.ones((2, 1), dtype=torch.int32),
        },
    )

    assert outputs["region_features"].shape == (2, 64)
    assert outputs["class_logits"].shape == (2, 5)
