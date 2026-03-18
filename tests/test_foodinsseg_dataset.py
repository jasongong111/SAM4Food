import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from configs.config import Config
from src.data.foodinsseg_dataset import FoodInsSegDataset


def _write_rgb_image(path: Path, size: int = 6) -> None:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = 120
    image[..., 1] = 80
    image[..., 2] = 40
    Image.fromarray(image).save(path)


def _write_binary_mask(path: Path, size: int = 6) -> None:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[1:5, 2:4] = 255
    Image.fromarray(mask).save(path)


def _write_empty_mask(path: Path, size: int = 4) -> None:
    mask = np.zeros((size, size), dtype=np.uint8)
    Image.fromarray(mask).save(path)


def _create_foodinsseg_fixture(
    tmp_path,
    *,
    split: str = "train",
    mask_writer=_write_binary_mask,
    image_size: int = 4,
):
    dataset_root = tmp_path / "FoodInsSeg"
    image_sets_dir = dataset_root / "ImageSets"
    image_dir = dataset_root / "Images" / "img_dir" / split
    annotation_dir = dataset_root / "Annotations" / split
    mask_dir = annotation_dir / "masks"

    image_sets_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    (image_sets_dir / f"{split}.txt").write_text("sample_001\n", encoding="utf-8")
    (dataset_root / "class_names.json").write_text(
        json.dumps({"7": "tomato"}), encoding="utf-8"
    )

    _write_rgb_image(image_dir / "sample_001.jpg", size=image_size)
    mask_writer(mask_dir / "sample_001_mask.png", size=image_size)

    annotation_payload = {
        "instances": [
            {
                "instance_id": 11,
                "class_id": 7,
                "mask_path": "masks/sample_001_mask.png",
            }
        ]
    }
    (annotation_dir / "sample_001.json").write_text(
        json.dumps(annotation_payload), encoding="utf-8"
    )

    return dataset_root


def test_foodinsseg_loads_instance_sample_from_dataset(tmp_path):
    dataset_root = _create_foodinsseg_fixture(tmp_path)

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.data.input_size = 8
    config.system.debug = False

    dataset = FoodInsSegDataset(config, split="train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (3, 8, 8)
    assert sample["instance_mask"].shape == (8, 8)
    assert sample["class_id"].item() == 7
    assert sample["class_name"] == "tomato"
    expected_first_channel = ((120 / 255.0) - config.data.mean[0]) / config.data.std[0]
    assert sample["image"][0, 0, 0].item() == pytest.approx(expected_first_channel)
    assert sample["metadata"]["image_id"] == "sample_001"
    assert sample["metadata"]["instance_id"] == 11
    assert sample["prompts"]["points"].shape[0] > 0
    assert "boxes" in sample["prompts"]

    instance_rows, instance_cols = sample["instance_mask"].nonzero(as_tuple=True)
    expected_bbox = [
        float(instance_cols.min().item()),
        float(instance_rows.min().item()),
        float(instance_cols.max().item() + 1),
        float(instance_rows.max().item() + 1),
    ]
    assert sample["prompts"]["bbox"].tolist() == expected_bbox
    assert sample["prompts"]["boxes"].shape == (1, 4)
    assert sample["prompts"]["boxes"][0].tolist() == expected_bbox

    point_coords = sample["prompts"]["points"].to(dtype=torch.int64)
    point_labels = sample["prompts"]["point_labels"].tolist()
    for point, label in zip(point_coords.tolist(), point_labels):
        x_coord, y_coord = point
        assert sample["instance_mask"][y_coord, x_coord].item() == float(label)


def test_foodinsseg_respects_prompt_config_counts(tmp_path):
    dataset_root = _create_foodinsseg_fixture(tmp_path)

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.data.input_size = 8
    config.data.num_point_prompts = 1
    config.data.num_boxes_per_mask = 0

    dataset = FoodInsSegDataset(config, split="train")
    sample = dataset[0]

    assert sample["prompts"]["points"].shape == (1, 2)
    assert sample["prompts"]["point_labels"].tolist() == [1]
    assert sample["prompts"]["boxes"].shape == (0, 4)


def test_foodinsseg_repeats_box_prompt_to_requested_count(tmp_path):
    dataset_root = _create_foodinsseg_fixture(tmp_path)

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.data.input_size = 8
    config.data.num_boxes_per_mask = 2

    dataset = FoodInsSegDataset(config, split="train")
    sample = dataset[0]

    assert sample["prompts"]["boxes"].shape == (2, 4)
    assert sample["prompts"]["boxes"][0].tolist() == sample["prompts"]["bbox"].tolist()
    assert sample["prompts"]["boxes"][1].tolist() == sample["prompts"]["bbox"].tolist()


def test_foodinsseg_skips_empty_instance_masks(tmp_path):
    dataset_root = _create_foodinsseg_fixture(tmp_path, mask_writer=_write_empty_mask)

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.system.debug = False

    with pytest.raises(FileNotFoundError, match="No FoodInsSeg instances found"):
        FoodInsSegDataset(config, split="train")


def test_foodinsseg_val_split_reuses_test_files(tmp_path):
    dataset_root = _create_foodinsseg_fixture(tmp_path, split="test")

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.data.input_size = 8

    dataset = FoodInsSegDataset(config, split="val")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["metadata"]["split"] == "val"
    assert sample["metadata"]["image_id"] == "sample_001"


def test_foodinsseg_debug_mode_returns_synthetic_sample_when_dataset_missing(tmp_path):
    config = Config()
    config.data.dataset_path = str(tmp_path / "missing")
    config.system.debug = True

    dataset = FoodInsSegDataset(config, split="train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["class_name"]
    assert "prompts" in sample
    assert sample["metadata"]["image_id"].startswith("debug_train_")
