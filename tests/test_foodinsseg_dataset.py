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


def _create_foodinsseg_coco_fixture(
    tmp_path,
    *,
    split: str = "train",
    image_size: int = 6,
    has_valid_instance: bool = True,
):
    """Create FoodInsSeg structure in COCO format."""
    dataset_root = tmp_path / "FoodInsSeg"
    image_dir = dataset_root / "images" / split
    annotation_dir = dataset_root / "annotations"

    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    file_name = "00000001.jpg"
    image_path = image_dir / file_name
    _write_rgb_image(image_path, size=image_size)

    # Polygon for region [1:5, 2:4] in 6x6 (y,x order for polygon: x,y pairs)
    # bbox [x,y,w,h] = [2,1,2,4]
    if has_valid_instance:
        # Polygon: rectangle (x,y) - (2,1), (4,1), (4,5), (2,5)
        polygon = [[2, 1, 4, 1, 4, 5, 2, 5]]
    else:
        polygon = []

    ann_file = "Train.json" if split == "train" else "Test.json"
    coco = {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "width": image_size,
                "height": image_size,
                "file_name": file_name,
            }
        ],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 7,
                "segmentation": polygon,
                "area": 8.0 if has_valid_instance else 0.0,
                "bbox": [2, 1, 2, 4] if has_valid_instance else [0, 0, 0, 0],
                "iscrowd": 0,
            }
        ]
        if has_valid_instance
        else [],
        "categories": [
            {"id": 7, "name": "tomato"},
        ],
    }
    (annotation_dir / ann_file).write_text(json.dumps(coco), encoding="utf-8")

    return dataset_root


def test_foodinsseg_loads_instance_sample_from_dataset(tmp_path):
    dataset_root = _create_foodinsseg_coco_fixture(tmp_path)

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
    assert sample["metadata"]["image_id"] == "1"
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
    dataset_root = _create_foodinsseg_coco_fixture(tmp_path)

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
    dataset_root = _create_foodinsseg_coco_fixture(tmp_path)

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
    dataset_root = _create_foodinsseg_coco_fixture(tmp_path, has_valid_instance=False)

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.system.debug = False

    with pytest.raises(FileNotFoundError, match="No FoodInsSeg instances found"):
        FoodInsSegDataset(config, split="train")


def test_foodinsseg_val_split_reuses_test_files(tmp_path):
    dataset_root = _create_foodinsseg_coco_fixture(tmp_path, split="test")

    config = Config()
    config.data.dataset_path = str(dataset_root)
    config.data.input_size = 8

    dataset = FoodInsSegDataset(config, split="val")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["metadata"]["split"] == "val"
    assert sample["metadata"]["image_id"] == "1"


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
