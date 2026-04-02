"""
FoodInsSeg dataset implementation for prompt-conditioned instance supervision.

Supports the COCO instance segmentation format:
  - images/train/, images/test/
  - annotations/Train.json, annotations/Test.json
  - annotations contain: id, image_id, category_id, segmentation (polygon), area, bbox [x,y,w,h], iscrowd
  - images: id, width, height, file_name
  - categories: id, name
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.config import Config


class FoodInsSegDataset(Dataset):
    """Instance dataset for FoodInsSeg in COCO format."""

    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        self.split = split
        self.physical_split = self._map_split(split)
        self.input_size = self.config.data.input_size
        self.dataset_path = Path(self.config.data.dataset_path or "data/FoodInsSeg")
        self.prompt_config = {
            "num_point_prompts": self.config.data.num_point_prompts,
            "num_boxes_per_mask": self.config.data.num_boxes_per_mask,
        }
        self.coco_data, self.class_names = self._load_coco_annotations()
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        if sample_info.get("is_debug_sample"):
            return self._build_debug_sample(sample_info)
        return self._build_dataset_sample(sample_info)

    def _map_split(self, split: str) -> str:
        split_map = {
            "train": "train",
            "val": "test",
            "test": "test",
        }
        return split_map.get(split, split)

    def _get_annotation_path(self) -> Path:
        """Train.json / Test.json with capital T per user spec."""
        name = "Train.json" if self.physical_split == "train" else "Test.json"
        return self.dataset_path / "annotations" / name

    def _load_coco_annotations(self) -> Tuple[Dict[str, Any], Dict[int, str]]:
        ann_path = self._get_annotation_path()
        if not ann_path.exists():
            if self.config.system.debug:
                return {}, {1: "debug_food"}
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = coco.get("categories", [])
        class_names = {c["id"]: c["name"] for c in categories}
        return coco, class_names

    def _load_samples(self) -> List[Dict[str, Any]]:
        coco = self.coco_data
        if not coco:
            if self.config.system.debug:
                return self._create_debug_index()
            raise FileNotFoundError(
                f"No FoodInsSeg annotations for split '{self.split}' in {self.dataset_path}"
            )

        images_by_id = {img["id"]: img for img in coco.get("images", [])}
        annotations = coco.get("annotations", [])
        image_dir = self.dataset_path / "images" / self.physical_split

        samples: List[Dict[str, Any]] = []
        for ann in annotations:
            image_id = ann.get("image_id")
            if image_id is None:
                continue
            img_info = images_by_id.get(image_id)
            if not img_info:
                continue

            segmentation = ann.get("segmentation")
            if not segmentation:
                continue
            iscrowd = ann.get("iscrowd", 0)
            if iscrowd != 0:
                continue

            file_name = img_info.get("file_name")
            if not file_name:
                continue
            image_path = image_dir / file_name
            if not image_path.exists():
                continue

            height = int(img_info.get("height", 0))
            width = int(img_info.get("width", 0))
            if height <= 0 or width <= 0:
                continue

            instance_mask = self._segmentation_to_mask(segmentation, height, width)
            if not instance_mask.any():
                continue

            samples.append(
                {
                    "image_id": str(image_id),
                    "image_path": image_path,
                    "file_name": file_name,
                    "height": height,
                    "width": width,
                    "annotation_id": int(ann.get("id", 0)),
                    "instance_id": int(ann.get("id", len(samples))),
                    "category_id": int(ann["category_id"]),
                    "segmentation": segmentation,
                    "bbox": ann.get("bbox", [0, 0, 0, 0]),
                }
            )

        if not samples:
            if self.config.system.debug:
                return self._create_debug_index()
            raise FileNotFoundError(
                f"No FoodInsSeg instances found for split '{self.split}' in {self.dataset_path}"
            )

        return samples

    def _segmentation_to_mask(
        self, segmentation: Any, height: int, width: int
    ) -> torch.Tensor:
        """Convert COCO segmentation (list of polygons) to binary mask."""
        if isinstance(segmentation, dict):
            return torch.zeros((height, width), dtype=torch.float32)

        mask_np = np.zeros((height, width), dtype=np.uint8)
        # COCO: segmentation is list of polygons, each polygon is [x1,y1,x2,y2,...,xn,yn]
        polygons = segmentation if segmentation else []
        if polygons and isinstance(polygons[0], (list, tuple)):
            pass  # Already list of polygons
        elif polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]  # Single polygon
        else:
            return torch.from_numpy(mask_np.astype(np.float32))

        for poly in polygons:
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(mask_np, [pts], 1)

        return torch.from_numpy((mask_np > 0).astype(np.float32))

    def _create_debug_index(self) -> List[Dict[str, Any]]:
        return [
            {
                "image_id": f"debug_{self.split}_0",
                "instance_id": 0,
                "class_id": 1,
                "category_id": 1,
                "is_debug_sample": True,
            }
        ]

    def _build_dataset_sample(self, sample_info: Dict[str, Any]) -> Dict[str, Any]:
        image = self._load_image(sample_info["image_path"], sample_info.get("height"), sample_info.get("width"))
        raw_instance_mask = self._segmentation_to_mask(
            sample_info["segmentation"],
            sample_info.get("height", self.input_size),
            sample_info.get("width", self.input_size),
        )
        instance_mask = self._resize_mask(raw_instance_mask)
        category_id = sample_info["category_id"]
        class_id = torch.tensor(category_id, dtype=torch.int64)
        class_name = self.class_names.get(category_id, str(category_id))

        return {
            "image": image,
            "instance_mask": instance_mask,
            "mask": instance_mask.clone(),
            "class_id": class_id,
            "class_name": class_name,
            "prompts": self._generate_prompts(instance_mask),
            "metadata": {
                "split": self.split,
                "image_id": sample_info["image_id"],
                "instance_id": sample_info["instance_id"],
                "image_path": str(sample_info["image_path"]),
                "annotation_id": sample_info.get("annotation_id"),
                "is_debug_sample": False,
            },
        }

    def _build_debug_sample(
        self, sample_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a deterministic synthetic sample when the dataset is unavailable."""
        sample_info = sample_info or self._create_debug_index()[0]

        image = self._normalize_image(
            torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)
        )
        instance_mask = torch.zeros((self.input_size, self.input_size), dtype=torch.float32)
        inner_start = max(1, self.input_size // 4)
        inner_end = max(inner_start + 1, self.input_size - inner_start)
        instance_mask[inner_start:inner_end, inner_start:inner_end] = 1.0

        class_id = torch.tensor(sample_info["class_id"], dtype=torch.int64)
        class_name = self.class_names.get(sample_info.get("category_id", 1), "debug_food")

        return {
            "image": image,
            "instance_mask": instance_mask,
            "mask": instance_mask.clone(),
            "class_id": class_id,
            "class_name": class_name,
            "prompts": self._generate_prompts(instance_mask),
            "metadata": {
                "split": self.split,
                "image_id": sample_info["image_id"],
                "instance_id": sample_info["instance_id"],
                "dataset_name": "FoodInsSeg",
                "source_dataset": self.config.data.dataset_name,
                "is_debug_sample": True,
            },
        }

    def _load_image(self, image_path: Path, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if float(image_np.max() - image_np.min()) < 0.5:
            image_np = np.round((image_np * 255.0) / 5.0) * 5.0 / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return self._normalize_image(image_tensor)

    def _normalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config.data.mean, dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.config.data.std, dtype=image_tensor.dtype).view(3, 1, 1)
        return (image_tensor - mean) / std

    def _resize_mask(self, instance_mask: torch.Tensor) -> torch.Tensor:
        mask = Image.fromarray((instance_mask.cpu().numpy() > 0).astype(np.uint8) * 255)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(mask_np)

    def _generate_prompts(self, instance_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_np = instance_mask.cpu().numpy()
        positive_coords = np.argwhere(mask_np > 0)
        num_point_prompts = max(int(self.prompt_config["num_point_prompts"]), 0)
        num_boxes_per_mask = max(int(self.prompt_config["num_boxes_per_mask"]), 0)

        if positive_coords.size == 0:
            return {
                "points": torch.empty((0, 2), dtype=torch.float32),
                "point_labels": torch.empty((0,), dtype=torch.int32),
                "bbox": torch.zeros((4,), dtype=torch.float32),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }

        negative_coords = np.argwhere(mask_np == 0)
        prompt_points = []
        prompt_labels = []

        if num_point_prompts > 0:
            prompt_points.append([float(positive_coords[0][1]), float(positive_coords[0][0])])
            prompt_labels.append(1)

        remaining_points = max(num_point_prompts - len(prompt_points), 0)
        if remaining_points > 0 and negative_coords.size > 0:
            for coord in negative_coords[:remaining_points]:
                prompt_points.append([float(coord[1]), float(coord[0])])
                prompt_labels.append(0)

        y_min, x_min = positive_coords.min(axis=0)
        y_max, x_max = positive_coords.max(axis=0)
        bbox = torch.tensor(
            [
                float(x_min),
                float(y_min),
                float(x_max + 1),
                float(y_max + 1),
            ],
            dtype=torch.float32,
        )
        boxes = (
            bbox.unsqueeze(0).repeat(num_boxes_per_mask, 1)
            if num_boxes_per_mask > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        bbox_value = bbox if num_boxes_per_mask > 0 else torch.zeros((4,), dtype=torch.float32)

        return {
            "points": torch.tensor(prompt_points, dtype=torch.float32),
            "point_labels": torch.tensor(prompt_labels, dtype=torch.int32),
            "bbox": bbox_value,
            "boxes": boxes,
        }
