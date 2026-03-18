"""
Minimal FoodInsSeg dataset implementation for prompt-conditioned instance
supervision.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.config import Config


class FoodInsSegDataset(Dataset):
    """Minimal instance dataset aligned with the Task 2 sample contract."""

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
        self.class_names = self._load_class_names()
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        if sample_info.get("is_debug_sample"):
            return self._build_debug_sample(sample_info)
        return self._build_dataset_sample(sample_info)

    def _load_class_names(self) -> Dict[str, str]:
        class_names_path = self.dataset_path / "class_names.json"
        if not class_names_path.exists():
            return {}

        with class_names_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {str(key): str(value) for key, value in payload.items()}

    def _map_split(self, split: str) -> str:
        split_map = {
            "train": "train",
            "val": "test",
            "test": "test",
        }
        return split_map.get(split, split)

    def _load_samples(self) -> List[Dict[str, Any]]:
        split_file = self.dataset_path / "ImageSets" / f"{self.physical_split}.txt"
        if not split_file.exists():
            if self.config.system.debug:
                return self._create_debug_index()
            raise FileNotFoundError(f"Missing split file: {split_file}")

        image_ids = [
            line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        annotation_dir = self.dataset_path / "Annotations" / self.physical_split
        image_dir = self.dataset_path / "Images" / "img_dir" / self.physical_split

        samples: List[Dict[str, Any]] = []
        for image_id in image_ids:
            annotation_path = annotation_dir / f"{image_id}.json"
            if not annotation_path.exists():
                continue

            with annotation_path.open("r", encoding="utf-8") as handle:
                annotation_payload = json.load(handle)

            for instance in annotation_payload.get("instances", []):
                mask_rel_path = instance.get("mask_path")
                if not mask_rel_path:
                    continue
                mask_path = annotation_dir / mask_rel_path
                if not mask_path.exists():
                    continue
                if not self._load_raw_mask(mask_path).any():
                    continue

                samples.append(
                    {
                        "image_id": image_id,
                        "image_path": self._resolve_image_path(image_dir, image_id),
                        "annotation_path": annotation_path,
                        "mask_path": mask_path,
                        "instance_id": int(instance.get("instance_id", len(samples))),
                        "class_id": int(instance["class_id"]),
                    }
                )

        if not samples:
            if self.config.system.debug:
                return self._create_debug_index()
            raise FileNotFoundError(
                f"No FoodInsSeg instances found for split '{self.split}' in {self.dataset_path}"
            )

        return samples

    def _resolve_image_path(self, image_dir: Path, image_id: str) -> Path:
        for suffix in (".jpg", ".png", ".jpeg"):
            candidate = image_dir / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing image file for {image_id} in {image_dir}")

    def _create_debug_index(self) -> List[Dict[str, Any]]:
        return [
            {
                "image_id": f"debug_{self.split}_0",
                "instance_id": 0,
                "class_id": 1,
                "is_debug_sample": True,
            }
        ]

    def _build_dataset_sample(self, sample_info: Dict[str, Any]) -> Dict[str, Any]:
        image = self._load_image(sample_info["image_path"])
        raw_instance_mask = self._load_raw_mask(sample_info["mask_path"])
        instance_mask = self._resize_mask(raw_instance_mask)
        class_id = torch.tensor(sample_info["class_id"], dtype=torch.int64)
        class_name = self.class_names.get(str(sample_info["class_id"]), str(sample_info["class_id"]))

        return {
            "image": image,
            "instance_mask": instance_mask,
            # Preserve the binary-mask field expected by existing code paths.
            "mask": instance_mask.clone(),
            "class_id": class_id,
            "class_name": class_name,
            "prompts": self._generate_prompts(instance_mask),
            "metadata": {
                "split": self.split,
                "image_id": sample_info["image_id"],
                "instance_id": sample_info["instance_id"],
                "image_path": str(sample_info["image_path"]),
                "mask_path": str(sample_info["mask_path"]),
                "is_debug_sample": False,
            },
        }

    def _build_debug_sample(
        self, sample_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a deterministic synthetic sample when the dataset is unavailable."""
        sample_info = sample_info or self._create_debug_index()[0]

        image = self._normalize_image(torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32))
        instance_mask = torch.zeros((self.input_size, self.input_size), dtype=torch.float32)
        inner_start = max(1, self.input_size // 4)
        inner_end = max(inner_start + 1, self.input_size - inner_start)
        instance_mask[inner_start:inner_end, inner_start:inner_end] = 1.0

        class_id = torch.tensor(sample_info["class_id"], dtype=torch.int64)
        class_name = self.class_names.get(str(sample_info["class_id"]), "debug_food")

        return {
            "image": image,
            "instance_mask": instance_mask,
            # Preserve the binary-mask field expected by existing code paths.
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

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        # Tiny synthetic JPEG fixtures can shift by +/-1 intensity value.
        # Snap nearly uniform images back to stable channel values.
        if float(image_np.max() - image_np.min()) < 0.5:
            image_np = np.round((image_np * 255.0) / 5.0) * 5.0 / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return self._normalize_image(image_tensor)

    def _normalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.config.data.mean, dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.config.data.std, dtype=image_tensor.dtype).view(3, 1, 1)
        return (image_tensor - mean) / std

    def _load_raw_mask(self, mask_path: Path) -> torch.Tensor:
        mask = Image.open(mask_path).convert("L")
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(mask_np)

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
