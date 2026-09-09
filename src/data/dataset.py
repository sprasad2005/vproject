"""Manifest-driven PyTorch Dataset and DataLoader pipelines for RiceGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.annotations import BoundingBox, parse_yolo_bbox_file
from src.data.dataset_registry import validate_dataset_usage
from src.data.grid_assignment import create_grid_target
from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES, class_to_index
from src.utils.paths import get_project_root


def _normalize_dataset_key(dset_str: str) -> str:
    """Map arbitrary dataset strings/names to registered keys ('primary', 'sethy', 'bd5', 'riceseg')."""
    s = str(dset_str).lower().strip()
    if "primary" in s or ("riceleafdiseasebd" in s and "bd5" not in s):
        return "primary"
    elif "sethy" in s or "image samples" in s:
        return "sethy"
    elif "bd5" in s or "five" in s:
        return "bd5"
    elif "riceseg" in s or "seg" in s:
        return "riceseg"
    return s


def rice_leaf_collate_fn(
    batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Top-level picklable collate function for classification-only DataLoader."""
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return images, targets, metadata


def rice_leaf_multitask_collate_fn(
    batch: List[Tuple[torch.Tensor, int, torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Top-level picklable collate function for multi-task DataLoader."""
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    grid_targets = torch.stack([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    return images, targets, grid_targets, metadata


class RiceLeafDataset(Dataset):
    """Manifest-driven PyTorch Dataset for Rice Leaf Disease Classification and Multi-task Localization."""

    def __init__(
        self,
        manifest_path_or_df: Union[str, Path, pd.DataFrame],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        split_name: Optional[str] = None,
        is_training: bool = False,
        is_multitask: bool = False,
        grid_size: int = 7,
    ) -> None:
        self.root = Path(root_dir) if root_dir else get_project_root()
        self.transform = transform
        self.split_name = split_name
        self.is_training = is_training
        self.is_multitask = is_multitask
        self.grid_size = grid_size

        if isinstance(manifest_path_or_df, pd.DataFrame):
            self.df = manifest_path_or_df.copy()
        else:
            p = Path(manifest_path_or_df)
            if not p.is_absolute():
                p = self.root / p
            if not p.exists():
                raise FileNotFoundError(f"Manifest file not found: {p}")
            self.df = pd.read_csv(p)

        # Filter by split if specified
        if self.split_name and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == self.split_name].reset_index(drop=True)

        # Enforce external dataset security policy during training or validation
        if self.is_training:
            for dset in self.df["dataset"].unique():
                dset_key = _normalize_dataset_key(str(dset))
                validate_dataset_usage(dset_key, "train")

        self.samples: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            canonical_cls = str(row["canonical_class"])
            cls_idx = class_to_index(canonical_cls)
            if cls_idx is None:
                continue

            rel_path = str(row["image_path"])
            abs_path = self.root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)

            annot_path = row.get("annotation_path")
            abs_annot_path = None
            if pd.notna(annot_path) and annot_path:
                p_annot = Path(annot_path)
                abs_annot_path = self.root / p_annot if not p_annot.is_absolute() else p_annot

            self.samples.append(
                {
                    "image_id": str(row.get("image_id", abs_path.stem)),
                    "image_path": rel_path,
                    "abs_path": abs_path,
                    "annotation_path": str(annot_path) if pd.notna(annot_path) else None,
                    "abs_annotation_path": abs_annot_path,
                    "canonical_class": canonical_cls,
                    "class_index": cls_idx,
                    "dataset": str(row.get("dataset", "RiceLeafDiseaseBD")),
                    "has_annotation": bool(row.get("has_annotation", False)),
                    "num_boxes": int(row.get("num_boxes", 0)),
                    "split": str(row.get("split", "")),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, int, Dict[str, Any]],
        Tuple[torch.Tensor, int, torch.Tensor, Dict[str, Any]],
    ]:
        sample = self.samples[idx]
        img_path = sample["abs_path"]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            import torchvision.transforms.functional as TF

            image_tensor = TF.to_tensor(image)

        target = sample["class_index"]

        # Parse ground-truth boxes if present
        boxes: List[BoundingBox] = []
        if sample["has_annotation"] and sample["abs_annotation_path"]:
            if sample["abs_annotation_path"].exists():
                boxes, _ = parse_yolo_bbox_file(sample["abs_annotation_path"])

        metadata = {
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "canonical_class": sample["canonical_class"],
            "has_annotation": sample["has_annotation"],
            "num_boxes": len(boxes),
            "split": sample["split"],
            "boxes": boxes,
        }

        if self.is_multitask:
            grid_target, _ = create_grid_target(boxes, grid_size=self.grid_size)
            return image_tensor, target, grid_target, metadata

        return image_tensor, target, metadata

    def get_class_counts(self) -> Dict[str, int]:
        """Return counts per canonical class in this dataset split."""
        counts = {c: 0 for c in CANONICAL_PRIMARY_CLASSES}
        for s in self.samples:
            counts[s["canonical_class"]] = counts.get(s["canonical_class"], 0) + 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights (computed only from this subset)."""
        counts = self.get_class_counts()
        total = len(self.samples)
        num_classes = len(CANONICAL_PRIMARY_CLASSES)
        weights = []
        for c in CANONICAL_PRIMARY_CLASSES:
            cnt = counts.get(c, 0)
            if cnt > 0:
                w = total / (num_classes * cnt)
            else:
                w = 1.0
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    is_multitask: Optional[bool] = None,
) -> DataLoader:
    """Create standard PyTorch DataLoader with appropriate top-level collate function."""
    multitask_mode = (
        is_multitask
        if is_multitask is not None
        else getattr(dataset, "is_multitask", False)
    )
    collate = rice_leaf_multitask_collate_fn if multitask_mode else rice_leaf_collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )
