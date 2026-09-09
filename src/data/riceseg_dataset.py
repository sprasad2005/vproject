"""RiceSeg5932 Dataset Loader for Phase 4 Quantitative XAI Evaluation.

Loads matched image-mask pairs strictly for post-training explainability validation.
Strictly prohibits training, validation, checkpoint selection, and threshold optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset_registry import validate_dataset_usage
from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES, class_to_index
from src.utils.paths import get_project_root

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RiceSegDataset(Dataset):
    """Manifest-driven PyTorch Dataset for RiceSeg5932 Pixel-Level Ground-Truth Masks.

    Returns:
        image_tensor: FloatTensor of shape [3, H, W] normalized with ImageNet statistics.
        segmentation_mask: FloatTensor of shape [1, H, W] with binary values {0.0, 1.0}.
        metadata: Dictionary containing image_id, paths, class names, and sample info.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path, pd.DataFrame] = "data/processed/manifests/riceseg_manifest.csv",
        root_dir: Optional[Union[str, Path]] = None,
        image_size: Union[int, Tuple[int, int]] = 224,
        filter_canonical_only: bool = True,
    ) -> None:
        # Enforce strict dataset governance policy
        validate_dataset_usage("riceseg", "xai_ground_truth_only")

        self.root = Path(root_dir) if root_dir else get_project_root()
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = (image_size[0], image_size[1])

        if isinstance(manifest_path, pd.DataFrame):
            self.df = manifest_path.copy()
        else:
            p = Path(manifest_path)
            if not p.is_absolute():
                p = self.root / p
            if not p.exists():
                raise FileNotFoundError(f"RiceSeg manifest not found at: {p}")
            self.df = pd.read_csv(p)

        self.filter_canonical_only = filter_canonical_only
        self.samples: List[Dict[str, Any]] = []
        self.excluded_samples: List[Dict[str, Any]] = []

        for _, row in self.df.iterrows():
            canonical_cls = str(row.get("canonical_class", ""))
            cls_idx = class_to_index(canonical_cls)

            rel_img_path = str(row["image_path"])
            rel_mask_path = str(row["mask_path"])
            abs_img_path = self.root / rel_img_path if not Path(rel_img_path).is_absolute() else Path(rel_img_path)
            abs_mask_path = self.root / rel_mask_path if not Path(rel_mask_path).is_absolute() else Path(rel_mask_path)

            sample_info = {
                "sample_id": str(row.get("image_id", abs_img_path.stem)),
                "image_path": rel_img_path,
                "abs_image_path": abs_img_path,
                "mask_path": rel_mask_path,
                "abs_mask_path": abs_mask_path,
                "raw_class": str(row.get("raw_class", "")),
                "canonical_class": canonical_cls,
                "class_index": cls_idx,
                "is_matched_pair": bool(row.get("is_matched_pair", True)),
            }

            # Check eligibility
            if filter_canonical_only and (cls_idx is None or canonical_cls not in CANONICAL_PRIMARY_CLASSES):
                sample_info["exclusion_reason"] = f"Class '{canonical_cls}' is not among 6 canonical primary classes."
                self.excluded_samples.append(sample_info)
                continue

            if not abs_img_path.exists():
                sample_info["exclusion_reason"] = f"Image file missing: {abs_img_path}"
                self.excluded_samples.append(sample_info)
                continue

            if not abs_mask_path.exists():
                sample_info["exclusion_reason"] = f"Mask file missing: {abs_mask_path}"
                self.excluded_samples.append(sample_info)
                continue

            self.samples.append(sample_info)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[index]

        # Load RGB image
        try:
            with Image.open(sample["abs_image_path"]) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading RiceSeg image {sample['abs_image_path']}: {e}")

        # Load grayscale segmentation mask
        try:
            with Image.open(sample["abs_mask_path"]) as m:
                mask = m.convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading RiceSeg mask {sample['abs_mask_path']}: {e}")

        orig_w, orig_h = image.size
        mask_w, mask_h = mask.size
        if (orig_w, orig_h) != (mask_w, mask_h):
            raise ValueError(
                f"Image and mask spatial mismatch for sample {sample['sample_id']}: "
                f"Image size=({orig_w}, {orig_h}), Mask size=({mask_w}, {mask_h})"
            )

        # 1. Resize Image (Bilinear)
        img_resized = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        # 2. Resize Mask (Nearest Neighbor to prevent label blur)
        mask_resized = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        # Convert Image to Tensor and Normalize with ImageNet constants
        img_tensor = TF.to_tensor(img_resized)
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Convert Mask to Binary Float Tensor {0.0, 1.0} (NO ImageNet normalization)
        mask_arr = np.array(mask_resized, dtype=np.float32)
        # Threshold at 127.5 to ensure strict binary mask (lesion = 1.0, background = 0.0)
        binary_mask_arr = (mask_arr > 127.5).astype(np.float32)
        mask_tensor = torch.from_numpy(binary_mask_arr).unsqueeze(0)  # [1, H, W]

        metadata = {
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "abs_image_path": str(sample["abs_image_path"]),
            "mask_path": sample["mask_path"],
            "abs_mask_path": str(sample["abs_mask_path"]),
            "canonical_class": sample["canonical_class"],
            "class_index": sample["class_index"],
            "original_size": (orig_w, orig_h),
            "lesion_pixel_count": int(np.sum(binary_mask_arr)),
            "lesion_area_fraction": float(np.mean(binary_mask_arr)),
        }

        return img_tensor, mask_tensor, metadata

    def get_class_counts(self) -> Dict[str, int]:
        """Return sample counts per canonical class in eligible samples."""
        counts = {c: 0 for c in CANONICAL_PRIMARY_CLASSES}
        for s in self.samples:
            counts[s["canonical_class"]] = counts.get(s["canonical_class"], 0) + 1
        return counts
