"""Dataset Manifest generation and metadata indexing engine for RiceGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data.annotations import parse_yolo_bbox_file
from src.data.label_mapping import normalize_class_name
from src.data.quality_checks import validate_image_file
from src.utils.paths import ensure_dir, get_path, get_project_root


class ManifestGenerator:
    """Builds standardized, machine-readable dataset manifests."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root = Path(root_dir) if root_dir else get_project_root()
        self.manifest_dir = ensure_dir(self.root / "data" / "processed" / "manifests")

    def generate_primary_manifest(self, primary_dir: Path) -> pd.DataFrame:
        """Generate metadata manifest for the primary RiceLeafDiseaseBD benchmark."""
        records: List[Dict[str, Any]] = []

        # Look for Original images directory
        orig_dirs = list(primary_dir.rglob("Original images"))
        if not orig_dirs:
            # Fallback to direct search
            image_candidates = [
                f for f in primary_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"] and "visuals" not in str(f)
            ]
        else:
            orig_root = orig_dirs[0]
            image_candidates = [
                f for f in orig_root.rglob("*")
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

        # Locate annotations directory
        annot_dirs = list(primary_dir.rglob("Annotated images*"))
        labels_map: Dict[str, Path] = {}
        if annot_dirs:
            for l_file in annot_dirs[0].rglob("*.txt"):
                labels_map[l_file.stem] = l_file

        for img_path in sorted(image_candidates):
            rel_path = str(img_path.relative_to(self.root)).replace("\\", "/")
            raw_class = img_path.parent.name
            canonical_class, _ = normalize_class_name(raw_class, "primary")
            img_id = img_path.stem

            val_info = validate_image_file(img_path)

            # Check corresponding YOLO label
            has_anno = img_id in labels_map
            anno_path_str = ""
            num_boxes = 0

            if has_anno:
                anno_path = labels_map[img_id]
                anno_path_str = str(anno_path.relative_to(self.root)).replace("\\", "/")
                boxes, _ = parse_yolo_bbox_file(anno_path, val_info["width"], val_info["height"])
                num_boxes = len(boxes)

            records.append(
                {
                    "image_id": img_id,
                    "image_path": rel_path,
                    "raw_class": raw_class,
                    "canonical_class": canonical_class,
                    "dataset": "RiceLeafDiseaseBD",
                    "width": val_info["width"],
                    "height": val_info["height"],
                    "channels": val_info["channels"],
                    "file_size": val_info["file_size"],
                    "file_hash": val_info["sha256"],
                    "dhash": val_info["dhash"],
                    "is_valid_image": val_info["is_valid"],
                    "annotation_path": anno_path_str,
                    "has_annotation": has_anno,
                    "num_boxes": num_boxes,
                    "split": "",  # To be populated by splitting engine
                }
            )

        df = pd.DataFrame(records)
        manifest_path = self.manifest_dir / "primary_manifest.csv"
        df.to_csv(manifest_path, index=False)
        return df

    def generate_sethy_manifest(self, sethy_dir: Path) -> pd.DataFrame:
        """Generate metadata manifest for Sethy5932 external benchmark."""
        records: List[Dict[str, Any]] = []

        image_files = [
            f for f in sethy_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        for img_path in sorted(image_files):
            rel_path = str(img_path.relative_to(self.root)).replace("\\", "/")
            raw_class = img_path.parent.name
            canonical_class, _ = normalize_class_name(raw_class, "sethy")
            img_id = img_path.stem

            val_info = validate_image_file(img_path)

            records.append(
                {
                    "image_id": img_id,
                    "image_path": rel_path,
                    "raw_class": raw_class,
                    "canonical_class": canonical_class,
                    "dataset": "Sethy5932",
                    "width": val_info["width"],
                    "height": val_info["height"],
                    "channels": val_info["channels"],
                    "file_size": val_info["file_size"],
                    "file_hash": val_info["sha256"],
                    "dhash": val_info["dhash"],
                    "is_valid_image": val_info["is_valid"],
                    "usage": "external_test_only",
                    "training_allowed": False,
                    "validation_allowed": False,
                    "calibration_allowed": False,
                    "threshold_optimization_allowed": False,
                }
            )

        df = pd.DataFrame(records)
        manifest_path = self.manifest_dir / "sethy_manifest.csv"
        df.to_csv(manifest_path, index=False)
        return df

    def generate_bd5_manifest(self, bd5_dir: Path) -> pd.DataFrame:
        """Generate metadata manifest for RiceLeafDiseaseBD5 field benchmark."""
        records: List[Dict[str, Any]] = []

        # Target distinct RiceLeafBD subfolder if present to avoid dual-folder recursion
        bd5_roots = list(bd5_dir.rglob("RiceLeafBD"))
        target_dir = bd5_roots[0] if bd5_roots else bd5_dir

        image_files = [
            f for f in target_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        for img_path in sorted(image_files):
            rel_path = str(img_path.relative_to(self.root)).replace("\\", "/")
            raw_class = img_path.parent.name
            canonical_class, _ = normalize_class_name(raw_class, "bd5")
            img_id = img_path.stem

            val_info = validate_image_file(img_path)

            records.append(
                {
                    "image_id": img_id,
                    "image_path": rel_path,
                    "raw_class": raw_class,
                    "canonical_class": canonical_class,
                    "dataset": "RiceLeafDiseaseBD5",
                    "width": val_info["width"],
                    "height": val_info["height"],
                    "channels": val_info["channels"],
                    "file_size": val_info["file_size"],
                    "file_hash": val_info["sha256"],
                    "dhash": val_info["dhash"],
                    "is_valid_image": val_info["is_valid"],
                    "usage": "external_field_test_only",
                    "training_allowed": False,
                    "validation_allowed": False,
                    "calibration_allowed": False,
                    "threshold_optimization_allowed": False,
                }
            )

        df = pd.DataFrame(records)
        manifest_path = self.manifest_dir / "bd5_manifest.csv"
        df.to_csv(manifest_path, index=False)
        return df

    def generate_riceseg_manifest(self, riceseg_dir: Path, sethy_manifest_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate metadata manifest for RiceSeg5932 XAI ground truth masks."""
        records: List[Dict[str, Any]] = []

        mask_files = [
            f for f in riceseg_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        sethy_img_map: Dict[str, str] = {}
        if sethy_manifest_df is not None and not sethy_manifest_df.empty:
            for _, row in sethy_manifest_df.iterrows():
                sethy_img_map[row["image_id"]] = row["image_path"]

        for mask_path in sorted(mask_files):
            rel_mask_path = str(mask_path.relative_to(self.root)).replace("\\", "/")
            raw_class = mask_path.parent.name
            canonical_class, _ = normalize_class_name(raw_class, "riceseg")
            mask_id = mask_path.stem

            matched_img_path = sethy_img_map.get(mask_id, "")
            is_matched = bool(matched_img_path)

            records.append(
                {
                    "image_id": mask_id,
                    "image_path": matched_img_path,
                    "mask_path": rel_mask_path,
                    "raw_class": raw_class,
                    "canonical_class": canonical_class,
                    "dataset": "RiceSeg5932",
                    "is_matched_pair": is_matched,
                    "usage": "xai_ground_truth_only",
                    "training_allowed": False,
                    "validation_allowed": False,
                    "calibration_allowed": False,
                    "threshold_optimization_allowed": False,
                }
            )

        df = pd.DataFrame(records)
        manifest_path = self.manifest_dir / "riceseg_manifest.csv"
        df.to_csv(manifest_path, index=False)
        return df
