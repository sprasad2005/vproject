"""Read-only dataset discovery, inspection, and integrity auditing engine for RiceGuard."""

from __future__ import annotations

import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.dataset_registry import DATASETS, get_dataset_info
from src.utils.paths import get_project_root


class DatasetInspector:
    """Discovers and inspects dataset repositories and archives in strict read-only mode."""

    SUPPORTED_IMAGE_EXTS: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )
    SUPPORTED_ANNOTATION_EXTS: Tuple[str, ...] = (
        ".xml",
        ".json",
        ".csv",
        ".txt",
        ".xlsx",
    )

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root = Path(root_dir) if root_dir else get_project_root()

    def find_dataset_folders(self) -> Dict[str, Dict[str, Any]]:
        """Search project directory for dataset roots, nested directories, and archive files."""
        discovered: Dict[str, Dict[str, Any]] = {}

        # Prioritize actual directories over loose archive files
        items = sorted(self.root.iterdir(), key=lambda p: (not p.is_dir(), p.name))

        for item in items:
            name_lower = item.name.lower()

            if "riceleafdiseasebd" in name_lower and "bd5" not in name_lower:
                if "primary" not in discovered or item.is_dir():
                    discovered["primary"] = {
                        "matched_path": item,
                        "relative_path": str(item.relative_to(self.root)).replace("\\", "/"),
                        "type": "directory" if item.is_dir() else "archive",
                    }
            elif "image samples" in name_lower or ("sethy" in name_lower):
                if "sethy" not in discovered or item.is_dir():
                    discovered["sethy"] = {
                        "matched_path": item,
                        "relative_path": str(item.relative_to(self.root)).replace("\\", "/"),
                        "type": "directory" if item.is_dir() else "archive",
                    }
            elif "bd5" in name_lower or "five-class" in name_lower:
                if "bd5" not in discovered or item.is_dir():
                    discovered["bd5"] = {
                        "matched_path": item,
                        "relative_path": str(item.relative_to(self.root)).replace("\\", "/"),
                        "type": "directory" if item.is_dir() else "archive",
                    }
            elif "riceseg" in name_lower or "segmentation" in name_lower:
                if "riceseg" not in discovered or item.is_dir():
                    discovered["riceseg"] = {
                        "matched_path": item,
                        "relative_path": str(item.relative_to(self.root)).replace("\\", "/"),
                        "type": "directory" if item.is_dir() else "archive",
                    }

        return discovered

    def inspect_primary_dataset(self, base_path: Path) -> Dict[str, Any]:
        """Inspect the primary RiceLeafDiseaseBD dataset (files, classes, annotations)."""
        reg_info = get_dataset_info("primary")
        result: Dict[str, Any] = {
            "name": reg_info["name"],
            "role": reg_info["role"],
            "training_allowed": reg_info["training_allowed"],
            "detected_root": str(base_path.relative_to(self.root)).replace("\\", "/"),
            "status": "DISCOVERED",
            "total_files": 0,
            "total_images": 0,
            "image_formats": [],
            "classes": {},
            "bounding_boxes": {
                "available": True,
                "format": "YOLO (.txt)",
                "total_annotation_files": 0,
                "classes_with_annotations": {},
            },
            "metadata_files": [],
            "issues": [],
        }

        # Check for extracted contents or nested zip
        zip_files = list(base_path.rglob("*.zip"))
        excel_files = list(base_path.rglob("*.xlsx"))
        pdf_files = list(base_path.rglob("*.pdf"))

        for m in excel_files + pdf_files:
            result["metadata_files"].append(str(m.relative_to(self.root)).replace("\\", "/"))

        if zip_files:
            primary_zip = zip_files[0]
            with zipfile.ZipFile(primary_zip, "r") as z:
                names = z.namelist()
                result["total_files"] = len(names)

                orig_images = [
                    n for n in names
                    if "Original images/" in n and n.lower().endswith(self.SUPPORTED_IMAGE_EXTS)
                ]
                result["total_images"] = len(orig_images)
                result["image_formats"] = sorted(list({Path(n).suffix.lower() for n in orig_images}))

                # Class breakdown of original images
                class_counts: Counter[str] = Counter()
                for n in orig_images:
                    parts = n.split("Original images/")
                    if len(parts) > 1 and "/" in parts[1]:
                        cls_name = parts[1].split("/")[0]
                        class_counts[cls_name] += 1
                result["classes"] = dict(sorted(class_counts.items()))

                # Bounding box labels breakdown
                label_files = [
                    n for n in names
                    if "Annotated images" in n and "labels/" in n and n.lower().endswith(".txt")
                ]
                result["bounding_boxes"]["total_annotation_files"] = len(label_files)
                lbl_counts: Counter[str] = Counter()
                for n in label_files:
                    parts = n.split("Annotated images ( visual with labels)/")
                    if len(parts) > 1 and "/" in parts[1]:
                        cls_name = parts[1].split("/")[0]
                        lbl_counts[cls_name] += 1
                result["bounding_boxes"]["classes_with_annotations"] = dict(sorted(lbl_counts.items()))

        return result

    def inspect_sethy_dataset(self, base_path: Path) -> Tuple[Dict[str, Any], Set[str]]:
        """Inspect the Sethy5932 external benchmark dataset."""
        reg_info = get_dataset_info("sethy")
        result: Dict[str, Any] = {
            "name": reg_info["name"],
            "role": reg_info["role"],
            "training_allowed": reg_info["training_allowed"],
            "detected_root": str(base_path.relative_to(self.root)).replace("\\", "/"),
            "status": "DISCOVERED",
            "total_images": 0,
            "image_formats": [],
            "classes": {},
            "archive_path": "",
            "issues": [],
        }
        image_stems: Set[str] = set()

        archives_7z = list(base_path.rglob("*.7z"))
        if archives_7z:
            arch = archives_7z[0]
            result["archive_path"] = str(arch.relative_to(self.root)).replace("\\", "/")
            try:
                out = subprocess.check_output(["tar", "-tf", str(arch)], encoding="utf-8", errors="ignore")
                lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
                img_lines = [line for line in lines if line.lower().endswith(self.SUPPORTED_IMAGE_EXTS)]
                result["total_images"] = len(img_lines)
                result["image_formats"] = sorted(list({Path(line).suffix.lower() for line in img_lines}))

                class_counts: Counter[str] = Counter()
                for line in img_lines:
                    parts = line.split("/")
                    if len(parts) >= 2:
                        cls_name = parts[1]
                        class_counts[cls_name] += 1
                    image_stems.add(Path(line).stem)
                result["classes"] = dict(sorted(class_counts.items()))
            except Exception as e:
                result["issues"].append(f"Archive listing error: {e}")

        return result, image_stems

    def inspect_bd5_dataset(self, base_path: Path) -> Dict[str, Any]:
        """Inspect the RiceLeafDiseaseBD5 external field dataset."""
        reg_info = get_dataset_info("bd5")
        result: Dict[str, Any] = {
            "name": reg_info["name"],
            "role": reg_info["role"],
            "training_allowed": reg_info["training_allowed"],
            "detected_root": str(base_path.relative_to(self.root)).replace("\\", "/"),
            "status": "DISCOVERED",
            "total_images": 0,
            "image_formats": [],
            "classes": {},
            "archive_path": "",
            "issues": [],
        }

        rar_files = list(base_path.rglob("*.rar"))
        if rar_files:
            arch = rar_files[0]
            result["archive_path"] = str(arch.relative_to(self.root)).replace("\\", "/")
            try:
                out = subprocess.check_output(["tar", "-tf", str(arch)], encoding="utf-8", errors="ignore")
                lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
                img_lines = [line for line in lines if line.lower().endswith(self.SUPPORTED_IMAGE_EXTS)]
                result["total_images"] = len(img_lines)
                result["image_formats"] = sorted(list({Path(line).suffix.lower() for line in img_lines}))

                class_counts: Counter[str] = Counter()
                for line in img_lines:
                    parts = line.split("/")
                    if len(parts) >= 2:
                        cls_name = parts[1]
                        class_counts[cls_name] += 1
                result["classes"] = dict(sorted(class_counts.items()))
            except Exception as e:
                result["issues"].append(f"Archive listing error: {e}")

        return result

    def inspect_riceseg_dataset(self, base_path: Path, sethy_stems: Set[str]) -> Dict[str, Any]:
        """Inspect the RiceSeg5932 dataset and evaluate image-mask pairing against Sethy stems."""
        reg_info = get_dataset_info("riceseg")
        result: Dict[str, Any] = {
            "name": reg_info["name"],
            "role": reg_info["role"],
            "training_allowed": reg_info["training_allowed"],
            "detected_root": str(base_path.relative_to(self.root)).replace("\\", "/"),
            "status": "DISCOVERED",
            "total_masks": 0,
            "mask_formats": [],
            "classes": {},
            "pairing_verification": {
                "total_masks": 0,
                "total_candidate_images": len(sethy_stems),
                "matched_image_mask_pairs": 0,
                "unmatched_images": 0,
                "unmatched_masks": 0,
                "pairing_possible": True,
            },
            "issues": [],
        }

        mask_files = [
            f for f in base_path.rglob("*")
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ]
        result["total_masks"] = len(mask_files)
        result["mask_formats"] = sorted(list({f.suffix.lower() for f in mask_files}))

        class_counts: Counter[str] = Counter()
        mask_stems: Set[str] = set()
        for f in mask_files:
            cls_name = f.parent.name
            class_counts[cls_name] += 1
            mask_stems.add(f.stem)

        result["classes"] = dict(sorted(class_counts.items()))

        # Pairing calculations
        matched = sethy_stems.intersection(mask_stems)
        unmatched_imgs = len(sethy_stems - mask_stems)
        unmatched_msks = len(mask_stems - sethy_stems)

        result["pairing_verification"]["total_masks"] = len(mask_stems)
        result["pairing_verification"]["matched_image_mask_pairs"] = len(matched)
        result["pairing_verification"]["unmatched_images"] = unmatched_imgs
        result["pairing_verification"]["unmatched_masks"] = unmatched_msks
        result["pairing_verification"]["pairing_possible"] = len(matched) > 0

        return result

    def run_full_inspection(self) -> Dict[str, Any]:
        """Execute complete discovery, classification audit, and pairing analysis."""
        discovered = self.find_dataset_folders()
        report: Dict[str, Any] = {
            "project_name": "RiceGuard",
            "phase": "Phase 0 — Research Foundation, Reproducibility Setup, and Existing Dataset Registration",
            "datasets": {},
            "summary": {
                "primary_detected": False,
                "sethy_detected": False,
                "bd5_detected": False,
                "riceseg_detected": False,
            },
        }

        sethy_stems: Set[str] = set()

        if "primary" in discovered:
            report["datasets"]["primary"] = self.inspect_primary_dataset(discovered["primary"]["matched_path"])
            report["summary"]["primary_detected"] = True

        if "sethy" in discovered:
            sethy_report, sethy_stems = self.inspect_sethy_dataset(discovered["sethy"]["matched_path"])
            report["datasets"]["sethy"] = sethy_report
            report["summary"]["sethy_detected"] = True

        if "bd5" in discovered:
            report["datasets"]["bd5"] = self.inspect_bd5_dataset(discovered["bd5"]["matched_path"])
            report["summary"]["bd5_detected"] = True

        if "riceseg" in discovered:
            report["datasets"]["riceseg"] = self.inspect_riceseg_dataset(discovered["riceseg"]["matched_path"], sethy_stems)
            report["summary"]["riceseg_detected"] = True

        return report
