"""Image integrity validation, corruption detection, and duplicate auditing engine for RiceGuard."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image, ImageFile

# Ensure truncated images do not crash PIL during validation
ImageFile.LOAD_TRUNCATED_IMAGES = False


def calculate_sha256(file_path_or_bytes: Union[str, Path, bytes]) -> str:
    """Compute SHA-256 hex digest for a file path or raw bytes."""
    if isinstance(file_path_or_bytes, bytes):
        return hashlib.sha256(file_path_or_bytes).hexdigest()

    path = Path(file_path_or_bytes)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_or_path: Union[str, Path, Image.Image], hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for near-duplicate image detection.

    Args:
        image_or_path: PIL Image or path to image.
        hash_size: Grid width (default 8 yields 64-bit hash).

    Returns:
        str: 64-bit hexadecimal hash string.
    """
    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("L")
    else:
        img = image_or_path.convert("L")

    # Fast bilinear downsampling to (hash_size + 1, hash_size)
    resized = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)

    # Compute horizontal difference gradient
    diff = pixels[:, 1:] > pixels[:, :-1]

    # Convert boolean array to hex string
    bit_str = "".join("1" if b else "0" for b in diff.flatten())
    hex_str = f"{int(bit_str, 2):0{hash_size * hash_size // 4}x}"
    return hex_str


def hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Compute Hamming distance between two hexadecimal hash strings."""
    val1 = int(hash1_hex, 16)
    val2 = int(hash2_hex, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")


def validate_image_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect and validate image integrity in a fast, single-pass operation.

    Returns:
        Dict[str, Any]: Detailed image metadata and validity status.
    """
    path = Path(file_path)
    result = {
        "file_path": str(path),
        "exists": path.exists(),
        "is_valid": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "format": "",
        "mode": "",
        "file_size": 0,
        "sha256": "",
        "dhash": "",
        "error_message": "",
    }

    if not path.exists():
        result["error_message"] = "File does not exist"
        return result

    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        result["file_size"] = len(raw_bytes)
        if result["file_size"] == 0:
            result["error_message"] = "Zero-byte file"
            return result

        result["sha256"] = hashlib.sha256(raw_bytes).hexdigest()

        with Image.open(io.BytesIO(raw_bytes)) as img:
            result["width"], result["height"] = img.size
            result["format"] = img.format or path.suffix.upper().replace(".", "")
            result["mode"] = img.mode
            result["channels"] = len(img.getbands())

            # Fast single-pass dHash
            gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = np.asarray(gray, dtype=np.float32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            bit_str = "".join("1" if b else "0" for b in diff.flatten())
            result["dhash"] = f"{int(bit_str, 2):016x}"

        # Sanity checks on dimensions
        if result["width"] < 16 or result["height"] < 16:
            result["error_message"] = f"Extremely small image dimensions: {result['width']}x{result['height']}"
            result["is_valid"] = False
        else:
            result["is_valid"] = True
            result["error_message"] = "OK"

    except Exception as e:
        result["is_valid"] = False
        result["error_message"] = f"Corrupted or unreadable image: {e}"

    return result


class DuplicateAuditor:
    """Audits exact and near-duplicate images across and within datasets."""

    def __init__(self, near_dup_threshold: int = 4) -> None:
        self.near_dup_threshold = near_dup_threshold
        self.records: List[Dict[str, Any]] = []

    def add_record(
        self,
        image_id: str,
        dataset: str,
        canonical_class: str,
        sha256: str,
        dhash: str,
        file_path: str,
    ) -> None:
        self.records.append(
            {
                "image_id": image_id,
                "dataset": dataset,
                "canonical_class": canonical_class,
                "sha256": sha256,
                "dhash": dhash,
                "file_path": file_path,
            }
        )

    def find_exact_duplicates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group records by identical SHA-256 hash."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.records:
            h = r["sha256"]
            if not h:
                continue
            groups.setdefault(h, []).append(r)
        return {h: recs for h, recs in groups.items() if len(recs) > 1}

    def find_near_duplicates(self) -> List[Dict[str, Any]]:
        """Identify candidate pairs with Hamming distance <= threshold."""
        near_dups: List[Dict[str, Any]] = []
        n = len(self.records)

        # For large datasets, compare within reasonable buckets
        for i in range(n):
            r1 = self.records[i]
            h1 = r1["dhash"]
            if not h1:
                continue
            for j in range(i + 1, min(i + 500, n)):
                r2 = self.records[j]
                h2 = r2["dhash"]
                if not h2:
                    continue
                # Skip exact SHA256 matches (handled separately)
                if r1["sha256"] == r2["sha256"]:
                    continue

                dist = hamming_distance(h1, h2)
                if dist <= self.near_dup_threshold:
                    near_dups.append(
                        {
                            "dataset_a": r1["dataset"],
                            "image_a": r1["image_id"],
                            "class_a": r1["canonical_class"],
                            "dataset_b": r2["dataset"],
                            "image_b": r2["image_id"],
                            "class_b": r2["canonical_class"],
                            "hamming_distance": dist,
                            "is_cross_dataset": r1["dataset"] != r2["dataset"],
                            "same_class": r1["canonical_class"] == r2["canonical_class"],
                        }
                    )

        return near_dups
