"""YOLO bounding-box annotation parsing, coordinate validation, and statistics engine for RiceGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.data.label_mapping import YOLO_ID_TO_CANONICAL


class BoundingBox:
    """Represents a normalized and pixel-scaled bounding box."""

    def __init__(
        self,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> None:
        self.class_id = int(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)
        self.img_width = img_width
        self.img_height = img_height
        self.canonical_class = YOLO_ID_TO_CANONICAL.get(self.class_id, f"class_{self.class_id}")

        # Validation status
        self.is_valid, self.validation_message = self._validate()

    def _validate(self) -> Tuple[bool, str]:
        """Validate that normalized coordinates strictly reside within [0, 1]."""
        if not (0.0 <= self.x_center <= 1.0):
            return False, f"x_center {self.x_center} out of bounds [0, 1]"
        if not (0.0 <= self.y_center <= 1.0):
            return False, f"y_center {self.y_center} out of bounds [0, 1]"
        if not (0.0 < self.width <= 1.0):
            return False, f"width {self.width} out of bounds (0, 1]"
        if not (0.0 < self.height <= 1.0):
            return False, f"height {self.height} out of bounds (0, 1]"

        # Check bounds with minor tolerance for floating precision
        x_min = self.x_center - (self.width / 2.0)
        x_max = self.x_center + (self.width / 2.0)
        y_min = self.y_center - (self.height / 2.0)
        y_max = self.y_center + (self.height / 2.0)

        eps = 1e-4
        if x_min < -eps or x_max > 1.0 + eps or y_min < -eps or y_max > 1.0 + eps:
            return False, f"Bounding box extends outside image boundaries: [{x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f}]"

        return True, "VALID"

    @property
    def normalized_area(self) -> float:
        """Return box area as fraction of total image area [0, 1]."""
        return self.width * self.height

    @property
    def pixel_coordinates(self) -> Optional[Tuple[int, int, int, int]]:
        """Return (x_min, y_min, x_max, y_max) in absolute pixel units."""
        if self.img_width is None or self.img_height is None:
            return None
        x_min = max(0, int((self.x_center - (self.width / 2.0)) * self.img_width))
        y_min = max(0, int((self.y_center - (self.height / 2.0)) * self.img_height))
        x_max = min(self.img_width, int((self.x_center + (self.width / 2.0)) * self.img_width))
        y_max = min(self.img_height, int((self.y_center + (self.height / 2.0)) * self.img_height))
        return (x_min, y_min, x_max, y_max)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "canonical_class": self.canonical_class,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "normalized_area": self.normalized_area,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "pixel_coordinates": self.pixel_coordinates,
        }


def parse_yolo_bbox_content(
    content_str: str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Tuple[List[BoundingBox], List[str]]:
    """Parse raw text string containing YOLO annotation lines.

    Format per line: <class_id> <x_center> <y_center> <width> <height>
    """
    boxes: List[BoundingBox] = []
    errors: List[str] = []

    lines = [line.strip() for line in content_str.strip().split("\n") if line.strip()]
    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: Expected 5 fields, found {len(parts)} ('{line}')")
            continue
        try:
            cid = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            box = BoundingBox(cid, xc, yc, w, h, img_width=img_width, img_height=img_height)
            if not box.is_valid:
                errors.append(f"Line {idx}: {box.validation_message}")
            boxes.append(box)
        except ValueError as e:
            errors.append(f"Line {idx}: Parsing error: {e}")

    return boxes, errors


def parse_yolo_bbox_file(
    file_path: Union[str, Path],
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Tuple[List[BoundingBox], List[str]]:
    """Read and parse a YOLO annotation text file from disk."""
    path = Path(file_path)
    if not path.exists():
        return [], [f"File not found: {path}"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return parse_yolo_bbox_content(content, img_width=img_width, img_height=img_height)
    except Exception as e:
        return [], [f"Failed to read file {path}: {e}"]


def parse_bounding_box_annotation(annotation_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """Legacy parser interface returning list of box dictionaries."""
    boxes, _ = parse_yolo_bbox_file(annotation_file)
    return [b.to_dict() for b in boxes]


def parse_segmentation_mask(mask_file: Union[str, Path]) -> Any:
    """Parse binary/grayscale lesion segmentation masks."""
    path = Path(mask_file)
    if not path.exists():
        return None
    try:
        import cv2
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
