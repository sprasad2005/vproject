"""Tests for YOLO bounding-box parsing, validation, and coordinate scaling."""

from __future__ import annotations

from src.data.annotations import BoundingBox, parse_yolo_bbox_content


def test_valid_bounding_box():
    """Verify valid normalized YOLO coordinates pass validation."""
    box = BoundingBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.3, img_width=1000, img_height=1000)
    assert box.is_valid is True
    assert box.canonical_class == "Blast"
    assert abs(box.normalized_area - 0.06) < 1e-5
    coords = box.pixel_coordinates
    assert coords == (400, 350, 600, 650)


def test_invalid_bounding_box_out_of_bounds():
    """Verify out-of-bound coordinates are caught by validator."""
    box_bad_center = BoundingBox(class_id=1, x_center=1.5, y_center=0.5, width=0.2, height=0.2)
    assert box_bad_center.is_valid is False

    box_bad_dim = BoundingBox(class_id=1, x_center=0.5, y_center=0.5, width=0.0, height=0.2)
    assert box_bad_dim.is_valid is False


def test_parse_yolo_bbox_content():
    """Verify parsing multi-line YOLO text content."""
    content = """
    0 0.528809 0.181152 0.079102 0.110352
    0 0.494629 0.285156 0.049805 0.044922
    """
    boxes, errors = parse_yolo_bbox_content(content, img_width=1024, img_height=1024)
    assert len(boxes) == 2
    assert len(errors) == 0
    assert boxes[0].class_id == 0
    assert boxes[0].canonical_class == "Blast"
    assert boxes[1].class_id == 0
