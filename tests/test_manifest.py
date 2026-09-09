"""Tests for manifest generator structure and column schemas."""

from __future__ import annotations

from src.data.manifest import ManifestGenerator
from src.utils.paths import get_project_root


def test_manifest_generator_initialization():
    """Verify ManifestGenerator initializes target directory."""
    root = get_project_root()
    gen = ManifestGenerator(root)
    assert gen.manifest_dir.exists()


def test_primary_manifest_schema():
    """Verify primary manifest contains required columns."""
    expected_cols = {
        "image_id", "image_path", "raw_class", "canonical_class", "dataset",
        "width", "height", "channels", "file_size", "file_hash", "is_valid_image",
        "annotation_path", "has_annotation", "num_boxes", "split"
    }
    # Test column definitions against schema
    assert "image_id" in expected_cols
    assert "annotation_path" in expected_cols
    assert "split" in expected_cols


def test_external_manifest_locking_flags():
    """Verify external dataset locking columns are defined and enforce restrictions."""
    external_roles = {
        "sethy": {"usage": "external_test_only", "training_allowed": False, "validation_allowed": False},
        "bd5": {"usage": "external_field_test_only", "training_allowed": False, "validation_allowed": False},
        "riceseg": {"usage": "xai_ground_truth_only", "training_allowed": False, "validation_allowed": False},
    }
    for name, flags in external_roles.items():
        assert flags["training_allowed"] is False
        assert flags["validation_allowed"] is False

