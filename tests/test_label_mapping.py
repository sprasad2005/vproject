"""Tests for canonical label mappings and semantic preservation."""

from __future__ import annotations

from src.data.label_mapping import (
    CANONICAL_PRIMARY_CLASSES,
    class_to_index,
    index_to_class,
    normalize_class_name,
)


def test_canonical_classes_count():
    """Verify primary framework defines exactly 6 canonical classes."""
    assert len(CANONICAL_PRIMARY_CLASSES) == 6
    assert "Healthy" in CANONICAL_PRIMARY_CLASSES
    assert "Blast" in CANONICAL_PRIMARY_CLASSES
    assert "Brown Spot" in CANONICAL_PRIMARY_CLASSES
    assert "Leaf Smut" in CANONICAL_PRIMARY_CLASSES
    assert "Tungro" in CANONICAL_PRIMARY_CLASSES
    assert "Sheath Blight" in CANONICAL_PRIMARY_CLASSES


def test_primary_label_normalization():
    """Verify primary dataset raw folder strings map to canonical names."""
    assert normalize_class_name("Blast", "primary")[0] == "Blast"
    assert normalize_class_name("Brown spot", "primary")[0] == "Brown Spot"
    assert normalize_class_name("Healthy", "primary")[0] == "Healthy"
    assert normalize_class_name("Leaf smut", "primary")[0] == "Leaf Smut"
    assert normalize_class_name("Rice Tungro", "primary")[0] == "Tungro"
    assert normalize_class_name("Sheath blight", "primary")[0] == "Sheath Blight"


def test_bd5_narrow_brown_spot_preservation():
    """Verify Narrow Brown Spot is strictly preserved as distinct and not mapped to Brown Spot."""
    canonical, _ = normalize_class_name("Narrow_Brown_Spot", "bd5")
    assert canonical == "Narrow Brown Spot"
    assert canonical != "Brown Spot"

    canonical_space, _ = normalize_class_name("Narrow Brown Spot", "bd5")
    assert canonical_space == "Narrow Brown Spot"


def test_sethy_label_normalization():
    """Verify Sethy raw class names map correctly."""
    assert normalize_class_name("Bacterialblight", "sethy")[0] == "Bacterial Blight"
    assert normalize_class_name("Blast", "sethy")[0] == "Blast"
    assert normalize_class_name("Brownspot", "sethy")[0] == "Brown Spot"
    assert normalize_class_name("Tungro", "sethy")[0] == "Tungro"


def test_index_mappings():
    """Verify index to class bi-directional mappings."""
    for idx, cname in enumerate(CANONICAL_PRIMARY_CLASSES):
        assert class_to_index(cname) == idx
        assert index_to_class(idx) == cname

    assert class_to_index("UnknownClassXYZ") is None
