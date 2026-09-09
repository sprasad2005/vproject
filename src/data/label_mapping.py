"""Standardized label mapping and canonical class governance for RiceGuard.

Maintains canonical class representations across primary and external benchmarks
while strictly preserving semantic distinctions (e.g. Narrow Brown Spot != Brown Spot).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Canonical Target Disease Classes for Primary Framework (6 Classes)
CANONICAL_PRIMARY_CLASSES: List[str] = [
    "Healthy",
    "Blast",
    "Brown Spot",
    "Leaf Smut",
    "Tungro",
    "Sheath Blight",
]

# Canonical Index Mappings
CANONICAL_CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_PRIMARY_CLASSES)}
CANONICAL_IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CANONICAL_PRIMARY_CLASSES)}

# Raw to Canonical Mapping Dictionaries
PRIMARY_RAW_TO_CANONICAL: Dict[str, str] = {
    "blast": "Blast",
    "brown spot": "Brown Spot",
    "brownspot": "Brown Spot",
    "healthy": "Healthy",
    "leaf smut": "Leaf Smut",
    "leafsmut": "Leaf Smut",
    "rice tungro": "Tungro",
    "tungro": "Tungro",
    "sheath blight": "Sheath Blight",
    "sheathblight": "Sheath Blight",
}

SETHY_RAW_TO_CANONICAL: Dict[str, str] = {
    "bacterialblight": "Bacterial Blight",
    "bacterial blight": "Bacterial Blight",
    "blast": "Blast",
    "brownspot": "Brown Spot",
    "brown spot": "Brown Spot",
    "tungro": "Tungro",
}

BD5_RAW_TO_CANONICAL: Dict[str, str] = {
    "normal_leaf": "Healthy",
    "normal": "Healthy",
    "blast": "Blast",
    "sheath_blight": "Sheath Blight",
    "sheath blight": "Sheath Blight",
    "tungro": "Tungro",
    "narrow_brown_spot": "Narrow Brown Spot",
    "narrow brown spot": "Narrow Brown Spot",
}

RICESEG_RAW_TO_CANONICAL: Dict[str, str] = {
    "bacterialblight": "Bacterial Blight",
    "blast": "Blast",
    "brownspot": "Brown Spot",
    "tungro": "Tungro",
}

# YOLO Class ID to Canonical Disease Name mapping (Primary dataset)
YOLO_ID_TO_CANONICAL: Dict[int, str] = {
    0: "Blast",
    1: "Brown Spot",
    2: "Leaf Smut",
    3: "Tungro",
    4: "Sheath Blight",
}


def normalize_class_name(
    raw_class: str,
    dataset_name: str = "primary",
) -> Tuple[str, str]:
    """Map a raw dataset class string to its canonical representation.

    Args:
        raw_class: The raw folder/label string from the dataset.
        dataset_name: Dataset identifier ('primary', 'sethy', 'bd5', 'riceseg').

    Returns:
        Tuple[str, str]: (canonical_class, mapping_status)
    """
    cleaned = raw_class.strip().lower().replace("-", "_")
    dname = dataset_name.lower().strip()

    if "primary" in dname or "riceleafdiseasebd" in dname:
        mapping = PRIMARY_RAW_TO_CANONICAL
    elif "sethy" in dname:
        mapping = SETHY_RAW_TO_CANONICAL
    elif "bd5" in dname:
        mapping = BD5_RAW_TO_CANONICAL
    elif "seg" in dname:
        mapping = RICESEG_RAW_TO_CANONICAL
    else:
        mapping = {**PRIMARY_RAW_TO_CANONICAL, **SETHY_RAW_TO_CANONICAL, **BD5_RAW_TO_CANONICAL}

    # Match normalized key
    if cleaned in mapping:
        return mapping[cleaned], "CANONICAL_MAPPED"

    # Fallback to key without underscores
    cleaned_no_underscore = cleaned.replace("_", " ")
    if cleaned_no_underscore in mapping:
        return mapping[cleaned_no_underscore], "CANONICAL_MAPPED"

    return raw_class, "UNMAPPED_RAW"


def get_canonical_class_list(dataset_name: str = "primary") -> List[str]:
    """Return sorted unique canonical classes for a given dataset."""
    dname = dataset_name.lower().strip()
    if "primary" in dname:
        return CANONICAL_PRIMARY_CLASSES
    elif "sethy" in dname:
        return ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]
    elif "bd5" in dname:
        return ["Blast", "Healthy", "Narrow Brown Spot", "Sheath Blight", "Tungro"]
    elif "seg" in dname:
        return ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]
    return CANONICAL_PRIMARY_CLASSES


def class_to_index(canonical_class: str) -> Optional[int]:
    """Return primary class index or None if class is out-of-framework."""
    return CANONICAL_CLASS_TO_IDX.get(canonical_class)


def index_to_class(index: int) -> Optional[str]:
    """Return canonical class name from index."""
    return CANONICAL_IDX_TO_CLASS.get(index)
