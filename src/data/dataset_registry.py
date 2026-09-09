"""Dataset registry and access control policies for RiceGuard.

Enforces strict isolation between primary training data, external benchmark datasets,
and XAI ground-truth segmentation masks to prevent data leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Master Dataset Access Control & Governance Registry
DATASETS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "name": "RiceLeafDiseaseBD",
        "role": "primary_training",
        "training_allowed": True,
        "validation_allowed": True,
        "internal_test_allowed": True,
        "calibration_allowed": True,
        "threshold_optimization_allowed": True,
        "xai_ground_truth": False,
        "description": "Primary dataset for training, validation, internal test, calibration, threshold optimization, and lesion supervision.",
        "classes": [
            "Healthy",
            "Blast",
            "Brown Spot",
            "Leaf Smut",
            "Tungro",
            "Sheath Blight",
        ],
    },
    "sethy": {
        "name": "Sethy5932",
        "role": "external_test",
        "training_allowed": False,
        "validation_allowed": False,
        "internal_test_allowed": False,
        "calibration_allowed": False,
        "threshold_optimization_allowed": False,
        "xai_ground_truth": False,
        "description": "External benchmark for cross-dataset generalization and domain shift.",
        "classes": [
            "Bacterial Blight",
            "Blast",
            "Brown Spot",
            "Tungro",
        ],
    },
    "bd5": {
        "name": "RiceLeafDiseaseBD5",
        "role": "external_field_test",
        "training_allowed": False,
        "validation_allowed": False,
        "internal_test_allowed": False,
        "calibration_allowed": False,
        "threshold_optimization_allowed": False,
        "xai_ground_truth": False,
        "description": "External field benchmark. Narrow Brown Spot is strictly preserved.",
        "classes": [
            "Blast",
            "Narrow Brown Spot",
            "Sheath Blight",
            "Tungro",
            "Normal",
        ],
    },
    "riceseg": {
        "name": "RiceSeg5932",
        "role": "xai_ground_truth",
        "training_allowed": False,
        "validation_allowed": False,
        "internal_test_allowed": False,
        "calibration_allowed": False,
        "threshold_optimization_allowed": False,
        "xai_ground_truth": True,
        "description": "Pixel-level lesion masks strictly for XAI evaluation (IoU, Dice).",
        "classes": [
            "Background",
            "Lesion",
        ],
    },
}


def get_dataset_info(dataset_key: str) -> Dict[str, Any]:
    """Retrieve metadata and security policies for a registered dataset."""
    key = dataset_key.lower().strip()
    if key not in DATASETS:
        raise KeyError(f"Dataset '{dataset_key}' not found in registry. Valid keys: {list(DATASETS.keys())}")
    return DATASETS[key]


def is_training_allowed(dataset_key: str) -> bool:
    """Check if training is permitted on the specified dataset."""
    info = get_dataset_info(dataset_key)
    return bool(info.get("training_allowed", False))


def is_validation_allowed(dataset_key: str) -> bool:
    """Check if validation/tuning is permitted on the specified dataset."""
    info = get_dataset_info(dataset_key)
    return bool(info.get("validation_allowed", False))


def is_external(dataset_key: str) -> bool:
    """Check if the dataset is an external benchmark."""
    info = get_dataset_info(dataset_key)
    return "external" in info.get("role", "")


def is_xai_ground_truth(dataset_key: str) -> bool:
    """Check if dataset contains ground-truth XAI masks."""
    info = get_dataset_info(dataset_key)
    return bool(info.get("xai_ground_truth", False))


def validate_dataset_usage(dataset_key: str, requested_usage: str) -> bool:
    """Enforce access policies, raising PermissionError on violation.

    Args:
        dataset_key: Dataset identifier ('primary', 'sethy', 'bd5', 'riceseg').
        requested_usage: One of 'train', 'val', 'test', 'calibration', 'threshold_opt', 'xai_eval'.

    Returns:
        bool: True if permitted.

    Raises:
        PermissionError: If usage violates dataset role constraints.
    """
    info = get_dataset_info(dataset_key)
    usage = requested_usage.lower().strip()

    if ("mask_training" in usage or "segmentation_supervision" in usage) and info["xai_ground_truth"]:
        raise PermissionError(
            f"SECURITY POLICY VIOLATION: Using segmentation masks for model training is prohibited on '{info['name']}'."
        )

    train_tokens = ["train", "training", "primary_training", "finetune", "fit"]
    if any(t in usage.split("_") or usage == t for t in train_tokens) and not info.get("training_allowed", False):
        raise PermissionError(
            f"SECURITY POLICY VIOLATION: Training is strictly prohibited on external dataset '{info['name']}'."
        )

    val_tokens = ["val", "validation", "tune", "primary_validation", "model_selection", "hyperparameter_tuning"]
    if any(t in usage.split("_") or usage == t for t in val_tokens) and not info.get("validation_allowed", False):
        raise PermissionError(
            f"SECURITY POLICY VIOLATION: Validation/tuning is strictly prohibited on external dataset '{info['name']}'."
        )

    calib_tokens = ["calibration", "calibrate"]
    if any(t in usage.split("_") or usage == t for t in calib_tokens) and not info.get("calibration_allowed", False):
        raise PermissionError(
            f"SECURITY POLICY VIOLATION: Calibration fitting is strictly prohibited on external dataset '{info['name']}'."
        )

    thresh_tokens = ["threshold_opt", "threshold_optimization", "topk_optimization"]
    if any(t in usage.split("_") or usage == t for t in thresh_tokens) and not info.get("threshold_optimization_allowed", False):
        raise PermissionError(
            f"SECURITY POLICY VIOLATION: Threshold optimization is strictly prohibited on external dataset '{info['name']}'."
        )

    return True


def list_registered_datasets() -> List[str]:
    """Return all registered dataset keys."""
    return list(DATASETS.keys())
