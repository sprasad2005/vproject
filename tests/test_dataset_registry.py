"""Tests for dataset governance, security protection policies, and access controls."""

from __future__ import annotations

import pytest

from src.data.dataset_registry import (
    get_dataset_info,
    is_external,
    is_training_allowed,
    is_validation_allowed,
    is_xai_ground_truth,
    list_registered_datasets,
    validate_dataset_usage,
)


def test_registered_dataset_keys():
    """Verify all 4 required datasets are registered."""
    keys = list_registered_datasets()
    assert "primary" in keys
    assert "sethy" in keys
    assert "bd5" in keys
    assert "riceseg" in keys


def test_primary_dataset_permissions():
    """Verify primary dataset permits training, validation, and internal testing."""
    assert is_training_allowed("primary") is True
    assert is_validation_allowed("primary") is True
    assert is_external("primary") is False
    assert is_xai_ground_truth("primary") is False

    # Should not raise
    assert validate_dataset_usage("primary", "train") is True
    assert validate_dataset_usage("primary", "val") is True


def test_sethy_training_prohibition():
    """Verify Sethy5932 strictly prohibits training and validation."""
    assert is_training_allowed("sethy") is False
    assert is_validation_allowed("sethy") is False
    assert is_external("sethy") is True

    with pytest.raises(PermissionError, match="Training is strictly prohibited"):
        validate_dataset_usage("sethy", "train")

    with pytest.raises(PermissionError, match="Validation/tuning is strictly prohibited"):
        validate_dataset_usage("sethy", "val")


def test_bd5_training_prohibition():
    """Verify RiceLeafDiseaseBD5 strictly prohibits training and validation."""
    assert is_training_allowed("bd5") is False
    assert is_validation_allowed("bd5") is False
    assert is_external("bd5") is True

    with pytest.raises(PermissionError, match="Training is strictly prohibited"):
        validate_dataset_usage("bd5", "train")

    with pytest.raises(PermissionError, match="Validation/tuning is strictly prohibited"):
        validate_dataset_usage("bd5", "val")


def test_riceseg_training_prohibition():
    """Verify RiceSeg5932 is strictly marked as XAI ground truth and prohibits classifier training."""
    assert is_training_allowed("riceseg") is False
    assert is_validation_allowed("riceseg") is False
    assert is_xai_ground_truth("riceseg") is True

    with pytest.raises(PermissionError, match="Training is strictly prohibited"):
        validate_dataset_usage("riceseg", "train")

    with pytest.raises(PermissionError, match="Using segmentation masks for model training is prohibited"):
        validate_dataset_usage("riceseg", "mask_training")


def test_calibration_and_threshold_optimization_blocking():
    """Verify calibration and threshold optimization are prohibited on external datasets."""
    assert validate_dataset_usage("primary", "calibration") is True
    assert validate_dataset_usage("primary", "threshold_opt") is True

    for key in ["sethy", "bd5", "riceseg"]:
        with pytest.raises(PermissionError, match="Calibration fitting is strictly prohibited"):
            validate_dataset_usage(key, "calibration")
        with pytest.raises(PermissionError, match="Threshold optimization is strictly prohibited"):
            validate_dataset_usage(key, "threshold_opt")



def test_invalid_dataset_key():
    """Verify requesting an unknown dataset raises KeyError."""
    with pytest.raises(KeyError):
        get_dataset_info("unknown_dataset_xyz")
