"""Tests for the RiceGuard configuration system."""

from __future__ import annotations

import pytest

from src.utils.config import (
    ConfigDict,
    deep_merge,
    load_all_configs,
    load_config,
    load_yaml,
    validate_config,
)
from src.utils.paths import get_configs_dir


def test_load_base_yaml():
    """Verify that base.yaml exists and contains required fields."""
    cfg_dir = get_configs_dir()
    base_file = cfg_dir / "base.yaml"
    assert base_file.exists()

    data = load_yaml(base_file)
    assert isinstance(data, dict)
    assert data["project"]["name"] == "RiceGuard"
    assert data["seed"] == 42
    assert "device" in data


def test_load_config_helper():
    """Verify load_config loads base config and returns ConfigDict."""
    cfg = load_config()
    assert isinstance(cfg, ConfigDict)
    assert cfg.project.name == "RiceGuard"
    assert cfg.seed == 42
    assert cfg.device.use_cuda is True


def test_deep_merge():
    """Verify deep merge correctly overrides nested dictionaries."""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"d": 99, "e": 4}, "f": 5}
    merged = deep_merge(base, override)

    assert merged["a"] == 1
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 99
    assert merged["b"]["e"] == 4
    assert merged["f"] == 5


def test_validate_config():
    """Verify validate_config detects missing required keys."""
    valid_cfg = {"project": {"name": "RiceGuard"}, "seed": 42, "device": {"use_cuda": True}}
    assert validate_config(valid_cfg) is True

    invalid_cfg = {"seed": 42}
    with pytest.raises(ValueError, match="Missing required configuration key"):
        validate_config(invalid_cfg)


def test_load_all_configs():
    """Verify load_all_configs unifies all component configurations cleanly."""
    unified = load_all_configs()
    assert hasattr(unified, "project")
    assert hasattr(unified, "datasets") or hasattr(unified, "primary_dataset")
    assert hasattr(unified, "models")
    assert hasattr(unified, "training")
    assert hasattr(unified, "calibration")
    assert hasattr(unified, "uncertainty")
    assert hasattr(unified, "xai")
    assert hasattr(unified, "robustness")
    assert hasattr(unified, "paths")
