"""Tests for device detection, path management, and environment foundation."""

from __future__ import annotations

import torch

from src.utils.device import get_device, get_device_info
from src.utils.paths import (
    get_checkpoints_dir,
    get_configs_dir,
    get_data_dir,
    get_experiments_dir,
    get_logs_dir,
    get_project_root,
    get_results_dir,
)


def test_device_detection():
    """Verify that get_device returns a valid PyTorch device."""
    dev = get_device(verbose=False)
    assert isinstance(dev, torch.device)
    assert dev.type in ["cuda", "cpu"]

    info = get_device_info(dev)
    assert isinstance(info, dict)
    assert "device_type" in info
    assert "is_cuda" in info


def test_path_resolution():
    """Verify that all core project path helpers resolve to valid existing directories."""
    root = get_project_root()
    assert root.exists()
    assert (root / "README.md").exists()

    assert get_configs_dir().exists()
    assert get_data_dir().exists()
    assert get_experiments_dir().exists()
    assert get_logs_dir().exists()
    assert get_checkpoints_dir().exists()
    assert get_results_dir().exists()
