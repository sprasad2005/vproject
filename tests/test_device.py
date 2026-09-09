"""Tests for PyTorch device management and hardware diagnostic reporting."""

from __future__ import annotations

import torch

from src.utils.device import get_device, get_device_info


def test_get_device_returns_valid_torch_device():
    """Verify get_device returns a torch.device instance (cuda or cpu)."""
    device = get_device(verbose=False)
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu"]


def test_get_device_fallback_when_cuda_disabled():
    """Verify device returns CPU when use_cuda is explicitly False."""
    device = get_device(use_cuda=False, verbose=False)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_get_device_info_structure():
    """Verify get_device_info returns structured dictionary with expected keys."""
    device = get_device(verbose=False)
    info = get_device_info(device)
    assert isinstance(info, dict)
    assert "device_type" in info
    assert "is_cuda" in info
    assert info["device_type"] in ["cuda", "cpu"]
