"""Tests for seed setting and reproducibility info collection."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.seed import get_reproducibility_info, set_seed


def test_set_seed_reproducibility():
    """Verify that set_seed ensures deterministic random number generation."""
    set_seed(1234)
    r1 = random.random()
    np1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(1234)
    r2 = random.random()
    np2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert r1 == r2
    assert np.allclose(np1, np2)
    assert torch.allclose(t1, t2)


def test_get_reproducibility_info():
    """Verify get_reproducibility_info returns structured environment dictionary."""
    info = get_reproducibility_info()
    assert isinstance(info, dict)
    assert "python_version" in info
    assert "pytorch_version" in info
    assert "numpy_version" in info
    assert "cuda_available" in info
    assert "operating_system" in info
