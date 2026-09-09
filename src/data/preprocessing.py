"""Image preprocessing pipelines and normalization transforms for RiceGuard."""

from __future__ import annotations

from typing import Tuple


def get_default_normalization() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return standard ImageNet mean and std for model backbones."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std
