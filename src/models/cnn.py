"""Base CNN architecture builder and factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_cnn_backbone(
    architecture_name: str,
    num_classes: int = 6,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    **kwargs: Any,
) -> nn.Module:
    """Factory interface for instantiating CNN backbones."""
    raise NotImplementedError("Model architectures will be implemented in Phase 1.")
