"""Lesion-Aware Attention Supervised Network architecture for RiceGuard."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn


class LesionAwareNet(nn.Module):
    """Lesion-grounded attention network integrating bounding box supervision signals."""

    def __init__(
        self,
        base_architecture: str = "resnet50",
        num_classes: int = 6,
        pretrained: bool = True,
        attention_type: str = "spatial_channel_lesion_guided",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.base_architecture = base_architecture
        self.num_classes = num_classes
        self.attention_type = attention_type

    def forward(self, x, bbox_targets: Optional[Any] = None):
        raise NotImplementedError("LesionAwareNet forward pass will be implemented in future phase.")
