"""EfficientNet baseline model implementation for RiceGuard."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torchvision.models as models


class RiceEfficientNet(nn.Module):
    """EfficientNet-B0 classifier for rice leaf disease detection.

    Uses standard torchvision EfficientNet-B0 backbone, preserving pretrained weights
    and replacing the final linear classifier layer for 6 disease classes.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained

        if architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        elif architecture == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet architecture: {architecture}")

        # The classifier in torchvision efficientnet is Sequential(Dropout(p=0.2), Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features
        p_drop = dropout_rate if dropout_rate is not None else self.backbone.classifier[0].p

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=p_drop, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings before classification head."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
