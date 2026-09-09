"""ResNet baseline model implementation for RiceGuard."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


class RiceResNet(nn.Module):
    """ResNet50 classifier for rice leaf disease detection.

    Uses standard torchvision ResNet50 backbone, preserving pretrained weights
    and replacing the final classification head for the canonical 6 disease classes.
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load standard torchvision ResNet
        if architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {architecture}")

        in_features = self.backbone.fc.in_features

        # Replace classification head cleanly
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings before classification head."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
