"""Vision Transformer (ViT) baseline model implementation for RiceGuard."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


class RiceViT(nn.Module):
    """Vision Transformer classifier for rice leaf disease detection.

    Uses standard torchvision ViT-B/16 backbone, preserving pretrained weights
    and replacing the final classification head for the canonical 6 disease classes.
    """

    def __init__(
        self,
        architecture: str = "vit_b_16",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained

        if architecture in ["vit_b_16", "vit"]:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
        elif architecture == "vit_b_32":
            weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_32(weights=weights)
        elif architecture == "vit_l_16":
            weights = models.ViT_L_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_l_16(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT architecture: {architecture}")

        # In torchvision ViT, heads is an OrderedDict / Sequential with 'head'
        in_features = self.backbone.heads.head.in_features

        if dropout_rate > 0.0:
            self.backbone.heads.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract transformer representation before classification head."""
        # Reshape and permute input tensor
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        # Classifier "token" as used by standard ViT
        x = x[:, 0]
        return x
