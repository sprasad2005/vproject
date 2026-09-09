"""Lesion-Aware Multi-Task Architecture for RiceGuard.

Combines standard 6-class disease classification with a lightweight 7x7 spatial
lesion localization head supervised by bounding-box annotations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from src.data.grid_assignment import decode_grid_predictions


class RiceMultiTaskEfficientNet(nn.Module):
    """Multi-task EfficientNet-B0 for simultaneous disease classification and lesion localization.

    Architecture:
        - Shared Backbone: EfficientNet-B0 features -> [B, 1280, 7, 7]
        - Classification Head: AdaptiveAvgPool2d(1) -> Flatten -> Dropout(0.2) -> Linear(1280, 6) -> [B, 6]
        - Localization Head: Conv2d(1280, 256, 3, pad=1) -> BatchNorm2d -> SiLU -> Conv2d(256, 5, 1) -> [B, 5, 7, 7]
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        grid_size: int = 7,
        loc_hidden_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.grid_size = grid_size
        self.loc_hidden_dim = loc_hidden_dim

        if architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_channels = 1280
        elif architecture == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b1(weights=weights)
            in_channels = 1280
        else:
            raise ValueError(f"Unsupported backbone architecture: {architecture}")

        # Shared feature extractor
        self.features = base_model.features

        # Classification stream
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_channels, num_classes),
        )

        # Localization stream: outputs 5 channels per cell (raw_obj, raw_dx, raw_dy, raw_w, raw_h)
        self.loc_head = nn.Sequential(
            nn.Conv2d(in_channels, loc_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(loc_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(loc_hidden_dim, 5, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            cls_logits: Classification logits [B, num_classes].
            loc_output: Raw localization grid output [B, 5, grid_size, grid_size].
        """
        # Shared feature map: [B, 1280, 7, 7]
        feat_map = self.features(x)

        # Classification branch
        pooled = self.avgpool(feat_map)
        flat = torch.flatten(pooled, 1)
        cls_logits = self.classifier(flat)

        # Localization branch
        loc_output = self.loc_head(feat_map)

        return cls_logits, loc_output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map [B, 1280, 7, 7]."""
        return self.features(x)

    def get_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature vector [B, 1280]."""
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        return torch.flatten(pooled, 1)

    def predict_boxes(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.5,
    ) -> List[List[Dict[str, Any]]]:
        """Inference helper to decode bounding boxes for a batch of images.

        Args:
            x: Input images [B, 3, 224, 224].
            conf_threshold: Objectness threshold in [0, 1].

        Returns:
            List of lists of detected bounding box dictionaries.
        """
        self.eval()
        with torch.no_grad():
            _, loc_output = self.forward(x)

        batch_boxes: List[List[Dict[str, Any]]] = []
        for i in range(loc_output.shape[0]):
            boxes = decode_grid_predictions(
                loc_output[i],
                conf_threshold=conf_threshold,
                grid_size=self.grid_size,
            )
            batch_boxes.append(boxes)
        return batch_boxes
