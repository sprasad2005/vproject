"""Grad-CAM Explainability Service for RiceGuard Backend."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.xai.gradcam import GradCAM


class XAIService:
    """Generates class-discriminative Grad-CAM attributions for RiceGuard models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.gradcam = GradCAM(model=model)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        orig_width: int,
        orig_height: int,
    ) -> np.ndarray:
        """Generate normalized Grad-CAM heatmap resized to original image dimensions.

        Args:
            input_tensor: Tensor of shape [1, 3, 224, 224].
            target_class: Integer class index to target.
            orig_width: Width of the original image in pixels.
            orig_height: Height of the original image in pixels.

        Returns:
            2D Float NumPy array of shape [orig_height, orig_width] with values in [0.0, 1.0].
        """
        # Generate 224x224 heatmap
        cam_224, _, _ = self.gradcam.generate_cam(
            input_tensor=input_tensor,
            target_class=target_class,
            target_type="classification",
        )

        # Resize to original image resolution with bilinear interpolation
        heatmap_resized = cv2.resize(
            cam_224,
            (orig_width, orig_height),
            interpolation=cv2.INTER_LINEAR,
        )

        return np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)
