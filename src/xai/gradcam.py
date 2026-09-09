"""Grad-CAM explanation generator for RiceGuard Phase 4 XAI Validation.

Computes class-discriminative spatial attribution maps for classification backbones
and multi-task architectures. Supports deterministic min-max normalization and
safe handling of degenerate/constant attribution maps.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_default_target_layer(model: nn.Module) -> nn.Module:
    """Find the default final convolutional layer for EfficientNet or ResNet architectures."""
    # Check for RiceEfficientNet wrapper
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        return model.backbone.features[-1]
    # Check for RiceMultiTaskEfficientNet
    if hasattr(model, "features"):
        return model.features[-1]
    # Check for ResNet
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]

    # Fallback: search modules in reverse for the last Conv2d or Conv2dNormActivation
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise ValueError("Could not automatically locate final convolutional layer in model.")


class GradCAM:
    """Grad-CAM Explainer for PyTorch models.

    Supports both single-output classifiers (Phase 2B) and multi-task models (Phase 3B).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer if target_layer is not None else find_default_target_layer(model)

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_type: str = "classification",  # "classification" or "localization"
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM attribution map for an input image.

        Args:
            input_tensor: Tensor of shape [1, 3, H, W] or [3, H, W].
            target_class: Optional target class index. If None, targets the predicted class.
            target_type: "classification" (primary) or "localization" (objectness peak).

        Returns:
            attribution_map: Normalized 2D numpy array [H, W] in [0, 1].
            predicted_class: Integer index of predicted class.
            confidence: Probability score for the target class.
        """
        self.model.eval()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        # Clear gradients
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            cls_logits, loc_output = output
        else:
            cls_logits = output
            loc_output = None

        probs = F.softmax(cls_logits, dim=1)
        pred_class_idx = int(torch.argmax(probs, dim=1).item())

        if target_type == "classification":
            target_idx = pred_class_idx if target_class is None else target_class
            confidence = float(probs[0, target_idx].item())
            target_score = cls_logits[0, target_idx]
        elif target_type == "localization":
            if loc_output is None:
                raise ValueError("Model does not provide localization output for localization attribution.")
            # Target highest objectness logit across the 7x7 grid
            obj_map = loc_output[0, 0]  # [7, 7]
            max_r, max_c = torch.unravel_index(torch.argmax(obj_map), obj_map.shape)
            target_score = obj_map[max_r, max_c]
            confidence = float(torch.sigmoid(target_score).item())
            target_idx = pred_class_idx
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        # Backward pass
        target_score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients during Grad-CAM backward pass.")

        # Compute channel weights: GAP over gradients [1, C, H_feat, W_feat] -> [1, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination: [1, C, H_feat, W_feat] -> [1, 1, H_feat, W_feat]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(cam)

        # Bilinear interpolation to match input image resolution
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)

        # Deterministic Min-Max Normalization to [0, 1]
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        denom = cam_max - cam_min

        if denom > 1e-8:
            norm_cam = (cam_np - cam_min) / denom
        else:
            # Handle constant/zero attribution safely
            norm_cam = np.zeros_like(cam_np, dtype=np.float32)

        return norm_cam, pred_class_idx, confidence
