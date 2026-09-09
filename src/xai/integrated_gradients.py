"""Integrated Gradients axiomatic attribution explainer for RiceGuard."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class IntegratedGradientsExplainer:
    """Computes Integrated Gradients attribution maps along linear interpolation paths."""

    def __init__(self, model: nn.Module, steps: int = 50) -> None:
        self.model = model
        self.steps = steps

    def attribute(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        raise NotImplementedError("Integrated Gradients attribution will be implemented in future phase.")

