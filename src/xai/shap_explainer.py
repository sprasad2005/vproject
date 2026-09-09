"""SHAP (SHapley Additive exPlanations) model explainer for RiceGuard."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class SHAPExplainer:
    """Kernel / Partition / Gradient SHAP explainer interface."""

    def __init__(self, model: nn.Module, background_data: Any = None) -> None:
        self.model = model
        self.background_data = background_data

    def explain(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        raise NotImplementedError("SHAP explanation generation will be implemented in future phase.")
