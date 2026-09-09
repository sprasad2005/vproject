"""Monte Carlo Dropout epistemic uncertainty estimator for RiceGuard."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def estimate_mc_dropout_uncertainty(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = 30,
) -> Dict[str, Any]:
    """Perform stochastic forward passes with dropout enabled at test-time."""
    raise NotImplementedError("MC Dropout estimation will be implemented in future phase.")
