"""Predictive entropy and mutual information quantification for RiceGuard."""

from __future__ import annotations

import numpy as np


def compute_predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Shannon predictive entropy for categorical probability distribution."""
    clipped = np.clip(probabilities, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1)
