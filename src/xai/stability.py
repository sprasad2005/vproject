"""Explanation stability metrics under image corruptions and perturbations."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_explanation_stability(
    clean_explanation: np.ndarray,
    perturbed_explanation: np.ndarray,
    eps: float = 1e-10,
) -> Dict[str, float]:
    """Quantify attribution stability between clean and corrupted image inputs."""
    flat_clean = clean_explanation.flatten()
    flat_pert = perturbed_explanation.flatten()

    std_clean = np.std(flat_clean)
    std_pert = np.std(flat_pert)

    if std_clean > eps and std_pert > eps:
        pearson_corr = float(np.corrcoef(flat_clean, flat_pert)[0, 1])
    else:
        pearson_corr = 1.0 if np.allclose(flat_clean, flat_pert) else 0.0

    l1_diff = float(np.mean(np.abs(clean_explanation - perturbed_explanation)))

    return {
        "pearson_correlation": pearson_corr,
        "l1_difference": l1_diff,
    }
