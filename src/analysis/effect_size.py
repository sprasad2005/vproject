"""Effect size and practical difference analysis utilities for RiceGuard Phase 5.

Provides standardized statistical calculations:
- Absolute and relative percentage differences
- Cohen's d for paired sample distributions
- Rank-biserial correlation for Wilcoxon signed-rank results
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


def compute_absolute_difference(val_a: float, val_b: float) -> float:
    """Compute absolute difference (val_b - val_a)."""
    return float(val_b - val_a)


def compute_relative_change_percent(val_a: float, val_b: float, eps: float = 1e-8) -> float:
    """Compute relative percentage change: ((val_b - val_a) / (|val_a| + eps)) * 100."""
    if abs(val_a) < eps:
        return 0.0
    return float(((val_b - val_a) / abs(val_a)) * 100.0)


def compute_cohens_d_paired(
    samples_a: Union[List[float], np.ndarray],
    samples_b: Union[List[float], np.ndarray],
) -> float:
    """Compute Cohen's d for paired samples: mean(diff) / std(diff).

    Args:
        samples_a: Baseline sample scores.
        samples_b: Comparison sample scores.

    Returns:
        Standardized effect size d.
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)

    if len(a) != len(b) or len(a) < 2:
        return 0.0

    diff = b - a
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff < 1e-8:
        return 0.0

    return float(mean_diff / std_diff)


def compute_rank_biserial_correlation(
    wilcoxon_stat: float,
    n_pairs: int,
) -> float:
    """Compute rank-biserial correlation r_rb from Wilcoxon signed-rank statistic.

    Formula: r_rb = 1 - (2 * W) / S, where S = n * (n + 1) / 2.
    Range: [-1.0, 1.0]. Positive means condition B ranks higher than condition A.
    """
    if n_pairs <= 0:
        return 0.0

    total_rank_sum = n_pairs * (n_pairs + 1) / 2.0
    if total_rank_sum < 1e-8:
        return 0.0

    r_rb = 1.0 - (2.0 * wilcoxon_stat) / total_rank_sum
    return float(np.clip(r_rb, -1.0, 1.0))


def interpret_cohens_d(d: float) -> str:
    """Provide standard rule-of-thumb interpretation for Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "Negligible"
    elif abs_d < 0.5:
        magnitude = "Small"
    elif abs_d < 0.8:
        magnitude = "Medium"
    else:
        magnitude = "Large"

    direction = "positive" if d >= 0 else "negative"
    return f"{magnitude} ({direction})"
