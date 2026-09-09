"""Quantitative XAI Grounding Metrics for RiceGuard Phase 4.

Computes:
- Energy Inside Mask (Primary Metric)
- Energy Outside Mask (Complementarity Verification)
- Attribution IoU (Top-K% Saliency Overlap)
- Pointing Game Accuracy (Hit Rate of Saliency Peak)
- Healthy Image Metrics (Concentration, Border Shortcut Ratio, Normalized Entropy)
- Statistical Aggregators (Mean, Median, Std, 95% CI)
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import scipy.stats as stats

from src.xai.mask_processing import get_border_margin_mask, validate_binary_mask


def compute_energy_inside_mask(
    attribution_map: np.ndarray,
    binary_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Energy Inside Mask: proportion of attribution located within the lesion mask.

    Formula: sum(attribution * mask) / sum(attribution)
    """
    attr = np.clip(np.asarray(attribution_map, dtype=np.float32), 0.0, None)
    mask = validate_binary_mask(binary_mask)

    total_energy = float(np.sum(attr))
    if total_energy < eps:
        return 0.0

    inside_energy = float(np.sum(attr * mask))
    return float(np.clip(inside_energy / (total_energy + eps), 0.0, 1.0))


def compute_energy_outside_mask(
    attribution_map: np.ndarray,
    binary_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Energy Outside Mask: proportion of attribution located outside the lesion mask.

    Formula: sum(attribution * (1 - mask)) / sum(attribution)
    """
    attr = np.clip(np.asarray(attribution_map, dtype=np.float32), 0.0, None)
    mask = validate_binary_mask(binary_mask)

    total_energy = float(np.sum(attr))
    if total_energy < eps:
        return 0.0

    outside_energy = float(np.sum(attr * (1.0 - mask)))
    return float(np.clip(outside_energy / (total_energy + eps), 0.0, 1.0))


def compute_attribution_iou(
    attribution_map: np.ndarray,
    binary_mask: np.ndarray,
    top_percent: float = 20.0,
    eps: float = 1e-8,
) -> float:
    """Compute Attribution IoU by thresholding attribution at the top-K percentile.

    Protocol:
        1. Find the (100 - top_percent) percentile value of attribution.
        2. Binarize attribution map: A_bin = (attribution >= threshold).
        3. Compute IoU with binary ground-truth lesion mask.

    Args:
        attribution_map: 2D float array in [0, 1].
        binary_mask: 2D binary array in {0, 1}.
        top_percent: Percentage of top attribution pixels to consider (default: 20.0).
        eps: Small constant to avoid division by zero.

    Returns:
        Intersection over Union (IoU) in [0, 1].
    """
    attr = np.asarray(attribution_map, dtype=np.float32)
    mask = validate_binary_mask(binary_mask)

    if float(np.sum(mask)) < 1.0:
        # If mask is empty, IoU is 0 (or 1 if attribution also empty)
        return 0.0

    threshold_val = float(np.percentile(attr, 100.0 - top_percent))
    attr_bin = (attr >= threshold_val).astype(np.float32)

    intersection = float(np.sum(attr_bin * mask))
    union = float(np.sum(np.clip(attr_bin + mask, 0.0, 1.0)))

    if union < eps:
        return 0.0

    return float(np.clip(intersection / (union + eps), 0.0, 1.0))


def compute_pointing_game(
    attribution_map: np.ndarray,
    binary_mask: np.ndarray,
) -> int:
    """Compute Pointing Game hit: 1 if argmax pixel falls within the lesion mask, 0 otherwise.

    Args:
        attribution_map: 2D float array.
        binary_mask: 2D binary array in {0, 1}.

    Returns:
        1 for a hit (inside lesion), 0 for a miss.
    """
    attr = np.asarray(attribution_map, dtype=np.float32)
    mask = validate_binary_mask(binary_mask)

    # If mask has no lesions, pointing game is miss (0)
    if float(np.sum(mask)) < 1.0:
        return 0

    max_idx = int(np.argmax(attr))
    r, c = np.unravel_index(max_idx, attr.shape)

    return 1 if mask[r, c] > 0.5 else 0


def compute_healthy_xai_metrics(
    attribution_map: np.ndarray,
    border_ratio: float = 0.10,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute explainability metrics specifically tailored for Healthy leaves (no lesions).

    Metrics:
    - attribution_concentration: Energy in the top 10% highest attribution pixels.
    - border_attention_ratio: Energy located in the outer border margin (detects background shortcuts).
    - normalized_entropy: Shannon entropy normalized by log(total_pixels).
    """
    attr = np.clip(np.asarray(attribution_map, dtype=np.float32), 0.0, None)
    h, w = attr.shape
    total_pixels = h * w
    total_energy = float(np.sum(attr))

    if total_energy < eps:
        return {
            "attribution_concentration": 0.0,
            "border_attention_ratio": 0.0,
            "normalized_entropy": 0.0,
        }

    # 1. Attribution concentration: top 10% energy
    thresh_90 = float(np.percentile(attr, 90.0))
    top_10_energy = float(np.sum(attr[attr >= thresh_90]))
    concentration = float(np.clip(top_10_energy / (total_energy + eps), 0.0, 1.0))

    # 2. Border Attention Ratio
    border_mask = get_border_margin_mask(h, w, border_ratio=border_ratio)
    border_energy = float(np.sum(attr * border_mask))
    border_ratio_val = float(np.clip(border_energy / (total_energy + eps), 0.0, 1.0))

    # 3. Normalized Entropy
    p = (attr.flatten() + eps) / (total_energy + total_pixels * eps)
    entropy_val = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(total_pixels))
    norm_entropy = float(np.clip(entropy_val / max_entropy, 0.0, 1.0))

    return {
        "attribution_concentration": concentration,
        "border_attention_ratio": border_ratio_val,
        "normalized_entropy": norm_entropy,
    }


def compute_distribution_summary(values: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Compute mean, median, standard deviation, and 95% confidence interval for a list of values."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "count": 0}

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n > 1 and std_val > 1e-8:
        # Student's t-distribution confidence interval
        se = std_val / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=se)
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    else:
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "count": n,
    }
