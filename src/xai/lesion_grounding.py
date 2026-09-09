"""Quantitative Lesion Grounding evaluation: IoU, Dice, Explanation Precision and Recall."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_lesion_grounding_metrics(
    explanation_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-10,
) -> Dict[str, float]:
    """Calculate overlap metrics between XAI explanation heatmaps and expert lesion segmentation masks.

    Args:
        explanation_map: Normalized saliency/attribution map in [0, 1].
        ground_truth_mask: Binary lesion mask {0, 1}.
        threshold: Binarization cutoff for explanation map.
        eps: Epsilon to prevent zero-division.

    Returns:
        Dict[str, float]: IoU, Dice, Precision, Recall scores.
    """
    binary_expl = (explanation_map >= threshold).astype(np.float32)
    binary_gt = (ground_truth_mask >= 0.5).astype(np.float32)

    intersection = np.sum(binary_expl * binary_gt)
    union = np.sum(binary_expl) + np.sum(binary_gt) - intersection

    iou = float((intersection + eps) / (union + eps))
    dice = float((2.0 * intersection + eps) / (np.sum(binary_expl) + np.sum(binary_gt) + eps))

    precision = float((intersection + eps) / (np.sum(binary_expl) + eps))
    recall = float((intersection + eps) / (np.sum(binary_gt) + eps))

    return {
        "iou": iou,
        "dice": dice,
        "explanation_precision": precision,
        "explanation_recall": recall,
    }
