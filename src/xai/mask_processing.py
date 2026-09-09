"""Mask processing and validation utilities for Phase 4 XAI Grounding.

Provides tools for validating segmentation masks, extracting bounding regions,
and generating evaluation masks (such as border margins for healthy images).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch


def validate_binary_mask(
    mask: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> np.ndarray:
    """Validate and convert mask to a 2D binary numpy array with values in {0, 1}.

    Args:
        mask: 2D or 3D numpy array or torch Tensor.
        threshold: Threshold for binarization.

    Returns:
        2D binary numpy array of type float32.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    # Squeeze unnecessary dimensions: [1, H, W] -> [H, W] or [1, 1, H, W] -> [H, W]
    while mask_np.ndim > 2 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]

    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask_np.shape}")

    # Binarize
    binary = (mask_np > threshold).astype(np.float32)
    return binary


def get_border_margin_mask(
    height: int,
    width: int,
    border_ratio: float = 0.10,
) -> np.ndarray:
    """Generate a binary mask where border margins (outer X%) are 1.0 and center is 0.0.

    Used for measuring background shortcut attention / border attention ratio on healthy images.

    Args:
        height: Image height.
        width: Image width.
        border_ratio: Fractional margin size on each border (e.g., 0.10 for outer 10%).

    Returns:
        2D float32 numpy array of shape (height, width) with 1.0 at borders, 0.0 in interior.
    """
    mask = np.ones((height, width), dtype=np.float32)
    h_margin = int(round(height * border_ratio))
    w_margin = int(round(width * border_ratio))

    if h_margin > 0 and w_margin > 0 and (height - h_margin > h_margin) and (width - w_margin > w_margin):
        mask[h_margin : height - h_margin, w_margin : width - w_margin] = 0.0

    return mask


def extract_mask_bboxes(
    binary_mask: np.ndarray,
    min_area: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """Extract bounding boxes (x_min, y_min, x_max, y_max) from connected components in binary mask.

    Args:
        binary_mask: 2D binary numpy array.
        min_area: Minimum pixel area to consider a valid component.

    Returns:
        List of bounding boxes in format (x1, y1, x2, y2).
    """
    mask = (binary_mask > 0.5).astype(np.uint8)
    if not np.any(mask):
        return []

    # Using simple row/col slicing or standard connected components
    try:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                boxes.append((x, y, x + w, y + h))
        return boxes
    except ImportError:
        # Fallback: single bounding box of all positive pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [(int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1)]
