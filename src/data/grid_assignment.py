"""Grid Target Assignment and Collision Management for RiceGuard Lesion Localization.

Converts normalized YOLO bounding boxes [x_center, y_center, width, height]
into a fixed-size spatial grid target tensor [5, S, S] representing:
  - Channel 0: Objectness binary indicator (1 = lesion present, 0 = background/healthy)
  - Channel 1: Normalized horizontal offset within grid cell dx in [0, 1]
  - Channel 2: Normalized vertical offset within grid cell dy in [0, 1]
  - Channel 3: Normalized bounding box width w in [0, 1]
  - Channel 4: Normalized bounding box height h in [0, 1]

Features:
- Deterministic grid assignment
- Multi-instance collision resolution (keeps largest-area lesion per cell)
- Zero objectness and zero box targets for Healthy images
- Dataset-wide pos_weight calculation for BCEWithLogitsLoss imbalance mitigation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from src.data.annotations import BoundingBox, parse_yolo_bbox_file


def create_grid_target(
    boxes: List[BoundingBox],
    grid_size: int = 7,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Convert a list of bounding boxes for a single image into a grid target tensor.

    Args:
        boxes: List of BoundingBox objects for the image (empty for Healthy images).
        grid_size: Spatial grid dimensions S x S (default: 7).

    Returns:
        target_tensor: FloatTensor of shape [5, grid_size, grid_size].
        stats: Dictionary containing box count, assigned count, and collision stats.
    """
    target = np.zeros((5, grid_size, grid_size), dtype=np.float32)
    # Track cell assignment area for deterministic collision resolution
    cell_areas = np.zeros((grid_size, grid_size), dtype=np.float32)

    total_boxes = len(boxes)
    assigned_boxes = 0
    collisions = 0

    for box in boxes:
        if not box.is_valid:
            continue

        # Determine grid cell coordinates (row i, col j)
        col = int(np.clip(np.floor(box.x_center * grid_size), 0, grid_size - 1))
        row = int(np.clip(np.floor(box.y_center * grid_size), 0, grid_size - 1))

        # Normalized cell offsets dx, dy in [0, 1]
        dx = float(box.x_center * grid_size - col)
        dy = float(box.y_center * grid_size - row)
        dx = float(np.clip(dx, 0.0, 1.0))
        dy = float(np.clip(dy, 0.0, 1.0))

        w = float(np.clip(box.width, 0.0, 1.0))
        h = float(np.clip(box.height, 0.0, 1.0))
        box_area = w * h

        # Collision detection: if cell already has an assigned box
        if target[0, row, col] == 1.0:
            collisions += 1
            # Deterministic collision strategy: retain the box with the larger area
            if box_area > cell_areas[row, col]:
                target[0, row, col] = 1.0
                target[1, row, col] = dx
                target[2, row, col] = dy
                target[3, row, col] = w
                target[4, row, col] = h
                cell_areas[row, col] = box_area
        else:
            target[0, row, col] = 1.0
            target[1, row, col] = dx
            target[2, row, col] = dy
            target[3, row, col] = w
            target[4, row, col] = h
            cell_areas[row, col] = box_area
            assigned_boxes += 1

    stats = {
        "total_boxes": total_boxes,
        "assigned_boxes": assigned_boxes,
        "collisions": collisions,
        "has_collision": collisions > 0,
    }

    return torch.from_numpy(target), stats


def decode_grid_predictions(
    loc_output: torch.Tensor,
    conf_threshold: float = 0.5,
    grid_size: int = 7,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Decode raw localization branch output [5, S, S] into predicted bounding boxes.

    Applies sigmoid activation to objectness logits and box offsets/dimensions:
      - objectness = sigmoid(raw_obj)
      - dx, dy = sigmoid(raw_dx), sigmoid(raw_dy)
      - w, h = sigmoid(raw_w), sigmoid(raw_h)
      - x_center = (col + dx) / S
      - y_center = (row + dy) / S

    Args:
        loc_output: Tensor of shape [5, grid_size, grid_size] (or [B, 5, S, S]).
        conf_threshold: Objectness threshold for considering a cell as positive.
        grid_size: Spatial grid size S (default: 7).
        top_k: Optional maximum number of highest-confidence boxes to retain.

    Returns:
        List of decoded box dictionaries: [{'x_center', 'y_center', 'width', 'height', 'confidence', 'grid_row', 'grid_col'}]
    """
    if loc_output.dim() == 4:
        loc_output = loc_output.squeeze(0)

    raw_obj = loc_output[0].cpu()
    raw_dx = loc_output[1].cpu()
    raw_dy = loc_output[2].cpu()
    raw_w = loc_output[3].cpu()
    raw_h = loc_output[4].cpu()

    obj_scores = torch.sigmoid(raw_obj).numpy()
    dx = torch.sigmoid(raw_dx).numpy()
    dy = torch.sigmoid(raw_dy).numpy()
    w = torch.sigmoid(raw_w).numpy()
    h = torch.sigmoid(raw_h).numpy()

    predicted_boxes: List[Dict[str, Any]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            conf = float(obj_scores[row, col])
            if conf >= conf_threshold:
                xc = float((col + dx[row, col]) / grid_size)
                yc = float((row + dy[row, col]) / grid_size)
                box_w = float(w[row, col])
                box_h = float(h[row, col])

                predicted_boxes.append(
                    {
                        "x_center": float(np.clip(xc, 0.0, 1.0)),
                        "y_center": float(np.clip(yc, 0.0, 1.0)),
                        "width": float(np.clip(box_w, 0.0, 1.0)),
                        "height": float(np.clip(box_h, 0.0, 1.0)),
                        "confidence": conf,
                        "grid_row": row,
                        "grid_col": col,
                    }
                )

    # Sort descending by confidence
    predicted_boxes.sort(key=lambda x: x["confidence"], reverse=True)

    if top_k is not None and top_k > 0 and len(predicted_boxes) > top_k:
        predicted_boxes = predicted_boxes[:top_k]

    return predicted_boxes


def calculate_train_split_pos_weight(
    train_manifest_csv: str | Path,
    root_dir: Optional[Path] = None,
    grid_size: int = 7,
) -> Dict[str, Any]:
    """Calculate objectness pos_weight strictly from primary_train.csv.

    Formula:
        pos_weight = total_negative_cells / total_positive_cells
    """
    df = pd.read_csv(train_manifest_csv)
    total_cells = len(df) * (grid_size * grid_size)
    total_positive_cells = 0
    total_collisions = 0
    max_lesions = 0

    for _, row in df.iterrows():
        has_annot = bool(row.get("has_annotation", False))
        annot_path = row.get("annotation_path")

        boxes: List[BoundingBox] = []
        if has_annot and annot_path and pd.notna(annot_path):
            full_p = Path(annot_path)
            if root_dir and not full_p.is_absolute():
                full_p = root_dir / full_p
            if full_p.exists():
                boxes, _ = parse_yolo_bbox_file(full_p)

        max_lesions = max(max_lesions, len(boxes))
        _, stats = create_grid_target(boxes, grid_size=grid_size)
        total_positive_cells += stats["assigned_boxes"]
        total_collisions += stats["collisions"]

    total_negative_cells = total_cells - total_positive_cells
    pos_weight = total_negative_cells / max(total_positive_cells, 1)

    return {
        "grid_size": grid_size,
        "total_images": len(df),
        "total_cells": total_cells,
        "total_positive_cells": total_positive_cells,
        "total_negative_cells": total_negative_cells,
        "total_collisions": total_collisions,
        "collision_rate": total_collisions / max(total_positive_cells + total_collisions, 1),
        "max_lesions_per_image": max_lesions,
        "pos_weight": float(round(pos_weight, 4)),
    }
