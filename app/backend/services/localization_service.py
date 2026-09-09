"""Lesion localization decoding service for RiceGuard Backend."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from app.backend.config import settings
from app.backend.schemas import LesionBox
from src.data.grid_assignment import decode_grid_predictions


class LocalizationService:
    """Decodes raw multi-task localization logits into pixel bounding boxes."""

    def __init__(
        self,
        confidence_threshold: float = settings.confidence_threshold,
        top_k: int = settings.top_k,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

    def decode_lesions(
        self,
        loc_output: torch.Tensor,
        orig_width: int,
        orig_height: int,
    ) -> List[LesionBox]:
        """Decode localization tensor into pixel-aligned LesionBox objects.

        Args:
            loc_output: Tensor [1, 5, 7, 7] or [5, 7, 7].
            orig_width: Original image width in pixels.
            orig_height: Original image height in pixels.

        Returns:
            List of LesionBox models clipped to image bounds.
        """
        raw_boxes = decode_grid_predictions(
            loc_output=loc_output,
            conf_threshold=self.confidence_threshold,
            grid_size=7,
            top_k=self.top_k,
        )

        lesion_boxes: List[LesionBox] = []

        for b in raw_boxes:
            xc = b["x_center"]
            yc = b["y_center"]
            bw = b["width"]
            bh = b["height"]
            conf = b["confidence"]

            # Convert normalized center-width-height to top-left pixel coordinates
            x1 = int(np.clip(np.round((xc - bw / 2.0) * orig_width), 0, orig_width - 1))
            y1 = int(np.clip(np.round((yc - bh / 2.0) * orig_height), 0, orig_height - 1))
            x2 = int(np.clip(np.round((xc + bw / 2.0) * orig_width), 0, orig_width))
            y2 = int(np.clip(np.round((yc + bh / 2.0) * orig_height), 0, orig_height))

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            lesion_boxes.append(
                LesionBox(
                    x=x1,
                    y=y1,
                    width=box_w,
                    height=box_h,
                    confidence=float(conf),
                )
            )

        return lesion_boxes
