"""Loss function definitions and builders for RiceGuard."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def build_criterion(
    config: Optional[Dict[str, Any]] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Construct loss criterion based on configuration.

    Supports standard CrossEntropyLoss, label smoothing, and optional class weighting.
    Class weights must be computed strictly from the training partition.
    """
    cfg = config or {}
    loss_cfg = cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    use_weighted = bool(loss_cfg.get("use_class_weights", False))

    weights = class_weights if (use_weighted and class_weights is not None) else None

    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
