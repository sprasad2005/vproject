"""Optimizer builder and configuration for RiceGuard."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Build PyTorch optimizer (AdamW, Adam, SGD) from configuration."""
    opt_name = config.get("optimizer", {}).get("name", "adamw").lower()
    lr = float(config.get("optimizer", {}).get("lr", 1e-4))
    weight_decay = float(config.get("optimizer", {}).get("weight_decay", 0.01))

    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = float(config.get("optimizer", {}).get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
