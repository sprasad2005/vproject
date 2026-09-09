"""Learning rate scheduler builder for RiceGuard."""

from __future__ import annotations

from typing import Any, Dict

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau, StepLR


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> LRScheduler:
    """Build learning rate scheduler from configuration."""
    sched_name = config.get("scheduler", {}).get("name", "cosine_annealing").lower()

    if sched_name == "cosine_annealing":
        t_max = int(config.get("scheduler", {}).get("t_max", 50))
        eta_min = float(config.get("scheduler", {}).get("eta_min", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif sched_name == "step":
        step_size = int(config.get("scheduler", {}).get("step_size", 10))
        gamma = float(config.get("scheduler", {}).get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "plateau":
        patience = int(config.get("scheduler", {}).get("patience", 5))
        factor = float(config.get("scheduler", {}).get("factor", 0.5))
        return ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")
