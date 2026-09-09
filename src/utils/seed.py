"""Reproducibility and random seed control utilities for RiceGuard."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set global random seeds across Python, NumPy, and PyTorch (CPU & CUDA).

    Args:
        seed: Random seed integer (default: 42).
        deterministic: If True, configure PyTorch CUDA / cuDNN for deterministic execution.

    Returns:
        int: The applied seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    return seed


def get_reproducibility_info(seed: Optional[int] = None) -> Dict[str, Any]:
    """Collect runtime environment, software, and hardware versions for experiment reproducibility.

    Args:
        seed: Optional random seed integer to record.

    Returns:
        Dict[str, Any]: Detailed reproducibility metadata.
    """
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    cuda_version = torch.version.cuda if (hasattr(torch.version, "cuda") and torch.version.cuda) else "N/A"
    cudnn_version = str(torch.backends.cudnn.version()) if (cuda_available and hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available()) else "N/A"

    info = {
        "seed": seed,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "operating_system": f"{platform.system()} {platform.release()} ({platform.version()})",
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
    }
    return info
