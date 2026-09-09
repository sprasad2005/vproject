"""Device management and auto-detection utility for RiceGuard."""

from __future__ import annotations

from typing import Optional, Union

import torch


def get_device(
    use_cuda: bool = True,
    device_id: int = 0,
    verbose: bool = True,
) -> torch.device:
    """Detect and return the optimal PyTorch device (CUDA GPU or CPU).

    Args:
        use_cuda: If True, attempt to use CUDA when available (default: True).
        device_id: CUDA GPU index to select (default: 0).
        verbose: If True, print device details to stdout (default: True).

    Returns:
        torch.device: The selected PyTorch device.
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        if verbose:
            gpu_name = torch.cuda.get_device_name(device_id)
            cuda_version = torch.version.cuda or "N/A"
            print("Device: cuda")
            print(f"GPU: {gpu_name}")
            print(f"CUDA Version: {cuda_version}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Device: cpu")

    return device


def get_device_info(device: Optional[Union[torch.device, str]] = None) -> dict:
    """Retrieve detailed device metadata for logging and experiment tracking."""
    if device is None:
        device = get_device(verbose=False)
    elif isinstance(device, str):
        device = torch.device(device)

    is_cuda = device.type == "cuda"
    info = {
        "device_type": device.type,
        "is_cuda": is_cuda,
    }
    if is_cuda:
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        total_vram_mb = props.total_memory / (1024 * 1024)
        info.update(
            {
                "gpu_index": idx,
                "gpu_name": torch.cuda.get_device_name(idx),
                "cuda_version": torch.version.cuda or "N/A",
                "device_count": torch.cuda.device_count(),
                "capability": torch.cuda.get_device_capability(idx),
                "total_vram_mb": round(total_vram_mb, 2),
                "total_vram_gb": round(total_vram_mb / 1024, 2),
            }
        )
    return info
