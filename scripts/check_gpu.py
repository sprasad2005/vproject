"""GPU detection and diagnostic script for RiceGuard.

Safely probes CUDA devices, reporting GPU name, count, and versions,
while gracefully handling CPU-only environments without crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def main():
    print("=" * 60)
    print(" RiceGuard GPU Diagnostic")
    print("=" * 60)

    try:
        import torch

        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        print("CUDA Available : False (PyTorch is not installed)")
        print("Selected Device: cpu")
        print("GPU Name       : N/A")
        print("GPU Count      : 0")
        print("CUDA Version   : N/A")
        print("PyTorch Version: N/A")
        print("=" * 60)
        return

    # Auto-detect device via RiceGuard utility
    from src.utils.device import get_device

    device = get_device(verbose=False)

    print(f"CUDA Available : {cuda_avail}")
    print(f"Selected Device: {device}")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        cuda_ver = torch.version.cuda or "N/A"
        print(f"GPU Name       : {gpu_name}")
        print(f"GPU Count      : {gpu_count}")
        print(f"CUDA Version   : {cuda_ver}")
    else:
        print("GPU Name       : N/A")
        print("GPU Count      : 0")
        print("CUDA Version   : N/A")

    print(f"PyTorch Version: {torch_ver}")
    print("=" * 60)


if __name__ == "__main__":
    main()
