"""Environment verification script for RiceGuard.

Checks Python version, PyTorch, Torchvision, NumPy, Pandas, OpenCV, Scikit-learn,
and CUDA/GPU availability, displaying PASS, FAIL, or WARNING status for each item.
"""

from __future__ import annotations

import sys


def print_status(item: str, status: str, details: str = ""):
    """Print formatted check status."""
    badge = f"[{status}]"
    print(f"{badge:<10} {item:<25} : {details}")


def main() -> int:
    print("=" * 70)
    print(" RiceGuard Environment Verification")
    print("=" * 70)
    failures = 0
    warnings = 0

    # 1. Python version check (>= 3.10)
    py_version = sys.version_info
    py_ver_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    if py_version.major == 3 and py_version.minor >= 10:
        print_status("Python Version", "PASS", f"{py_ver_str} (>= 3.10 requirement satisfied)")
    else:
        print_status("Python Version", "FAIL", f"{py_ver_str} (< 3.10 required)")
        failures += 1

    # 2. PyTorch import
    try:
        import torch

        torch_ver = torch.__version__
        print_status("PyTorch Import", "PASS", f"v{torch_ver}")
    except ImportError as e:
        print_status("PyTorch Import", "FAIL", str(e))
        failures += 1
        torch = None

    # 3. Torchvision import
    try:
        import torchvision

        print_status("Torchvision Import", "PASS", f"v{torchvision.__version__}")
    except ImportError as e:
        print_status("Torchvision Import", "FAIL", str(e))
        failures += 1

    # 4. NumPy import
    try:
        import numpy

        print_status("NumPy Import", "PASS", f"v{numpy.__version__}")
    except ImportError as e:
        print_status("NumPy Import", "FAIL", str(e))
        failures += 1

    # 5. Pandas import
    try:
        import pandas

        print_status("Pandas Import", "PASS", f"v{pandas.__version__}")
    except ImportError as e:
        print_status("Pandas Import", "FAIL", str(e))
        failures += 1

    # 6. OpenCV import
    try:
        import cv2

        print_status("OpenCV Import", "PASS", f"v{cv2.__version__}")
    except ImportError as e:
        print_status("OpenCV Import", "FAIL", str(e))
        failures += 1

    # 7. Scikit-learn import
    try:
        import sklearn

        print_status("Scikit-learn Import", "PASS", f"v{sklearn.__version__}")
    except ImportError as e:
        print_status("Scikit-learn Import", "FAIL", str(e))
        failures += 1

    # 8. PyYAML import
    try:
        import yaml

        print_status("PyYAML Import", "PASS", f"v{yaml.__version__}")
    except ImportError as e:
        print_status("PyYAML Import", "FAIL", str(e))
        failures += 1

    # 9. tqdm import
    try:
        import tqdm

        print_status("tqdm Import", "PASS", f"v{tqdm.__version__}")
    except ImportError as e:
        print_status("tqdm Import", "FAIL", str(e))
        failures += 1

    # 10. CUDA and GPU Detection
    if torch is not None:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "N/A"
            gpu_count = torch.cuda.device_count()
            print_status("CUDA Availability", "PASS", f"Available ({gpu_count} GPU(s) detected)")
            print_status("GPU Device Info", "PASS", f"{gpu_name} (CUDA {cuda_ver})")
        else:
            print_status(
                "CUDA Availability", "WARNING", "No CUDA GPU detected; CPU execution enabled."
            )
            print_status("GPU Device Info", "WARNING", "N/A (Running on CPU)")
            warnings += 1
    else:
        print_status("CUDA Availability", "FAIL", "PyTorch not available to test CUDA")
        failures += 1

    print("=" * 70)
    if failures == 0:
        print(f" Environment Verification COMPLETE: All required packages PASSED ({warnings} warnings).")
        return 0
    else:
        print(f" Environment Verification FAILED: {failures} critical failure(s).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
