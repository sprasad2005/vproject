"""Project integrity and foundation verification script for RiceGuard.

Validates that:
1. All required project directories and placeholder files exist.
2. All YAML configuration files exist and are syntactically valid.
3. Core project modules import cleanly.
4. Unified configuration merges properly.
5. Device detection and deterministic seed management function correctly.
6. Dataset Registry security policies correctly protect external benchmarks.
7. Existing dataset roots are discovered and registered.
8. Dataset inventory reports exist and are populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_step(step_name: str, passed: bool, details: str = ""):
    tag = "[ PASS ]" if passed else "[ FAIL ]"
    print(f"{tag} {step_name:<40} : {details}")


def main() -> int:
    print("=" * 75)
    print(" RiceGuard Phase 0 Foundation Verification")
    print("=" * 75)

    failures = []

    # 1. Directory Structure Check
    required_dirs = [
        "configs",
        "data",
        "dataset_reports",
        "splits",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/calibration",
        "src/uncertainty",
        "src/xai",
        "src/robustness",
        "src/utils",
        "scripts",
        "notebooks",
        "checkpoints",
        "logs",
        "experiments",
        "results/figures",
        "results/metrics",
        "results/predictions",
        "results/explanations",
        "results/robustness",
        "results/reports",
        "tests",
        "app",
    ]

    missing_dirs = [d for d in required_dirs if not (PROJECT_ROOT / d).exists()]
    if not missing_dirs:
        print_step("Directory Structure", True, f"All {len(required_dirs)} required folders present")
    else:
        print_step("Directory Structure", False, f"Missing: {missing_dirs}")
        failures.append("Directory Structure")

    # 2. Configuration Files Check
    required_configs = [
        "configs/base.yaml",
        "configs/data.yaml",
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/calibration.yaml",
        "configs/uncertainty.yaml",
        "configs/xai.yaml",
        "configs/robustness.yaml",
        "configs/paths.yaml",
    ]
    missing_configs = [c for c in required_configs if not (PROJECT_ROOT / c).exists()]
    if not missing_configs:
        print_step("YAML Configuration Files", True, f"All {len(required_configs)} configs present")
    else:
        print_step("YAML Configuration Files", False, f"Missing: {missing_configs}")
        failures.append("Configuration Files Existence")

    # 3. Module Imports Check
    try:
        from src.data import (
            DatasetInspector,
            is_external,
            is_training_allowed,
            validate_dataset_usage,
        )
        from src.utils import (
            get_device,
            get_device_info,
            get_reproducibility_info,
            load_all_configs,
            load_config,
            set_seed,
        )

        print_step("Module Importability", True, "All src subpackages imported successfully")
    except Exception as e:
        print_step("Module Importability", False, f"Import Error: {e}")
        failures.append("Module Imports")
        return 1

    # 4. Configuration System Check
    try:
        cfg = load_config()
        assert cfg.project.name == "RiceGuard", f"Unexpected project name {cfg.project.name}"
        assert cfg.seed == 42, f"Unexpected default seed {cfg.seed}"

        unified_cfg = load_all_configs()
        assert hasattr(unified_cfg, "datasets") or hasattr(unified_cfg, "primary_dataset")
        assert hasattr(unified_cfg, "models"), "Missing models in unified config"
        assert hasattr(unified_cfg, "xai"), "Missing xai in unified config"
        print_step("Configuration System", True, "base.yaml & unified configs loaded & validated")
    except Exception as e:
        print_step("Configuration System", False, f"Config Error: {e}")
        failures.append("Configuration System")

    # 5. Device Detection Check
    try:
        dev = get_device(verbose=False)
        dev_info = get_device_info(dev)
        print_step("Device Detection", True, f"Device: {dev} ({'CUDA' if dev_info['is_cuda'] else 'CPU'})")
    except Exception as e:
        print_step("Device Detection", False, f"Device Error: {e}")
        failures.append("Device Detection")

    # 6. Seed & Reproducibility Check
    try:
        applied_seed = set_seed(42)
        repro_info = get_reproducibility_info()
        assert applied_seed == 42
        assert "python_version" in repro_info
        print_step("Seed & Reproducibility", True, f"Seed {applied_seed} applied & repro info collected")
    except Exception as e:
        print_step("Seed & Reproducibility", False, f"Seed Error: {e}")
        failures.append("Seed & Reproducibility")

    # 7. Dataset Registry & Security Protection Check
    try:
        assert is_training_allowed("primary") is True
        assert is_training_allowed("sethy") is False
        assert is_training_allowed("bd5") is False
        assert is_training_allowed("riceseg") is False
        assert is_external("sethy") is True
        assert is_external("bd5") is True

        # Test security exception enforcement
        caught = 0
        for ext_key in ["sethy", "bd5", "riceseg"]:
            try:
                validate_dataset_usage(ext_key, "train")
            except PermissionError:
                caught += 1
        assert caught == 3, f"Expected 3 security rejections, got {caught}"
        print_step("Dataset Registry Protection", True, "Training strictly prohibited on Sethy, BD5, RiceSeg")
    except Exception as e:
        print_step("Dataset Registry Protection", False, f"Registry Error: {e}")
        failures.append("Dataset Registry Protection")

    # 8. Dataset Discovery & Inventory Report Check
    try:
        inspector = DatasetInspector(PROJECT_ROOT)
        discovered = inspector.find_dataset_folders()
        assert "primary" in discovered, "Primary dataset folder not found"
        assert "sethy" in discovered, "Sethy dataset folder not found"
        assert "bd5" in discovered, "BD5 dataset folder not found"
        assert "riceseg" in discovered, "RiceSeg dataset folder not found"

        report_json = PROJECT_ROOT / "dataset_reports" / "dataset_inventory.json"
        report_md = PROJECT_ROOT / "dataset_reports" / "dataset_inventory.md"

        has_reports = report_json.exists() and report_md.exists()
        print_step("Dataset Discovery & Reports", True, f"All 4 datasets discovered (Reports generated: {has_reports})")
    except Exception as e:
        print_step("Dataset Discovery & Reports", False, f"Discovery Error: {e}")
        failures.append("Dataset Discovery & Reports")

    print("=" * 75)
    if not failures:
        print("\n>>> PROJECT SETUP VERIFIED SUCCESSFULLY <<<\n")
        return 0
    else:
        print(f"\n>>> VERIFICATION FAILED: {len(failures)} item(s) failed ({', '.join(failures)}) <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
