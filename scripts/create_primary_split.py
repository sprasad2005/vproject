"""Primary Dataset Partitioning Script for RiceGuard.

Generates stratified 70/15/15 splits (train, validation, internal test) with seed=42
and strict exact-duplicate leakage prevention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 75)
    print(" RiceGuard Primary Dataset Splitting Engine")
    print("=" * 75)

    from src.data.splitting import PrimaryDatasetSplitter

    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifests" / "primary_manifest.csv"
    if not manifest_path.exists():
        print(f"[!] Primary manifest not found at {manifest_path}. Run generate_manifests.py first.")
        from src.data.dataset_inspector import DatasetInspector
        from src.data.manifest import ManifestGenerator
        inspector = DatasetInspector(PROJECT_ROOT)
        discovered = inspector.find_dataset_folders()
        if "primary" in discovered:
            gen = ManifestGenerator(PROJECT_ROOT)
            gen.generate_primary_manifest(discovered["primary"]["matched_path"])

    df = pd.read_csv(manifest_path)
    print(f"[*] Loaded primary manifest: {len(df):,} records")

    splitter = PrimaryDatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, root_dir=PROJECT_ROOT)
    train_df, val_df, test_df, meta = splitter.split(df)

    print("\n--- Partition Summary ---")
    print(f"  * Train set      : {len(train_df):,} images ({len(train_df)/len(df):.1%})")
    print(f"  * Validation set : {len(val_df):,} images ({len(val_df)/len(df):.1%})")
    print(f"  * Internal Test  : {len(test_df):,} images ({len(test_df)/len(df):.1%})")
    print(f"  * Total          : {len(df):,} images")

    print("\n--- Stratified Class Breakdown (Train / Val / Test) ---")
    classes = sorted(list(meta["class_distributions"]["train"].keys()))
    for c in classes:
        tr = meta["class_distributions"]["train"].get(c, 0)
        va = meta["class_distributions"]["val"].get(c, 0)
        te = meta["class_distributions"]["test"].get(c, 0)
        print(f"  * {c:<15}: {tr:>5} train | {va:>4} val | {te:>4} test")

    leak_passed = meta["leakage_verification"]["exact_duplicate_leakage_passed"]
    print(f"\n[LEAKAGE AUDIT] Exact Duplicate Leakage Status: {'PASSED (0 OVERLAPS)' if leak_passed else 'FAILED'}")

    print("\n" + "=" * 75)
    print(f" Split files written to {PROJECT_ROOT / 'splits'}:")
    print("   * splits/primary_train.csv")
    print("   * splits/primary_val.csv")
    print("   * splits/primary_test.csv")
    print("   * splits/primary_split_metadata.json")
    print("   * dataset_reports/split_leakage_report.md")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
