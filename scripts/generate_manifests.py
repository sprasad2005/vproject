"""Automated Manifest Generation Script for RiceGuard.

Generates metadata-only indexing manifests for Primary, Sethy, BD5, and RiceSeg datasets
under data/processed/manifests/ without modifying or duplicating raw image files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 75)
    print(" RiceGuard Dataset Manifest Generation Engine")
    print("=" * 75)

    from src.data.dataset_inspector import DatasetInspector
    from src.data.manifest import ManifestGenerator

    inspector = DatasetInspector(PROJECT_ROOT)
    discovered = inspector.find_dataset_folders()
    gen = ManifestGenerator(PROJECT_ROOT)

    manifest_paths = {}

    # 1. Primary Manifest
    if "primary" in discovered:
        p_dir = discovered["primary"]["matched_path"]
        print(f"[*] Processing Primary Dataset from: {p_dir.name}...")
        p_df = gen.generate_primary_manifest(p_dir)
        manifest_paths["primary"] = PROJECT_ROOT / "data" / "processed" / "manifests" / "primary_manifest.csv"
        print(f"    -> Generated primary_manifest.csv ({len(p_df):,} rows)")

    # 2. Sethy Manifest
    s_df = None
    if "sethy" in discovered:
        s_dir = discovered["sethy"]["matched_path"]
        print(f"[*] Processing Sethy Dataset from: {s_dir.name}...")
        s_df = gen.generate_sethy_manifest(s_dir)
        manifest_paths["sethy"] = PROJECT_ROOT / "data" / "processed" / "manifests" / "sethy_manifest.csv"
        print(f"    -> Generated sethy_manifest.csv ({len(s_df):,} rows)")

    # 3. BD5 Manifest
    if "bd5" in discovered:
        b_dir = discovered["bd5"]["matched_path"]
        print(f"[*] Processing BD5 Dataset from: {b_dir.name}...")
        b_df = gen.generate_bd5_manifest(b_dir)
        manifest_paths["bd5"] = PROJECT_ROOT / "data" / "processed" / "manifests" / "bd5_manifest.csv"
        print(f"    -> Generated bd5_manifest.csv ({len(b_df):,} rows)")

    # 4. RiceSeg Manifest
    if "riceseg" in discovered:
        r_dir = discovered["riceseg"]["matched_path"]
        print(f"[*] Processing RiceSeg Dataset from: {r_dir.name}...")
        r_df = gen.generate_riceseg_manifest(r_dir, s_df)
        manifest_paths["riceseg"] = PROJECT_ROOT / "data" / "processed" / "manifests" / "riceseg_manifest.csv"
        print(f"    -> Generated riceseg_manifest.csv ({len(r_df):,} rows)")

    print("\n" + "=" * 75)
    print(" All Manifests Successfully Created under data/processed/manifests/")
    for k, p in manifest_paths.items():
        print(f"   * [{k.upper()}] {p.relative_to(PROJECT_ROOT)}")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
