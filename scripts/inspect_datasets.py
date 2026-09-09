"""Automated Dataset Discovery, Inspection, and Inventory Reporting Script for RiceGuard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def generate_markdown_report(report: dict, output_path: Path) -> None:
    """Render structured inspection dictionary into a clean Markdown table report."""
    md_lines = [
        "# RiceGuard Dataset Inventory & Integrity Audit Report",
        "",
        f"**Project**: {report['project_name']}  ",
        f"**Phase**: {report['phase']}  ",
        "",
        "---",
        "",
        "## 1. Discovered Dataset Summary",
        "",
        "| Dataset Key | Dataset Name | Registered Role | Discovered Location | Training Allowed | Total Images / Masks | Status |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
    ]

    for key, d in report["datasets"].items():
        total_items = d.get("total_images", d.get("total_masks", "N/A"))
        train_allowed = "YES" if d.get("training_allowed") else "**NO**"
        md_lines.append(
            f"| `{key}` | **{d['name']}** | `{d['role']}` | `{d['detected_root']}` | {train_allowed} | {total_items:,} | {d['status']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Dataset Specific Audits",
        "",
    ])

    # Primary Dataset Details
    if "primary" in report["datasets"]:
        p = report["datasets"]["primary"]
        md_lines.extend([
            f"### A. Primary Benchmark: `{p['name']}`",
            f"* **Discovered Path**: `{p['detected_root']}`",
            f"* **Total Original Images**: {p['total_images']:,}",
            f"* **Image Formats**: {', '.join(p['image_formats'])}",
            f"* **Bounding Box Annotations**: Available ({p['bounding_boxes']['total_annotation_files']:,} `{p['bounding_boxes']['format']}` files)",
            "* **Class Distribution (Original Images)**:",
        ])
        for c, count in p["classes"].items():
            md_lines.append(f"  * **{c}**: {count:,} images")
        md_lines.append("")

    # Sethy Dataset Details
    if "sethy" in report["datasets"]:
        s = report["datasets"]["sethy"]
        md_lines.extend([
            f"### B. External Benchmark 1: `{s['name']}`",
            f"* **Discovered Path**: `{s['detected_root']}`",
            f"* **Total Images**: {s['total_images']:,}",
            f"* **Image Formats**: {', '.join(s['image_formats'])}",
            f"* **Training Permitted**: `{s['training_allowed']}` *(Strictly External Test Only)*",
            "* **Class Distribution**:",
        ])
        for c, count in s["classes"].items():
            md_lines.append(f"  * **{c}**: {count:,} images")
        md_lines.append("")

    # BD5 Dataset Details
    if "bd5" in report["datasets"]:
        b = report["datasets"]["bd5"]
        md_lines.extend([
            f"### C. External Field Benchmark 2: `{b['name']}`",
            f"* **Discovered Path**: `{b['detected_root']}`",
            f"* **Total Field Images**: {b['total_images']:,}",
            f"* **Image Formats**: {', '.join(b['image_formats'])}",
            f"* **Training Permitted**: `{b['training_allowed']}` *(Strictly Field Test Only)*",
            "* **Class Distribution**:",
        ])
        for c, count in b["classes"].items():
            md_lines.append(f"  * **{c}**: {count:,} images")
        md_lines.append("")

    # RiceSeg Dataset Details
    if "riceseg" in report["datasets"]:
        r = report["datasets"]["riceseg"]
        pv = r["pairing_verification"]
        md_lines.extend([
            f"### D. XAI Ground Truth: `{r['name']}`",
            f"* **Discovered Path**: `{r['detected_root']}`",
            f"* **Total Pixel-Level Masks**: {r['total_masks']:,}",
            f"* **Mask Formats**: {', '.join(r['mask_formats'])}",
            "* **Mask Class Breakdown**:",
        ])
        for c, count in r["classes"].items():
            md_lines.append(f"  * **{c}**: {count:,} masks")
        md_lines.extend([
            "* **Image-Mask Pairing Verification**:",
            f"  * Candidate External Images: {pv['total_candidate_images']:,}",
            f"  * Matched Image-Mask Pairs: **{pv['matched_image_mask_pairs']:,}**",
            f"  * Unmatched Images: {pv['unmatched_images']}",
            f"  * Unmatched Masks: {pv['unmatched_masks']}",
            f"  * Pairing Feasible: **{pv['pairing_possible']}**",
            "",
        ])

    md_lines.extend([
        "---",
        "",
        "## 3. Security & Access Policy Enforcement",
        "",
        "| Dataset | Permitted Uses | Forbidden Uses |",
        "| :--- | :--- | :--- |",
        "| `RiceLeafDiseaseBD` | Training, Validation, Internal Test, Bounding Box Supervision | Cross-contamination |",
        "| `Sethy5932` | Final External Generalization Test, Domain Shift Analysis | **Training, Validation, Threshold Tuning** |",
        "| `RiceLeafDiseaseBD5` | Final External Field Robustness Test, Domain Shift Analysis | **Training, Validation, Class Merging** |",
        "| `RiceSeg5932` | Post-hoc XAI Explanation Mask IoU/Dice Validation | **Classifier Training, Feature Extraction** |",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main() -> int:
    print("=" * 75)
    print(" RiceGuard Automated Dataset Discovery & Inspection Engine")
    print("=" * 75)

    from src.data.dataset_inspector import DatasetInspector
    from src.utils.paths import ensure_dir, get_path

    inspector = DatasetInspector(PROJECT_ROOT)
    report = inspector.run_full_inspection()

    # Save outputs to dataset_reports/
    reports_dir = ensure_dir(get_path("dataset_reports"))
    json_path = reports_dir / "dataset_inventory.json"
    md_path = reports_dir / "dataset_inventory.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    generate_markdown_report(report, md_path)

    # Print summary to console
    print("\n--- Discovered Datasets & Inventory ---")
    for key, d in report["datasets"].items():
        total = d.get("total_images", d.get("total_masks", "N/A"))
        print(f"[{key.upper():<8}] {d['name']:<22} | Role: {d['role']:<20} | Items: {total:>6} | Path: {d['detected_root']}")

    if "primary" in report["datasets"]:
        p = report["datasets"]["primary"]
        print(f"\n[PRIMARY] Classes detected: {list(p['classes'].keys())}")
        print(f"[PRIMARY] Bounding Box Labels: {p['bounding_boxes']['total_annotation_files']:,} files ({p['bounding_boxes']['format']})")

    if "riceseg" in report["datasets"]:
        r = report["datasets"]["riceseg"]
        pv = r["pairing_verification"]
        print(f"\n[RICESEG] Total Masks: {r['total_masks']:,} across {list(r['classes'].keys())}")
        print(f"[RICESEG] Matched Image-Mask Pairs with Sethy benchmark: {pv['matched_image_mask_pairs']:,} (Pairing Feasible: {pv['pairing_possible']})")

    print("\n" + "=" * 75)
    print(" Reports successfully generated:")
    print(f"   * {json_path.relative_to(PROJECT_ROOT)}")
    print(f"   * {md_path.relative_to(PROJECT_ROOT)}")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
