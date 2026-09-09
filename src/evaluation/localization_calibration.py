"""Validation-Only Localization Calibration and Decoding Selection for RiceGuard Phase 3B.

Performs deterministic confidence threshold sweeping and Top-K decoding selection
exclusively on the validation split (primary_val.csv). Internal test data is strictly prohibited.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.annotations import BoundingBox
from src.data.grid_assignment import decode_grid_predictions
from src.evaluation.localization_metrics import (
    evaluate_dataset_localization,
    evaluate_single_image_localization,
)

matplotlib.use("Agg")

DEFAULT_THRESHOLD_GRID = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

DEFAULT_TOP_K_GRID = [3, 5, 10, None]


def evaluate_decoding_configuration(
    all_raw_loc_outputs: List[torch.Tensor],
    all_ground_truths: List[List[Union[BoundingBox, Dict[str, Any]]]],
    canonical_classes: List[str],
    conf_threshold: float,
    top_k: Optional[int] = None,
    grid_size: int = 7,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a specific (confidence_threshold, top_k) decoding configuration.

    Args:
        all_raw_loc_outputs: List of raw model localization outputs [5, S, S] per image.
        all_ground_truths: List of ground-truth bounding box lists per image.
        canonical_classes: List of canonical class names per image.
        conf_threshold: Confidence threshold for lesion cells.
        top_k: Optional Top-K maximum bounding boxes per image (None = unlimited).
        grid_size: Spatial grid size S (default: 7).
        iou_threshold: IoU threshold for True Positive (default: 0.5).

    Returns:
        Dictionary with aggregate metrics and per-image box counts.
    """
    all_pred_boxes: List[List[Dict[str, Any]]] = []
    total_boxes = 0
    all_matched_ious: List[float] = []

    for raw_loc in all_raw_loc_outputs:
        boxes = decode_grid_predictions(
            raw_loc,
            conf_threshold=conf_threshold,
            grid_size=grid_size,
            top_k=top_k,
        )
        all_pred_boxes.append(boxes)
        total_boxes += len(boxes)

    # Dataset-level evaluation
    metrics = evaluate_dataset_localization(
        all_predictions=all_pred_boxes,
        all_ground_truths=all_ground_truths,
        canonical_classes=canonical_classes,
        iou_threshold=iou_threshold,
    )

    # Compute median IoU across all matched boxes
    for preds, gts in zip(all_pred_boxes, all_ground_truths):
        single_res = evaluate_single_image_localization(preds, gts, iou_threshold=iou_threshold)
        all_matched_ious.extend(single_res["matched_ious"])

    median_iou = float(np.median(all_matched_ious)) if all_matched_ious else 0.0
    num_images = max(len(canonical_classes), 1)
    avg_boxes_per_image = total_boxes / num_images

    top_k_str = "unlimited" if top_k is None else str(top_k)

    return {
        "confidence_threshold": float(round(conf_threshold, 4)),
        "top_k": top_k,
        "top_k_str": top_k_str,
        "precision": metrics["localization_precision"],
        "recall": metrics["localization_recall"],
        "f1": metrics["localization_f1"],
        "mean_matched_iou": metrics["mean_matched_iou"],
        "median_matched_iou": float(round(median_iou, 4)),
        "healthy_false_positive_rate": metrics["healthy_evaluation"]["healthy_false_positive_rate"],
        "total_predicted_boxes": total_boxes,
        "avg_predicted_boxes_per_image": float(round(avg_boxes_per_image, 2)),
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "raw_metrics": metrics,
    }


def run_threshold_sweep(
    all_raw_loc_outputs: List[torch.Tensor],
    all_ground_truths: List[List[Union[BoundingBox, Dict[str, Any]]]],
    canonical_classes: List[str],
    thresholds: Optional[List[float]] = None,
    grid_size: int = 7,
    iou_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run threshold sweep with unlimited Top-K on validation data.

    Returns:
        sweep_results: List of metric dicts for each threshold.
        best_config: Deterministically selected best threshold dict.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID

    results: List[Dict[str, Any]] = []
    for th in thresholds:
        res = evaluate_decoding_configuration(
            all_raw_loc_outputs=all_raw_loc_outputs,
            all_ground_truths=all_ground_truths,
            canonical_classes=canonical_classes,
            conf_threshold=th,
            top_k=None,
            grid_size=grid_size,
            iou_threshold=iou_threshold,
        )
        results.append(res)

    best_config = select_best_configuration(results)
    return results, best_config


def run_joint_threshold_topk_sweep(
    all_raw_loc_outputs: List[torch.Tensor],
    all_ground_truths: List[List[Union[BoundingBox, Dict[str, Any]]]],
    canonical_classes: List[str],
    thresholds: Optional[List[float]] = None,
    top_k_values: Optional[List[Optional[int]]] = None,
    grid_size: int = 7,
    iou_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run full Cartesian product sweep of threshold x Top-K on validation data.

    Returns:
        sweep_results: List of metric dicts for each (threshold, top_k) pair.
        best_config: Deterministically selected best decoding configuration dict.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID
    if top_k_values is None:
        top_k_values = DEFAULT_TOP_K_GRID

    results: List[Dict[str, Any]] = []
    for top_k in top_k_values:
        for th in thresholds:
            res = evaluate_decoding_configuration(
                all_raw_loc_outputs=all_raw_loc_outputs,
                all_ground_truths=all_ground_truths,
                canonical_classes=canonical_classes,
                conf_threshold=th,
                top_k=top_k,
                grid_size=grid_size,
                iou_threshold=iou_threshold,
            )
            results.append(res)

    best_config = select_best_configuration(results)
    return results, best_config


def select_best_configuration(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best decoding configuration using strict hierarchical criteria.

    Hierarchy:
        1. Primary: Maximize Localization F1.
        2. Tie-break 1: Higher Localization Precision.
        3. Tie-break 2: Lower Healthy False Positive Rate.
        4. Tie-break 3: Lower Average Predicted Boxes per Image.
        5. Tie-break 4: Higher Mean Matched IoU.
    """
    if not results:
        raise ValueError("Results list cannot be empty for configuration selection.")

    def sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        f1 = item["f1"]
        prec = item["precision"]
        healthy_fpr = item["healthy_false_positive_rate"]
        avg_boxes = item["avg_predicted_boxes_per_image"]
        mean_iou = item["mean_matched_iou"]

        # Python sorts ascending, so negate maximizing criteria
        return (
            -float(round(f1, 4)),
            -float(round(prec, 4)),
            float(round(healthy_fpr, 4)),
            float(round(avg_boxes, 2)),
            -float(round(mean_iou, 4)),
        )

    sorted_results = sorted(results, key=sort_key)
    return sorted_results[0]


def save_calibration_artifacts(
    threshold_results: List[Dict[str, Any]],
    best_threshold_config: Dict[str, Any],
    joint_results: List[Dict[str, Any]],
    best_joint_config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Save all calibration tables, JSON configs, and diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}

    # 1. threshold_sweep.csv
    th_csv_path = output_dir / "threshold_sweep.csv"
    with open(th_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "confidence_threshold",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "median_matched_iou",
            "healthy_false_positive_rate",
            "total_predicted_boxes",
            "avg_predicted_boxes_per_image",
        ])
        for r in threshold_results:
            writer.writerow([
                f"{r['confidence_threshold']:.2f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['mean_matched_iou']:.4f}",
                f"{r['median_matched_iou']:.4f}",
                f"{r['healthy_false_positive_rate'] * 100:.2f}%",
                r["total_predicted_boxes"],
                f"{r['avg_predicted_boxes_per_image']:.2f}",
            ])
    artifacts["threshold_sweep_csv"] = th_csv_path

    # 2. selected_threshold.json
    sel_th_json_path = output_dir / "selected_threshold.json"
    th_clean = {k: v for k, v in best_threshold_config.items() if k != "raw_metrics"}
    with open(sel_th_json_path, "w", encoding="utf-8") as f:
        json.dump(th_clean, f, indent=2)
    artifacts["selected_threshold_json"] = sel_th_json_path

    # 3. threshold_sweep.png
    th_png_path = output_dir / "threshold_sweep.png"
    plt.figure(figsize=(9, 5))
    ths = [r["confidence_threshold"] for r in threshold_results]
    precs = [r["precision"] for r in threshold_results]
    recs = [r["recall"] for r in threshold_results]
    f1s = [r["f1"] for r in threshold_results]
    hfprs = [r["healthy_false_positive_rate"] for r in threshold_results]

    plt.plot(ths, precs, marker="o", label="Precision", color="#1f77b4", lw=2)
    plt.plot(ths, recs, marker="s", label="Recall", color="#2ca02c", lw=2)
    plt.plot(ths, f1s, marker="^", label="Localization F1", color="#d62728", lw=2.5)
    plt.plot(ths, hfprs, marker="x", label="Healthy FP Rate", color="#ff7f0e", linestyle="--", lw=1.5)
    plt.axvline(
        x=best_threshold_config["confidence_threshold"],
        color="purple",
        linestyle=":",
        label=f"Selected Thresh ({best_threshold_config['confidence_threshold']:.2f})",
    )
    plt.title("Phase 3B — Validation Confidence Threshold Sweep", fontsize=12, fontweight="bold")
    plt.xlabel("Confidence Threshold", fontsize=10)
    plt.ylabel("Score / Rate", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(th_png_path, dpi=150)
    plt.close()
    artifacts["threshold_sweep_png"] = th_png_path

    # 4. joint_threshold_topk_sweep.csv
    joint_csv_path = output_dir / "joint_threshold_topk_sweep.csv"
    with open(joint_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "confidence_threshold",
            "top_k",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "median_matched_iou",
            "healthy_false_positive_rate",
            "total_predicted_boxes",
            "avg_predicted_boxes_per_image",
        ])
        for r in joint_results:
            writer.writerow([
                f"{r['confidence_threshold']:.2f}",
                r["top_k_str"],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['mean_matched_iou']:.4f}",
                f"{r['median_matched_iou']:.4f}",
                f"{r['healthy_false_positive_rate'] * 100:.2f}%",
                r["total_predicted_boxes"],
                f"{r['avg_predicted_boxes_per_image']:.2f}",
            ])
    artifacts["joint_threshold_topk_sweep_csv"] = joint_csv_path

    # 5. selected_decoding_config.json
    sel_joint_json_path = output_dir / "selected_decoding_config.json"
    joint_clean = {k: v for k, v in best_joint_config.items() if k != "raw_metrics"}
    with open(sel_joint_json_path, "w", encoding="utf-8") as f:
        json.dump(joint_clean, f, indent=2)
    artifacts["selected_decoding_config_json"] = sel_joint_json_path

    # 6. joint_threshold_topk_sweep.png (Heatmap / Comparison chart)
    joint_png_path = output_dir / "joint_threshold_topk_sweep.png"
    df_joint = pd.DataFrame(joint_results)
    top_k_categories = sorted(df_joint["top_k_str"].unique(), key=lambda x: 999 if x == "unlimited" else int(x))

    plt.figure(figsize=(10, 5.5))
    for tk in top_k_categories:
        sub = df_joint[df_joint["top_k_str"] == tk].sort_values("confidence_threshold")
        label = f"Top-K = {tk}"
        plt.plot(sub["confidence_threshold"], sub["f1"], marker="o", label=label, lw=2)

    best_th = best_joint_config["confidence_threshold"]
    best_tk = best_joint_config["top_k_str"]
    plt.axvline(
        x=best_th,
        color="black",
        linestyle=":",
        label=f"Selected: Th={best_th:.2f}, K={best_tk} (F1={best_joint_config['f1']:.4f})",
    )
    plt.title("Phase 3B — Joint Confidence Threshold × Top-K Sweep (Validation F1)", fontsize=12, fontweight="bold")
    plt.xlabel("Confidence Threshold", fontsize=10)
    plt.ylabel("Localization F1", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(joint_png_path, dpi=150)
    plt.close()
    artifacts["joint_threshold_topk_sweep_png"] = joint_png_path

    return artifacts
