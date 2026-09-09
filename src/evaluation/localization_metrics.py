"""Lesion Localization Evaluation Metrics for RiceGuard.

Computes IoU matching, Mean IoU, Precision, Recall, F1, and Healthy lesion false-positive rate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from src.data.annotations import BoundingBox


def compute_box_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two normalized boxes [xc, yc, w, h].

    Args:
        box1: Dict with 'x_center', 'y_center', 'width', 'height'.
        box2: Dict with 'x_center', 'y_center', 'width', 'height'.

    Returns:
        float: IoU value in [0.0, 1.0].
    """
    b1_x1 = box1["x_center"] - (box1["width"] / 2.0)
    b1_x2 = box1["x_center"] + (box1["width"] / 2.0)
    b1_y1 = box1["y_center"] - (box1["height"] / 2.0)
    b1_y2 = box1["y_center"] + (box1["height"] / 2.0)

    b2_x1 = box2["x_center"] - (box2["width"] / 2.0)
    b2_x2 = box2["x_center"] + (box2["width"] / 2.0)
    b2_y1 = box2["y_center"] - (box2["height"] / 2.0)
    b2_y2 = box2["y_center"] + (box2["height"] / 2.0)

    # Intersection coordinates
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union area
    b1_area = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    b2_area = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area <= 1e-8:
        return 0.0

    return float(inter_area / union_area)


def evaluate_single_image_localization(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Union[BoundingBox, Dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate localization predictions against ground truth for a single image.

    Greedy matching: sorts pred_boxes by confidence descending and assigns to highest IoU gt_box.
    """
    # Standardize ground truth boxes to dicts
    gt_dicts: List[Dict[str, float]] = []
    for b in gt_boxes:
        if isinstance(b, BoundingBox):
            gt_dicts.append({"x_center": b.x_center, "y_center": b.y_center, "width": b.width, "height": b.height})
        elif isinstance(b, dict):
            gt_dicts.append({"x_center": float(b["x_center"]), "y_center": float(b["y_center"]), "width": float(b["width"]), "height": float(b["height"])})

    sorted_preds = sorted(pred_boxes, key=lambda x: x.get("confidence", 1.0), reverse=True)
    matched_gt = set()
    matched_ious: List[float] = []

    tp = 0
    fp = 0

    for p in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_dicts):
            if g_idx in matched_gt:
                continue
            iou = compute_box_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_dicts) - len(matched_gt)

    return {
        "num_gt": len(gt_dicts),
        "num_pred": len(pred_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_ious": matched_ious,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def evaluate_dataset_localization(
    all_predictions: List[List[Dict[str, Any]]],
    all_ground_truths: List[List[Union[BoundingBox, Dict[str, Any]]]],
    canonical_classes: List[str],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute aggregate localization metrics across an entire dataset split.

    Args:
        all_predictions: List of detected box lists per image.
        all_ground_truths: List of ground-truth box lists per image.
        canonical_classes: List of canonical class names for each image.
        iou_threshold: IoU threshold for counting a detection as True Positive (default: 0.5).

    Returns:
        Dictionary of aggregate localization metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    all_matched_ious: List[float] = []

    # Healthy false positive tracking
    healthy_images_count = 0
    healthy_images_with_false_detections = 0
    healthy_false_detections_count = 0

    # Per-class stats
    class_stats: Dict[str, Dict[str, Any]] = {}

    for preds, gts, cls_name in zip(all_predictions, all_ground_truths, canonical_classes):
        img_eval = evaluate_single_image_localization(preds, gts, iou_threshold=iou_threshold)

        total_tp += img_eval["tp"]
        total_fp += img_eval["fp"]
        total_fn += img_eval["fn"]
        total_gt += img_eval["num_gt"]
        total_pred += img_eval["num_pred"]
        all_matched_ious.extend(img_eval["matched_ious"])

        # Track Healthy FPR
        if cls_name == "Healthy":
            healthy_images_count += 1
            if len(preds) > 0:
                healthy_images_with_false_detections += 1
                healthy_false_detections_count += len(preds)

        # Track class-level
        if cls_name not in class_stats:
            class_stats[cls_name] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "matched_ious": []}
        class_stats[cls_name]["tp"] += img_eval["tp"]
        class_stats[cls_name]["fp"] += img_eval["fp"]
        class_stats[cls_name]["fn"] += img_eval["fn"]
        class_stats[cls_name]["gt"] += img_eval["num_gt"]
        class_stats[cls_name]["pred"] += img_eval["num_pred"]
        class_stats[cls_name]["matched_ious"].extend(img_eval["matched_ious"])

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    healthy_fpr = (
        healthy_images_with_false_detections / max(healthy_images_count, 1)
        if healthy_images_count > 0
        else 0.0
    )

    per_class_summary = {}
    for c_name, st in class_stats.items():
        c_prec = st["tp"] / max(st["tp"] + st["fp"], 1)
        c_rec = st["tp"] / max(st["tp"] + st["fn"], 1)
        c_f1 = 2 * (c_prec * c_rec) / max(c_prec + c_rec, 1e-8)
        per_class_summary[c_name] = {
            "total_gt_boxes": st["gt"],
            "total_pred_boxes": st["pred"],
            "tp": st["tp"],
            "fp": st["fp"],
            "fn": st["fn"],
            "precision": float(round(c_prec, 4)),
            "recall": float(round(c_rec, 4)),
            "f1": float(round(c_f1, 4)),
            "mean_iou": float(round(float(np.mean(st["matched_ious"])), 4)) if st["matched_ious"] else 0.0,
        }

    return {
        "iou_threshold": iou_threshold,
        "total_images": len(canonical_classes),
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "localization_precision": float(round(precision, 4)),
        "localization_recall": float(round(recall, 4)),
        "localization_f1": float(round(f1, 4)),
        "mean_matched_iou": float(round(mean_iou, 4)),
        "healthy_evaluation": {
            "healthy_images_total": healthy_images_count,
            "healthy_images_with_detections": healthy_images_with_false_detections,
            "healthy_false_positive_rate": float(round(healthy_fpr, 4)),
            "healthy_total_false_boxes": healthy_false_detections_count,
        },
        "per_class_localization": per_class_summary,
    }
