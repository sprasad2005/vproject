"""Failure-Case Analysis Engine for Phase 5 Consolidation.

Analyzes fine-grained performance differences and failure modes between
Phase 2B (EfficientNet-B0 Baseline) and Phase 3B (Refined Multi-Task).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import RiceLeafDataset
from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES
from src.evaluation.localization_metrics import compute_box_iou


class FailureAnalyzer:
    """Performs deterministic comparative failure mode analysis between Phase 2B and Phase 3B."""

    def __init__(
        self,
        phase2b_model: nn.Module,
        phase3b_model: nn.Module,
        device: torch.device,
        conf_threshold: float = 0.60,
        top_k: int = 3,
        seed: int = 42,
    ) -> None:
        self.phase2b_model = phase2b_model.to(device).eval()
        self.phase3b_model = phase3b_model.to(device).eval()
        self.device = device
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.seed = seed

    def evaluate_test_set(
        self,
        test_dataset: RiceLeafDataset,
    ) -> Dict[str, Any]:
        """Run inference across the test split and categorize all sample outcomes."""
        records: List[Dict[str, Any]] = []

        with torch.no_grad():
            for idx in range(len(test_dataset)):
                img_tensor, cls_label, meta = test_dataset[idx]
                sample_meta = test_dataset.samples[idx]
                img_batch = img_tensor.unsqueeze(0).to(self.device)

                # Phase 2B Inference
                out_2b = self.phase2b_model(img_batch)
                probs_2b = torch.softmax(out_2b, dim=1).cpu().numpy()[0]
                pred_2b = int(np.argmax(probs_2b))
                conf_2b = float(probs_2b[pred_2b])

                # Phase 3B Inference
                cls_logits_3b, loc_output_3b = self.phase3b_model(img_batch)
                probs_3b = torch.softmax(cls_logits_3b, dim=1).cpu().numpy()[0]
                pred_3b = int(np.argmax(probs_3b))
                conf_3b = float(probs_3b[pred_3b])

                # Phase 3B Box Decoding
                pred_boxes = self.phase3b_model.predict_boxes(
                    img_batch,
                    conf_threshold=self.conf_threshold,
                )[0]
                if self.top_k is not None and len(pred_boxes) > self.top_k:
                    pred_boxes = sorted(pred_boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)[: self.top_k]

                # Ground Truth info
                gt_cls = int(cls_label)
                gt_cls_name = CANONICAL_PRIMARY_CLASSES[gt_cls]
                pred_2b_name = CANONICAL_PRIMARY_CLASSES[pred_2b]
                pred_3b_name = CANONICAL_PRIMARY_CLASSES[pred_3b]

                raw_boxes = meta.get("boxes", [])
                gt_boxes = []
                for b in raw_boxes:
                    if hasattr(b, "x_center"):
                        gt_boxes.append({
                            "x_center": float(b.x_center),
                            "y_center": float(b.y_center),
                            "width": float(b.width),
                            "height": float(b.height),
                        })
                    elif isinstance(b, dict) and "x_center" in b:
                        gt_boxes.append(b)

                # Bounding Box Matching IoU
                max_matched_iou = 0.0
                has_tp_box = False
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    for pb in pred_boxes:
                        for gb in gt_boxes:
                            iou = compute_box_iou(pb, gb)
                            if iou > max_matched_iou:
                                max_matched_iou = iou
                            if iou >= 0.5:
                                has_tp_box = True

                corr_2b = int(pred_2b == gt_cls)
                corr_3b = int(pred_3b == gt_cls)

                # Categorize Failure / Success Mode
                cat = "other"
                if corr_2b == 1 and corr_3b == 0:
                    cat = "baseline_wins"
                elif corr_3b == 1 and corr_2b == 0:
                    cat = "multitask_wins"
                elif corr_2b == 0 and corr_3b == 0:
                    cat = "mutual_errors"
                elif corr_2b == 1 and corr_3b == 1:
                    cat = "mutual_success"

                # Sub-failure conditions
                is_healthy_fp = bool(gt_cls_name == "Healthy" and len(pred_boxes) > 0)
                is_lesion_miss = bool(gt_cls_name != "Healthy" and len(gt_boxes) > 0 and not has_tp_box)
                is_high_conf_poor_loc = bool(gt_cls_name != "Healthy" and conf_3b > 0.80 and max_matched_iou < 0.20)
                is_good_loc_bad_cls = bool(corr_3b == 0 and max_matched_iou >= 0.50)

                records.append({
                    "sample_idx": idx,
                    "image_path": str(sample_meta.get("image_path", "")),
                    "canonical_class": gt_cls_name,
                    "gt_class_idx": gt_cls,
                    "phase2b_pred_class": pred_2b_name,
                    "phase2b_pred_idx": pred_2b,
                    "phase2b_conf": conf_2b,
                    "phase2b_correct": corr_2b,
                    "phase3b_pred_class": pred_3b_name,
                    "phase3b_pred_idx": pred_3b,
                    "phase3b_conf": conf_3b,
                    "phase3b_correct": corr_3b,
                    "category": cat,
                    "num_gt_boxes": len(gt_boxes),
                    "num_pred_boxes": len(pred_boxes),
                    "max_matched_iou": max_matched_iou,
                    "has_tp_box": has_tp_box,
                    "is_healthy_fp": is_healthy_fp,
                    "is_lesion_miss": is_lesion_miss,
                    "is_high_conf_poor_loc": is_high_conf_poor_loc,
                    "is_good_loc_bad_cls": is_good_loc_bad_cls,
                })

        df = pd.DataFrame(records)
        return {"df": df, "records": records}

    def summarize_failure_modes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute structured quantitative summary of failure modes."""
        total = len(df)
        baseline_wins = df[df["category"] == "baseline_wins"]
        multitask_wins = df[df["category"] == "multitask_wins"]
        mutual_errors = df[df["category"] == "mutual_errors"]
        mutual_success = df[df["category"] == "mutual_success"]

        healthy_df = df[df["canonical_class"] == "Healthy"]
        healthy_fps = healthy_df[healthy_df["is_healthy_fp"]]

        disease_df = df[df["canonical_class"] != "Healthy"]
        lesion_misses = disease_df[disease_df["is_lesion_miss"]]
        high_conf_poor_loc = disease_df[disease_df["is_high_conf_poor_loc"]]
        good_loc_bad_cls = df[df["is_good_loc_bad_cls"]]

        summary = {
            "total_samples": total,
            "mutual_success": {
                "count": len(mutual_success),
                "percentage": float(len(mutual_success) / total * 100.0),
            },
            "baseline_wins": {
                "count": len(baseline_wins),
                "percentage": float(len(baseline_wins) / total * 100.0),
                "avg_phase2b_conf": float(baseline_wins["phase2b_conf"].mean()) if len(baseline_wins) > 0 else 0.0,
                "avg_phase3b_conf": float(baseline_wins["phase3b_conf"].mean()) if len(baseline_wins) > 0 else 0.0,
            },
            "multitask_wins": {
                "count": len(multitask_wins),
                "percentage": float(len(multitask_wins) / total * 100.0),
                "avg_phase2b_conf": float(multitask_wins["phase2b_conf"].mean()) if len(multitask_wins) > 0 else 0.0,
                "avg_phase3b_conf": float(multitask_wins["phase3b_conf"].mean()) if len(multitask_wins) > 0 else 0.0,
            },
            "mutual_errors": {
                "count": len(mutual_errors),
                "percentage": float(len(mutual_errors) / total * 100.0),
            },
            "healthy_false_positives": {
                "count": len(healthy_fps),
                "healthy_total": len(healthy_df),
                "percentage": float(len(healthy_fps) / len(healthy_df) * 100.0) if len(healthy_df) > 0 else 0.0,
            },
            "disease_lesion_misses": {
                "count": len(lesion_misses),
                "disease_total": len(disease_df),
                "percentage": float(len(lesion_misses) / len(disease_df) * 100.0) if len(disease_df) > 0 else 0.0,
            },
            "high_conf_poor_localization": {
                "count": len(high_conf_poor_loc),
                "percentage": float(len(high_conf_poor_loc) / total * 100.0),
            },
            "good_localization_incorrect_classification": {
                "count": len(good_loc_bad_cls),
                "percentage": float(len(good_loc_bad_cls) / total * 100.0),
            },
        }

        return summary

    def select_deterministic_failure_samples(
        self,
        df: pd.DataFrame,
        num_per_category: int = 3,
    ) -> Dict[str, List[int]]:
        """Select representative sample indices for each failure mode deterministically."""
        categories = {
            "baseline_wins": df[df["category"] == "baseline_wins"],
            "multitask_wins": df[df["category"] == "multitask_wins"],
            "mutual_errors": df[df["category"] == "mutual_errors"],
            "healthy_fp": df[df["is_healthy_fp"]],
            "lesion_miss": df[df["is_lesion_miss"]],
        }

        picks: Dict[str, List[int]] = {}
        for cat_name, sub_df in categories.items():
            if len(sub_df) == 0:
                picks[cat_name] = []
                continue
            sorted_indices = sub_df.index.tolist()
            n_picks = min(num_per_category, len(sorted_indices))
            # Pick evenly spaced deterministic indices
            chosen = [sorted_indices[int(p)] for p in np.linspace(0, len(sorted_indices) - 1, n_picks, dtype=int)]
            picks[cat_name] = chosen

        return picks
