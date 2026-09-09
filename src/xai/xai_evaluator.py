"""XAI Evaluation Orchestrator for Phase 4 Quantitative Validation.

Conducts end-to-end evaluation comparing Phase 2B Baseline vs Phase 3B Refined Multi-Task:
- Generates Grad-CAM attribution maps
- Computes spatial lesion grounding metrics (Energy Inside, IoU, Pointing Game)
- Stratifies by classification correctness
- Computes paired statistical significance (Wilcoxon, McNemar)
- Executes perturbation faithfulness benchmarking
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.xai.attribution_metrics import (
    compute_attribution_iou,
    compute_distribution_summary,
    compute_energy_inside_mask,
    compute_energy_outside_mask,
    compute_healthy_xai_metrics,
    compute_pointing_game,
)
from src.xai.faithfulness import evaluate_sample_faithfulness
from src.xai.gradcam import GradCAM


def run_mcnemar_test(
    table: np.ndarray,
) -> Tuple[float, float]:
    """Run McNemar's test for paired binary outcomes with continuity correction / exact binomial.

    Args:
        table: 2x2 contingency matrix [[n00, n01], [n10, n11]],
               where n01 = Model 1 incorrect & Model 2 correct,
                     n10 = Model 1 correct & Model 2 incorrect.

    Returns:
        statistic: Chi-square statistic or exact test statistic.
        p_value: Two-sided p-value.
    """
    b = float(table[0, 1])  # Model 1 Fail, Model 2 Pass
    c = float(table[1, 0])  # Model 1 Pass, Model 2 Fail
    total_discordant = b + c

    if total_discordant == 0:
        return 0.0, 1.0

    if total_discordant < 25:
        # Exact two-sided binomial test
        k = int(min(b, c))
        n = int(total_discordant)
        p_val = float(2.0 * stats.binom.cdf(k, n, 0.5))
        p_val = min(1.0, p_val)
        stat = float(abs(b - c))
        return stat, p_val
    else:
        # Edwards continuity correction
        stat = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_val = float(1.0 - stats.chi2.cdf(stat, df=1))
        return stat, p_val


def run_paired_wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, Any]:
    """Compute paired Wilcoxon signed-rank test between Model A and Model B scores.

    Returns dictionary with stat, p_value, mean_diff, median_diff, effect_size.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = b - a  # Positive means Model B > Model A

    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))

    # Check if all differences are zero
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "effect_size": 0.0,
            "n_paired": len(a),
        }

    try:
        res = stats.wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat = 0.0
        p_val = 1.0

    # Rank-biserial correlation or standardized z effect size
    std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    cohen_d = float(mean_diff / (std_diff + 1e-8)) if std_diff > 1e-8 else 0.0

    return {
        "statistic": stat,
        "p_value": p_val,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "effect_size": cohen_d,
        "n_paired": len(a),
    }


class XAIEvaluator:
    """Evaluates and compares Phase 2B and Phase 3B models against lesion ground-truth masks."""

    def __init__(
        self,
        phase2b_model: nn.Module,
        phase3b_model: nn.Module,
        device: torch.device,
        top_percent_iou: float = 20.0,
        faithfulness_samples: int = 300,
        faithfulness_steps: int = 20,
        seed: int = 42,
    ) -> None:
        self.phase2b_model = phase2b_model.to(device)
        self.phase3b_model = phase3b_model.to(device)
        self.device = device
        self.top_percent_iou = top_percent_iou
        self.faithfulness_samples = faithfulness_samples
        self.faithfulness_steps = faithfulness_steps
        self.seed = seed

        self.gradcam_2b = GradCAM(self.phase2b_model)
        self.gradcam_3b = GradCAM(self.phase3b_model)

    def evaluate_dataset(
        self,
        dataset: Dataset,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Run full quantitative XAI grounding evaluation on all dataset samples."""
        total_samples = len(dataset)
        sample_results: List[Dict[str, Any]] = []

        for idx in range(total_samples):
            img_tensor, mask_tensor, meta = dataset[idx]
            binary_mask = mask_tensor.squeeze().numpy()
            target_cls = meta["class_index"]
            cls_name = meta["canonical_class"]

            # Phase 2B Grad-CAM
            cam_2b, pred_2b, conf_2b = self.gradcam_2b.generate_cam(
                img_tensor,
                target_class=None,  # target predicted class
            )
            is_correct_2b = int(pred_2b == target_cls) if target_cls is not None else 0

            # Phase 3B Grad-CAM (Primary classification attribution)
            cam_3b, pred_3b, conf_3b = self.gradcam_3b.generate_cam(
                img_tensor,
                target_class=None,  # target predicted class
                target_type="classification",
            )
            is_correct_3b = int(pred_3b == target_cls) if target_cls is not None else 0

            # Compute Grounding Metrics (Lesion Disease Classes)
            if cls_name != "Healthy" and float(np.sum(binary_mask)) > 0:
                e_in_2b = compute_energy_inside_mask(cam_2b, binary_mask)
                e_out_2b = compute_energy_outside_mask(cam_2b, binary_mask)
                iou_2b = compute_attribution_iou(cam_2b, binary_mask, top_percent=self.top_percent_iou)
                pt_2b = compute_pointing_game(cam_2b, binary_mask)

                e_in_3b = compute_energy_inside_mask(cam_3b, binary_mask)
                e_out_3b = compute_energy_outside_mask(cam_3b, binary_mask)
                iou_3b = compute_attribution_iou(cam_3b, binary_mask, top_percent=self.top_percent_iou)
                pt_3b = compute_pointing_game(cam_3b, binary_mask)
            else:
                # Healthy leaf metrics (if evaluated)
                compute_healthy_xai_metrics(cam_2b)
                compute_healthy_xai_metrics(cam_3b)
                e_in_2b = e_out_2b = iou_2b = pt_2b = None
                e_in_3b = e_out_3b = iou_3b = pt_3b = None

            record = {
                "sample_id": meta["sample_id"],
                "image_path": meta["image_path"],
                "mask_path": meta["mask_path"],
                "canonical_class": cls_name,
                "class_index": target_cls,
                "lesion_area_fraction": meta.get("lesion_area_fraction", 0.0),
                # Phase 2B Results
                "phase2b_pred": pred_2b,
                "phase2b_conf": conf_2b,
                "phase2b_correct": is_correct_2b,
                "phase2b_energy_in": e_in_2b,
                "phase2b_energy_out": e_out_2b,
                "phase2b_iou": iou_2b,
                "phase2b_pointing": pt_2b,
                # Phase 3B Results
                "phase3b_pred": pred_3b,
                "phase3b_conf": conf_3b,
                "phase3b_correct": is_correct_3b,
                "phase3b_energy_in": e_in_3b,
                "phase3b_energy_out": e_out_3b,
                "phase3b_iou": iou_3b,
                "phase3b_pointing": pt_3b,
            }
            sample_results.append(record)

            if progress_callback and (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
                progress_callback(idx + 1, total_samples, "evaluation")

        df = pd.DataFrame(sample_results)
        return {"df": df, "sample_results": sample_results}

    def compute_stratified_summaries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregated metrics stratified by classification correctness."""
        disease_df = df[df["canonical_class"] != "Healthy"].copy()

        # Stratifications
        strats = {
            "all_samples": disease_df,
            "correct_2b_only": disease_df[disease_df["phase2b_correct"] == 1],
            "correct_3b_only": disease_df[disease_df["phase3b_correct"] == 1],
            "correct_both": disease_df[(disease_df["phase2b_correct"] == 1) & (disease_df["phase3b_correct"] == 1)],
            "incorrect_any": disease_df[(disease_df["phase2b_correct"] == 0) | (disease_df["phase3b_correct"] == 0)],
        }

        strat_summaries: Dict[str, Any] = {}
        for strat_name, sub_df in strats.items():
            if len(sub_df) == 0:
                continue

            summary_2b = {
                "energy_in": compute_distribution_summary(sub_df["phase2b_energy_in"].dropna().values),
                "energy_out": compute_distribution_summary(sub_df["phase2b_energy_out"].dropna().values),
                "attribution_iou": compute_distribution_summary(sub_df["phase2b_iou"].dropna().values),
                "pointing_accuracy": float(sub_df["phase2b_pointing"].mean()),
                "accuracy": float(sub_df["phase2b_correct"].mean()),
            }

            summary_3b = {
                "energy_in": compute_distribution_summary(sub_df["phase3b_energy_in"].dropna().values),
                "energy_out": compute_distribution_summary(sub_df["phase3b_energy_out"].dropna().values),
                "attribution_iou": compute_distribution_summary(sub_df["phase3b_iou"].dropna().values),
                "pointing_accuracy": float(sub_df["phase3b_pointing"].mean()),
                "accuracy": float(sub_df["phase3b_correct"].mean()),
            }

            # Statistical Comparisons on this stratum
            stat_energy = run_paired_wilcoxon_test(
                sub_df["phase2b_energy_in"].dropna().tolist(),
                sub_df["phase3b_energy_in"].dropna().tolist(),
            )
            stat_iou = run_paired_wilcoxon_test(
                sub_df["phase2b_iou"].dropna().tolist(),
                sub_df["phase3b_iou"].dropna().tolist(),
            )

            # McNemar table for Pointing Game
            pt_2b = sub_df["phase2b_pointing"].values
            pt_3b = sub_df["phase3b_pointing"].values
            n00 = int(np.sum((pt_2b == 0) & (pt_3b == 0)))
            n01 = int(np.sum((pt_2b == 0) & (pt_3b == 1)))
            n10 = int(np.sum((pt_2b == 1) & (pt_3b == 0)))
            n11 = int(np.sum((pt_2b == 1) & (pt_3b == 1)))
            mc_table = np.array([[n00, n01], [n10, n11]])
            mc_stat, mc_pval = run_mcnemar_test(mc_table)

            strat_summaries[strat_name] = {
                "count": len(sub_df),
                "phase2b": summary_2b,
                "phase3b": summary_3b,
                "statistics": {
                    "energy_in_wilcoxon": stat_energy,
                    "attribution_iou_wilcoxon": stat_iou,
                    "pointing_mcnemar": {
                        "statistic": mc_stat,
                        "p_value": mc_pval,
                        "contingency_table": [[n00, n01], [n10, n11]],
                    },
                },
            }

        return strat_summaries

    def compute_per_class_summaries(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute grounding metrics broken down per disease class."""
        classes = sorted(df["canonical_class"].unique())
        per_class: Dict[str, Dict[str, Any]] = {}

        for cls_name in classes:
            sub = df[df["canonical_class"] == cls_name]
            if len(sub) == 0 or cls_name == "Healthy":
                continue

            per_class[cls_name] = {
                "sample_count": len(sub),
                "phase2b_acc": float(sub["phase2b_correct"].mean()),
                "phase3b_acc": float(sub["phase3b_correct"].mean()),
                "phase2b_energy_in_mean": float(sub["phase2b_energy_in"].mean()),
                "phase2b_energy_in_median": float(sub["phase2b_energy_in"].median()),
                "phase3b_energy_in_mean": float(sub["phase3b_energy_in"].mean()),
                "phase3b_energy_in_median": float(sub["phase3b_energy_in"].median()),
                "energy_in_diff": float(sub["phase3b_energy_in"].mean() - sub["phase2b_energy_in"].mean()),
                "phase2b_iou_mean": float(sub["phase2b_iou"].mean()),
                "phase3b_iou_mean": float(sub["phase3b_iou"].mean()),
                "iou_diff": float(sub["phase3b_iou"].mean() - sub["phase2b_iou"].mean()),
                "phase2b_pointing_acc": float(sub["phase2b_pointing"].mean()),
                "phase3b_pointing_acc": float(sub["phase3b_pointing"].mean()),
                "pointing_diff": float(sub["phase3b_pointing"].mean() - sub["phase2b_pointing"].mean()),
            }

        return per_class

    def run_faithfulness_benchmark(
        self,
        dataset: Dataset,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Run Deletion and Insertion perturbation benchmark on fixed subset."""
        rng = np.random.RandomState(self.seed)
        total_ds = len(dataset)
        n_samples = min(self.faithfulness_samples, total_ds)
        selected_indices = rng.choice(total_ds, size=n_samples, replace=False)

        del_aucs_2b, ins_aucs_2b = [], []
        del_aucs_3b, ins_aucs_3b = [], []
        del_curves_2b, ins_curves_2b = [], []
        del_curves_3b, ins_curves_3b = [], []

        for i, idx in enumerate(selected_indices):
            img_tensor, _, meta = dataset[int(idx)]
            target_cls = meta["class_index"]

            # Phase 2B Grad-CAM & Faithfulness
            cam_2b, pred_2b, _ = self.gradcam_2b.generate_cam(img_tensor, target_class=target_cls)
            res_2b = evaluate_sample_faithfulness(
                model=self.phase2b_model,
                image_tensor=img_tensor,
                attribution_map=cam_2b,
                target_class=target_cls,
                num_steps=self.faithfulness_steps,
            )
            del_aucs_2b.append(res_2b["deletion_auc"])
            ins_aucs_2b.append(res_2b["insertion_auc"])
            del_curves_2b.append(res_2b["deletion_curve"])
            ins_curves_2b.append(res_2b["insertion_curve"])

            # Phase 3B Grad-CAM & Faithfulness
            cam_3b, pred_3b, _ = self.gradcam_3b.generate_cam(img_tensor, target_class=target_cls)
            res_3b = evaluate_sample_faithfulness(
                model=self.phase3b_model,
                image_tensor=img_tensor,
                attribution_map=cam_3b,
                target_class=target_cls,
                num_steps=self.faithfulness_steps,
            )
            del_aucs_3b.append(res_3b["deletion_auc"])
            ins_aucs_3b.append(res_3b["insertion_auc"])
            del_curves_3b.append(res_3b["deletion_curve"])
            ins_curves_3b.append(res_3b["insertion_curve"])

            if progress_callback and (i + 1) % 25 == 0 or (i + 1) == n_samples:
                progress_callback(i + 1, n_samples, "faithfulness")

        fractions = [k / float(self.faithfulness_steps) for k in range(self.faithfulness_steps + 1)]

        # Statistical comparison of Faithfulness AUCs
        del_wilcoxon = run_paired_wilcoxon_test(del_aucs_2b, del_aucs_3b)
        ins_wilcoxon = run_paired_wilcoxon_test(ins_aucs_2b, ins_aucs_3b)

        return {
            "sample_count": n_samples,
            "step_fractions": fractions,
            "phase2b": {
                "mean_deletion_auc": float(np.mean(del_aucs_2b)),
                "std_deletion_auc": float(np.std(del_aucs_2b, ddof=1)),
                "mean_insertion_auc": float(np.mean(ins_aucs_2b)),
                "std_insertion_auc": float(np.std(ins_aucs_2b, ddof=1)),
                "mean_deletion_curve": np.mean(del_curves_2b, axis=0).tolist(),
                "mean_insertion_curve": np.mean(ins_curves_2b, axis=0).tolist(),
            },
            "phase3b": {
                "mean_deletion_auc": float(np.mean(del_aucs_3b)),
                "std_deletion_auc": float(np.std(del_aucs_3b, ddof=1)),
                "mean_insertion_auc": float(np.mean(ins_aucs_3b)),
                "std_insertion_auc": float(np.std(ins_aucs_3b, ddof=1)),
                "mean_deletion_curve": np.mean(del_curves_3b, axis=0).tolist(),
                "mean_insertion_curve": np.mean(ins_curves_3b, axis=0).tolist(),
            },
            "statistics": {
                "deletion_wilcoxon": del_wilcoxon,
                "insertion_wilcoxon": ins_wilcoxon,
            },
        }
