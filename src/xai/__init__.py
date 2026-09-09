"""Explainable AI (XAI) and Lesion Grounding Module for RiceGuard."""

from src.xai.attribution_metrics import (
    compute_attribution_iou,
    compute_distribution_summary,
    compute_energy_inside_mask,
    compute_energy_outside_mask,
    compute_healthy_xai_metrics,
    compute_pointing_game,
)
from src.xai.faithfulness import (
    evaluate_faithfulness_batch,
    evaluate_sample_faithfulness,
)
from src.xai.gradcam import GradCAM
from src.xai.mask_processing import (
    extract_mask_bboxes,
    get_border_margin_mask,
    validate_binary_mask,
)
from src.xai.xai_evaluator import (
    XAIEvaluator,
    run_mcnemar_test,
    run_paired_wilcoxon_test,
)

__all__ = [
    "GradCAM",
    "compute_energy_inside_mask",
    "compute_energy_outside_mask",
    "compute_attribution_iou",
    "compute_pointing_game",
    "compute_healthy_xai_metrics",
    "compute_distribution_summary",
    "evaluate_sample_faithfulness",
    "evaluate_faithfulness_batch",
    "validate_binary_mask",
    "get_border_margin_mask",
    "extract_mask_bboxes",
    "XAIEvaluator",
    "run_mcnemar_test",
    "run_paired_wilcoxon_test",
]
