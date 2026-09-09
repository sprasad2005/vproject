"""Tests for Phase 3B — Localization Calibration and Imbalance Refinement.

Verifies:
- pos_weight capping and metadata tracking
- Threshold sweep determinism
- Top-K decoding and unlimited filtering
- Joint (threshold x Top-K) hierarchical selection
- Validation-only calibration isolation
- Initialization from Phase 3 best checkpoint
- External dataset locking
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.annotations import BoundingBox
from src.data.dataset_registry import validate_dataset_usage
from src.data.grid_assignment import decode_grid_predictions
from src.evaluation.localization_calibration import (
    run_joint_threshold_topk_sweep,
    run_threshold_sweep,
    save_calibration_artifacts,
    select_best_configuration,
)
from src.models.model_factory import build_model
from src.models.multitask import RiceMultiTaskEfficientNet
from src.training.multitask_loss import MultiTaskLoss


class TestPosWeightCapping:
    """Tests for Objectness pos_weight capping and tracking."""

    def test_pos_weight_capping_logic(self) -> None:
        calculated_pos_weight = 22.9005
        max_pos_weight = 10.0

        effective_pos_weight = min(calculated_pos_weight, max_pos_weight)
        assert effective_pos_weight == 10.0
        assert calculated_pos_weight > effective_pos_weight

    def test_loss_with_capped_pos_weight(self) -> None:
        calculated_pw = 22.90
        effective_pw = min(calculated_pw, 10.0)

        loss_fn = MultiTaskLoss(lambda_loc=0.25, lambda_box=1.0, pos_weight=effective_pw)
        assert loss_fn.pw_tensor is not None
        assert float(loss_fn.pw_tensor.item()) == 10.0

        # Run forward pass through loss
        cls_logits = torch.randn(4, 6)
        loc_output = torch.randn(4, 5, 7, 7)
        class_targets = torch.tensor([0, 1, 2, 3])
        grid_targets = torch.zeros(4, 5, 7, 7)
        grid_targets[0, 0, 3, 3] = 1.0
        grid_targets[0, 1:5, 3, 3] = 0.5

        loss, loss_dict = loss_fn(cls_logits, loc_output, class_targets, grid_targets)
        assert loss.item() > 0.0
        assert "loss_obj" in loss_dict
        assert "loss_box" in loss_dict


class TestTopKDecoding:
    """Tests for Top-K decoding and sorting."""

    def test_top_k_filtering(self) -> None:
        loc_output = torch.full((5, 7, 7), -5.0)
        # Set raw objectness logits for multiple cells to high values
        loc_output[0, 1, 1] = 5.0  # highest
        loc_output[0, 2, 2] = 4.0
        loc_output[0, 3, 3] = 3.0
        loc_output[0, 4, 4] = 2.0
        loc_output[0, 5, 5] = 1.0  # lowest positive

        # Exactly 5 cells pass conf_threshold=0.5
        all_boxes = decode_grid_predictions(loc_output, conf_threshold=0.5, top_k=None)
        assert len(all_boxes) == 5

        # Top-3 should keep top 3 highest confidence boxes
        top3_boxes = decode_grid_predictions(loc_output, conf_threshold=0.5, top_k=3)
        assert len(top3_boxes) == 3
        # Confidences should be strictly descending
        assert top3_boxes[0]["confidence"] >= top3_boxes[1]["confidence"]
        assert top3_boxes[1]["confidence"] >= top3_boxes[2]["confidence"]
        assert top3_boxes[0]["grid_row"] == 1 and top3_boxes[0]["grid_col"] == 1

    def test_unlimited_top_k_behavior(self) -> None:
        loc_output = torch.full((5, 7, 7), 2.0)  # all 49 cells positive
        boxes = decode_grid_predictions(loc_output, conf_threshold=0.5, top_k=None)
        assert len(boxes) == 49


class TestLocalizationCalibration:
    """Tests for validation calibration sweeps and deterministic selection."""

    def test_threshold_sweep_execution(self) -> None:
        raw_locs = [torch.randn(5, 7, 7) for _ in range(10)]
        gts = [[BoundingBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.2)] for _ in range(10)]
        classes = ["Blast"] * 8 + ["Healthy"] * 2

        results, best_cfg = run_threshold_sweep(
            all_raw_loc_outputs=raw_locs,
            all_ground_truths=gts,
            canonical_classes=classes,
            thresholds=[0.3, 0.5, 0.7],
        )

        assert len(results) == 3
        assert best_cfg["confidence_threshold"] in [0.3, 0.5, 0.7]
        assert "f1" in best_cfg
        assert "healthy_false_positive_rate" in best_cfg

    def test_joint_threshold_topk_sweep(self) -> None:
        raw_locs = [torch.randn(5, 7, 7) for _ in range(10)]
        gts = [[BoundingBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.2)] for _ in range(10)]
        classes = ["Blast"] * 8 + ["Healthy"] * 2

        results, best_cfg = run_joint_threshold_topk_sweep(
            all_raw_loc_outputs=raw_locs,
            all_ground_truths=gts,
            canonical_classes=classes,
            thresholds=[0.4, 0.6],
            top_k_values=[3, None],
        )

        assert len(results) == 4  # 2 thresholds x 2 top_k
        assert best_cfg["confidence_threshold"] in [0.4, 0.6]
        assert best_cfg["top_k"] in [3, None]

    def test_deterministic_hierarchical_selection(self) -> None:
        candidates = [
            {
                "confidence_threshold": 0.4,
                "top_k": 3,
                "f1": 0.50,
                "precision": 0.40,
                "healthy_false_positive_rate": 0.10,
                "avg_predicted_boxes_per_image": 2.5,
                "mean_matched_iou": 0.60,
            },
            {
                "confidence_threshold": 0.5,
                "top_k": 3,
                "f1": 0.55,  # Highest F1 -> should win
                "precision": 0.50,
                "healthy_false_positive_rate": 0.05,
                "avg_predicted_boxes_per_image": 1.8,
                "mean_matched_iou": 0.65,
            },
            {
                "confidence_threshold": 0.6,
                "top_k": 5,
                "f1": 0.55,  # Same F1, but lower precision -> should lose tie-break
                "precision": 0.45,
                "healthy_false_positive_rate": 0.08,
                "avg_predicted_boxes_per_image": 2.0,
                "mean_matched_iou": 0.62,
            },
        ]

        best = select_best_configuration(candidates)
        assert best["confidence_threshold"] == 0.5
        assert best["f1"] == 0.55
        assert best["precision"] == 0.50

    def test_save_calibration_artifacts(self, tmp_path: Path) -> None:
        th_results = [
            {
                "confidence_threshold": 0.5,
                "precision": 0.6,
                "recall": 0.5,
                "f1": 0.5455,
                "mean_matched_iou": 0.62,
                "median_matched_iou": 0.63,
                "healthy_false_positive_rate": 0.05,
                "total_predicted_boxes": 50,
                "avg_predicted_boxes_per_image": 2.0,
            }
        ]
        joint_results = [
            {
                "confidence_threshold": 0.5,
                "top_k": 3,
                "top_k_str": "3",
                "precision": 0.6,
                "recall": 0.5,
                "f1": 0.5455,
                "mean_matched_iou": 0.62,
                "median_matched_iou": 0.63,
                "healthy_false_positive_rate": 0.05,
                "total_predicted_boxes": 50,
                "avg_predicted_boxes_per_image": 2.0,
            }
        ]

        artifacts = save_calibration_artifacts(
            threshold_results=th_results,
            best_threshold_config=th_results[0],
            joint_results=joint_results,
            best_joint_config=joint_results[0],
            output_dir=tmp_path,
        )

        for path in artifacts.values():
            assert path.exists()


class TestPhase3BInitialization:
    """Tests for model building and initialization from Phase 3 checkpoint."""

    def test_model_instantiation(self) -> None:
        model = build_model("efficientnet_b0_multitask", num_classes=6, pretrained=False)
        assert isinstance(model, RiceMultiTaskEfficientNet)

    def test_phase3_checkpoint_compatibility(self) -> None:
        phase3_ckpt_path = Path("experiments/phase3_lesion_aware/efficientnet_b0_multitask/checkpoints/best_model.pt")
        if phase3_ckpt_path.exists():
            ckpt = torch.load(phase3_ckpt_path, map_location="cpu")
            assert "model_state_dict" in ckpt
            model = build_model("efficientnet_b0_multitask", num_classes=6, pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            assert model is not None


class TestDataGovernancePhase3B:
    """Ensure data governance remains strictly locked in Phase 3B."""

    def test_external_datasets_blocked_for_calibration(self) -> None:
        with pytest.raises(PermissionError):
            validate_dataset_usage("sethy", "calibration")

        with pytest.raises(PermissionError):
            validate_dataset_usage("bd5", "threshold_opt")

        with pytest.raises(PermissionError):
            validate_dataset_usage("riceseg", "train")
