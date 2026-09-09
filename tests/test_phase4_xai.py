"""Test Suite for Phase 4: Quantitative Lesion-Grounded XAI Validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.dataset_registry import validate_dataset_usage
from src.xai.attribution_metrics import (
    compute_attribution_iou,
    compute_energy_inside_mask,
    compute_energy_outside_mask,
    compute_healthy_xai_metrics,
    compute_pointing_game,
)
from src.xai.faithfulness import (
    evaluate_sample_faithfulness,
    integrate_auc,
)
from src.xai.gradcam import GradCAM
from src.xai.mask_processing import (
    extract_mask_bboxes,
    get_border_margin_mask,
    validate_binary_mask,
)
from src.xai.xai_evaluator import (
    run_mcnemar_test,
    run_paired_wilcoxon_test,
)


class DummyClassifier(nn.Module):
    """Simple conv model for testing Grad-CAM and XAI metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        p = self.pool(h)
        return self.fc(p.flatten(1))


class DummyMultiTask(nn.Module):
    """Simple multi-task conv model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, 6)
        self.loc_head = nn.Conv2d(32, 5, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        cls_logits = self.classifier(self.pool(feat).flatten(1))
        loc_out = self.loc_head(feat)
        return cls_logits, loc_out


class TestDataGovernance:
    """Strict data governance enforcement tests."""

    def test_riceseg_allowed_for_xai_only(self):
        # Must succeed for xai_ground_truth_only
        validate_dataset_usage("riceseg", "xai_ground_truth_only")

    def test_riceseg_prohibited_for_training(self):
        with pytest.raises(PermissionError):
            validate_dataset_usage("riceseg", "primary_training")

    def test_riceseg_prohibited_for_validation(self):
        with pytest.raises(PermissionError):
            validate_dataset_usage("riceseg", "primary_validation")

    def test_riceseg_prohibited_for_model_selection(self):
        with pytest.raises(PermissionError):
            validate_dataset_usage("riceseg", "hyperparameter_tuning")


class TestMaskProcessing:
    """Mask validation, bounding box extraction, and border margins."""

    def test_validate_binary_mask_tensor(self):
        t = torch.tensor([[[0.0, 0.8], [0.1, 1.0]]])
        binary = validate_binary_mask(t, threshold=0.5)
        assert binary.shape == (2, 2)
        assert np.array_equal(binary, np.array([[0.0, 1.0], [0.0, 1.0]]))

    def test_border_margin_mask(self):
        mask = get_border_margin_mask(height=20, width=20, border_ratio=0.10)
        assert mask.shape == (20, 20)
        # Corners should be 1.0 (border)
        assert mask[0, 0] == 1.0
        assert mask[19, 19] == 1.0
        # Center should be 0.0
        assert mask[10, 10] == 0.0

    def test_extract_mask_bboxes(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:20, 10:25] = 1.0
        bboxes = extract_mask_bboxes(mask)
        assert len(bboxes) >= 1
        x1, y1, x2, y2 = bboxes[0]
        assert x1 == 10 and y1 == 10 and x2 == 25 and y2 == 20


class TestGradCAM:
    """Grad-CAM attribution generation and normalization tests."""

    def test_gradcam_dimensions_and_range(self):
        model = DummyClassifier()
        gradcam = GradCAM(model, target_layer=model.conv1)
        x = torch.randn(1, 3, 64, 64)
        cam, pred_cls, conf = gradcam.generate_cam(x)

        assert cam.shape == (64, 64)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
        assert 0 <= pred_cls < 6
        assert 0.0 <= conf <= 1.0

    def test_gradcam_multitask_support(self):
        model = DummyMultiTask()
        gradcam = GradCAM(model)
        x = torch.randn(1, 3, 64, 64)
        cam, pred_cls, conf = gradcam.generate_cam(x, target_type="classification")
        assert cam.shape == (64, 64)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_constant_attribution_handling(self):
        # A zero/constant activation model should produce all zeros gracefully without NaN
        model = DummyClassifier()
        with torch.no_grad():
            model.conv1.weight.fill_(0.0)
            model.conv1.bias.fill_(0.0)

        gradcam = GradCAM(model, target_layer=model.conv1)
        x = torch.randn(1, 3, 32, 32)
        cam, _, _ = gradcam.generate_cam(x)
        assert np.all(cam == 0.0)
        assert not np.isnan(cam).any()


class TestAttributionMetrics:
    """Quantitative XAI metric tests."""

    def test_energy_inside_outside_complementarity(self):
        # Perfect synthetic attribution map and mask
        h, w = 50, 50
        mask = np.zeros((h, w), dtype=np.float32)
        mask[10:30, 10:30] = 1.0

        attr = np.random.uniform(0.0, 1.0, (h, w)).astype(np.float32)

        e_in = compute_energy_inside_mask(attr, mask)
        e_out = compute_energy_outside_mask(attr, mask)

        assert 0.0 <= e_in <= 1.0
        assert 0.0 <= e_out <= 1.0
        assert np.isclose(e_in + e_out, 1.0, atol=1e-5)

    def test_attribution_iou(self):
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.float32)
        mask[0:20, 0:100] = 1.0  # Top 20% area = 2000 pixels

        attr = np.zeros((h, w), dtype=np.float32)
        attr[0:20, 0:100] = 1.0  # Identical top 20% attribution

        iou = compute_attribution_iou(attr, mask, top_percent=20.0)
        assert np.isclose(iou, 1.0, atol=1e-3)

    def test_pointing_game_hit_and_miss(self):
        h, w = 50, 50
        mask = np.zeros((h, w), dtype=np.float32)
        mask[10:20, 10:20] = 1.0

        # Hit
        attr_hit = np.zeros((h, w), dtype=np.float32)
        attr_hit[15, 15] = 1.0
        assert compute_pointing_game(attr_hit, mask) == 1

        # Miss
        attr_miss = np.zeros((h, w), dtype=np.float32)
        attr_miss[45, 45] = 1.0
        assert compute_pointing_game(attr_miss, mask) == 0

    def test_healthy_xai_metrics(self):
        h, w = 50, 50
        attr = np.ones((h, w), dtype=np.float32)
        metrics = compute_healthy_xai_metrics(attr, border_ratio=0.10)
        assert "attribution_concentration" in metrics
        assert "border_attention_ratio" in metrics
        assert "normalized_entropy" in metrics
        assert 0.0 <= metrics["border_attention_ratio"] <= 1.0
        assert 0.0 <= metrics["normalized_entropy"] <= 1.0


class TestFaithfulness:
    """Perturbation faithfulness tests."""

    def test_auc_integration(self):
        curve = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        auc = integrate_auc(curve, fractions)
        assert np.isclose(auc, 0.5, atol=1e-3)

    def test_sample_faithfulness_execution(self):
        model = DummyClassifier()
        img = torch.randn(3, 32, 32)
        attr = np.random.uniform(0, 1, (32, 32)).astype(np.float32)
        res = evaluate_sample_faithfulness(model, img, attr, target_class=0, num_steps=5)

        assert "deletion_auc" in res
        assert "insertion_auc" in res
        assert len(res["deletion_curve"]) == 6
        assert len(res["insertion_curve"]) == 6


class TestStatisticalTests:
    """Paired statistical testing."""

    def test_wilcoxon_paired(self):
        a = [0.2, 0.3, 0.25, 0.4, 0.35]
        b = [0.4, 0.5, 0.45, 0.6, 0.55]
        res = run_paired_wilcoxon_test(a, b)
        assert res["mean_diff"] > 0
        assert res["p_value"] <= 0.10

    def test_mcnemar_contingency(self):
        table = np.array([[50, 30], [5, 65]])  # Discordant: 30 vs 5
        stat, pval = run_mcnemar_test(table)
        assert stat > 0
        assert pval < 0.05
