"""Comprehensive Unit Tests for Phase 3: Lesion-Aware Multi-Task Model and Localization."""

import pytest
import torch

from src.data.annotations import BoundingBox
from src.data.dataset_registry import validate_dataset_usage
from src.data.grid_assignment import (
    create_grid_target,
    decode_grid_predictions,
)
from src.evaluation.localization_metrics import (
    compute_box_iou,
    evaluate_dataset_localization,
)
from src.models.model_factory import build_model, count_parameters
from src.models.multitask import RiceMultiTaskEfficientNet
from src.training.multitask_loss import MultiTaskLoss


class TestPhase3MultiTaskModel:
    """Test multi-task architecture structure and forward pass."""

    def test_model_instantiation(self):
        model = build_model("efficientnet_b0_multitask", num_classes=6, pretrained=False)
        assert isinstance(model, RiceMultiTaskEfficientNet)
        assert model.num_classes == 6
        assert model.grid_size == 7

    def test_forward_pass_shapes(self):
        model = RiceMultiTaskEfficientNet(num_classes=6, pretrained=False, grid_size=7)
        model.eval()
        dummy_input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            cls_logits, loc_output = model(dummy_input)

        assert cls_logits.shape == (4, 6)
        assert loc_output.shape == (4, 5, 7, 7)

    def test_parameter_counts(self):
        model = RiceMultiTaskEfficientNet(num_classes=6, pretrained=False)
        params = count_parameters(model)
        assert params["total_parameters"] > 4_000_000
        assert params["trainable_parameters"] == params["total_parameters"]


class TestGridAssignment:
    """Test bounding box to 7x7 grid conversion and decoding."""

    def test_single_box_grid_target(self):
        # Box centered at (0.5, 0.5) with width=0.2, height=0.2
        # On a 7x7 grid: col = floor(0.5 * 7) = 3, row = floor(0.5 * 7) = 3
        # dx = 0.5 * 7 - 3 = 0.5, dy = 0.5 * 7 - 3 = 0.5
        box = BoundingBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        target, stats = create_grid_target([box], grid_size=7)

        assert target.shape == (5, 7, 7)
        assert stats["total_boxes"] == 1
        assert stats["assigned_boxes"] == 1
        assert stats["collisions"] == 0

        # Verify assigned cell (row 3, col 3)
        assert target[0, 3, 3].item() == 1.0
        assert pytest.approx(target[1, 3, 3].item(), 1e-4) == 0.5
        assert pytest.approx(target[2, 3, 3].item(), 1e-4) == 0.5
        assert pytest.approx(target[3, 3, 3].item(), 1e-4) == 0.2
        assert pytest.approx(target[4, 3, 3].item(), 1e-4) == 0.2

        # All other cells must be zero
        assert target[0].sum().item() == 1.0

    def test_healthy_empty_grid_target(self):
        target, stats = create_grid_target([], grid_size=7)
        assert target.shape == (5, 7, 7)
        assert stats["total_boxes"] == 0
        assert stats["assigned_boxes"] == 0
        assert stats["collisions"] == 0
        assert torch.all(target == 0.0)

    def test_collision_resolution_largest_area(self):
        # Two boxes landing in same cell (row 2, col 2)
        # Cell 2 covers [2/7, 3/7) = [0.2857, 0.4285]
        box_small = BoundingBox(class_id=1, x_center=0.35, y_center=0.35, width=0.05, height=0.05)
        box_large = BoundingBox(class_id=2, x_center=0.36, y_center=0.36, width=0.15, height=0.15)

        # Small box first, large box second
        target1, stats1 = create_grid_target([box_small, box_large], grid_size=7)
        assert stats1["collisions"] == 1
        assert pytest.approx(target1[3, 2, 2].item(), 1e-4) == 0.15

        # Large box first, small box second -> must produce identical winner
        target2, stats2 = create_grid_target([box_large, box_small], grid_size=7)
        assert stats2["collisions"] == 1
        assert pytest.approx(target2[3, 2, 2].item(), 1e-4) == 0.15

    def test_decode_grid_predictions(self):
        loc_tensor = torch.full((5, 7, 7), -10.0)
        # Set large logit for objectness at row 1, col 1
        loc_tensor[0, 1, 1] = 10.0  # sigmoid(10) ≈ 1.0
        loc_tensor[1, 1, 1] = 0.0   # sigmoid(0) = 0.5 (dx)
        loc_tensor[2, 1, 1] = 0.0   # sigmoid(0) = 0.5 (dy)
        loc_tensor[3, 1, 1] = 0.0   # sigmoid(0) = 0.5 (w)
        loc_tensor[4, 1, 1] = 0.0   # sigmoid(0) = 0.5 (h)

        boxes = decode_grid_predictions(loc_tensor, conf_threshold=0.5, grid_size=7)
        assert len(boxes) == 1
        box = boxes[0]
        assert box["confidence"] > 0.99
        # xc = (1 + 0.5) / 7 = 1.5 / 7
        assert pytest.approx(box["x_center"], 1e-4) == 1.5 / 7.0
        assert pytest.approx(box["y_center"], 1e-4) == 1.5 / 7.0
        assert pytest.approx(box["width"], 1e-4) == 0.5
        assert pytest.approx(box["height"], 1e-4) == 0.5


class TestMultiTaskLoss:
    """Test MultiTaskLoss composite calculation, gradient flow, and edge cases."""

    def test_loss_computation_and_backward(self):
        loss_fn = MultiTaskLoss(lambda_loc=0.25, lambda_box=1.0, pos_weight=15.0)

        cls_logits = torch.randn(4, 6, requires_grad=True)
        loc_output = torch.randn(4, 5, 7, 7, requires_grad=True)
        class_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        # 2 positive cells in batch
        grid_targets = torch.zeros(4, 5, 7, 7)
        grid_targets[0, 0, 2, 2] = 1.0
        grid_targets[0, 1:5, 2, 2] = 0.5
        grid_targets[1, 0, 4, 4] = 1.0
        grid_targets[1, 1:5, 4, 4] = 0.3

        total_loss, loss_dict = loss_fn(cls_logits, loc_output, class_targets, grid_targets)

        assert total_loss.item() > 0.0
        assert "loss_cls" in loss_dict
        assert "loss_loc" in loss_dict
        assert "loss_obj" in loss_dict
        assert "loss_box" in loss_dict

        # Verify gradient flow
        total_loss.backward()
        assert cls_logits.grad is not None
        assert loc_output.grad is not None

    def test_loss_with_healthy_only_batch(self):
        loss_fn = MultiTaskLoss(lambda_loc=0.25, lambda_box=1.0, pos_weight=15.0)

        cls_logits = torch.randn(2, 6, requires_grad=True)
        loc_output = torch.randn(2, 5, 7, 7, requires_grad=True)
        class_targets = torch.tensor([0, 0], dtype=torch.long)
        grid_targets = torch.zeros(2, 5, 7, 7)  # All healthy

        total_loss, loss_dict = loss_fn(cls_logits, loc_output, class_targets, grid_targets)

        assert not torch.isnan(total_loss)
        assert loss_dict["loss_box"] == 0.0
        total_loss.backward()
        assert loc_output.grad is not None


class TestLocalizationMetrics:
    """Test IoU and localization evaluation metrics."""

    def test_compute_box_iou(self):
        b1 = {"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}
        # Identical box -> IoU = 1.0
        assert pytest.approx(compute_box_iou(b1, b1), 1e-4) == 1.0

        # Non-overlapping box -> IoU = 0.0
        b2 = {"x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}
        assert compute_box_iou(b1, b2) == 0.0

    def test_dataset_localization_evaluation(self):
        pred_boxes = [[{"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2, "confidence": 0.9}]]
        gt_boxes = [[{"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}]]
        classes = ["Brown spot"]

        res = evaluate_dataset_localization(pred_boxes, gt_boxes, classes, iou_threshold=0.5)
        assert res["true_positives"] == 1
        assert res["false_positives"] == 0
        assert res["false_negatives"] == 0
        assert res["localization_precision"] == 1.0
        assert res["localization_recall"] == 1.0
        assert res["localization_f1"] == 1.0


class TestDataGovernance:
    """Ensure data governance remains strictly enforced in Phase 3."""

    def test_external_datasets_blocked(self):
        with pytest.raises(PermissionError):
            validate_dataset_usage("sethy", "train")

        with pytest.raises(PermissionError):
            validate_dataset_usage("bd5", "train")

        with pytest.raises(PermissionError):
            validate_dataset_usage("riceseg", "train")
