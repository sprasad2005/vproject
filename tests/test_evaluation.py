"""Tests for ModelEvaluator metrics calculation and report generation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluate import ModelEvaluator
from src.models.model_factory import build_model
from src.training.metrics import compute_classification_metrics


class MockEvalDataset(Dataset):
    def __init__(self, size: int = 12):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        img = torch.randn(3, 224, 224)
        target = idx % 6
        metadata = {"image_id": f"test_{idx}", "canonical_class": f"Class_{target}"}
        return img, target, metadata


def test_compute_classification_metrics():
    """Verify compute_classification_metrics returns accurate scores."""
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_model_evaluator_execution(tmp_path):
    """Verify ModelEvaluator evaluates dataset and saves reports."""
    ds = MockEvalDataset(size=12)

    def collate(batch):
        return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), [b[2] for b in batch]

    loader = DataLoader(ds, batch_size=4, collate_fn=collate)
    model = build_model("efficientnet_b0", num_classes=6, pretrained=False)

    evaluator = ModelEvaluator(model=model, model_name="test_eval", device="cpu")
    results = evaluator.evaluate(loader)

    assert "overall_metrics" in results
    assert "accuracy" in results["overall_metrics"]
    assert "efficiency" in results
    assert results["sample_count"] == 12

    evaluator.save_reports(results, output_dir=tmp_path / "metrics")
    assert (tmp_path / "metrics" / "test_eval_evaluation_summary.json").exists()
    assert (tmp_path / "metrics" / "test_eval_per_class_metrics.csv").exists()
    assert (tmp_path / "metrics" / "test_eval_classification_report.csv").exists()
