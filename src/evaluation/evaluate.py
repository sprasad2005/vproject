"""Evaluation engine, benchmark test runner, and metric reporting for RiceGuard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES
from src.models.model_factory import (
    benchmark_inference_latency,
    build_model,
    count_parameters,
    get_model_size_mb,
)
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir, get_project_root

matplotlib.use("Agg")


class ModelEvaluator:
    """Evaluates trained checkpoints on internal test sets with full diagnostic reporting."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_name: str = "model",
        device: Optional[Union[str, torch.device]] = None,
        num_classes: int = 6,
    ) -> None:
        self.root = get_project_root()
        self.model_name = model_name
        self.logger = get_logger(f"evaluator.{model_name}")

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            chk_p = Path(checkpoint_path)
            if not chk_p.is_absolute():
                chk_p = self.root / chk_p
            if not chk_p.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {chk_p}")

            chk = torch.load(chk_p, map_location=self.device, weights_only=False)
            m_name = chk.get("model_name", model_name)
            dropout_rate = chk.get("config", {}).get("model", {}).get("dropout_rate", 0.0)
            self.model = build_model(m_name, num_classes=num_classes, pretrained=False, dropout_rate=dropout_rate)
            self.model.load_state_dict(chk["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model_name = m_name
            self.logger.info(f"Loaded checkpoint from {chk_p} (Epoch {chk.get('epoch', '?')})")
        else:
            raise ValueError("Either model or checkpoint_path must be provided.")

        self.model.eval()

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run complete evaluation over the dataset."""
        self.model.eval()
        all_targets: List[int] = []
        all_preds: List[int] = []
        all_probs: List[List[float]] = []
        all_metadata: List[Dict[str, Any]] = []

        for images, targets, metadata in data_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_metadata.extend(metadata)

        metrics = compute_classification_metrics(all_targets, all_preds, CANONICAL_PRIMARY_CLASSES)

        # Measure efficiency profile
        param_counts = count_parameters(self.model)
        size_mb = get_model_size_mb(self.model)
        latency_info = benchmark_inference_latency(self.model, device="cpu")

        results = {
            "model_name": self.model_name,
            "overall_metrics": {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
            },
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
            "efficiency": {
                "total_parameters": param_counts["total_parameters"],
                "trainable_parameters": param_counts["trainable_parameters"],
                "model_size_mb": size_mb,
                "cpu_latency_mean_ms": latency_info["mean_latency_ms"],
                "cpu_latency_median_ms": latency_info["median_latency_ms"],
                "cpu_latency_p95_ms": latency_info["p95_latency_ms"],
                "throughput_fps": latency_info["throughput_fps"],
            },
            "sample_count": len(all_targets),
        }

        return results

    def save_reports(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Save classification reports, per-class metrics CSV, summary JSON, and confusion matrix plot."""
        out_dir = Path(output_dir) if output_dir else self.root / "results" / "metrics"
        ensure_dir(out_dir)

        fig_dir = ensure_dir(self.root / "results" / "figures" / "phase2")

        # 1. Summary JSON
        summary_path = out_dir / f"{self.model_name}_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. Per-class metrics CSV
        per_class_rows = []
        for cls_name, vals in results["per_class"].items():
            per_class_rows.append(
                {
                    "class": cls_name,
                    "precision": vals["precision"],
                    "recall": vals["recall"],
                    "f1_score": vals["f1"],
                    "support": vals["support"],
                }
            )
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_df.to_csv(out_dir / f"{self.model_name}_per_class_metrics.csv", index=False)

        # 3. Classification Report CSV
        cr_df = pd.DataFrame(results["classification_report"]).transpose()
        cr_df.to_csv(out_dir / f"{self.model_name}_classification_report.csv")

        # 4. Confusion Matrix Heatmap (Pure Matplotlib)
        cm = np.array(results["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Show all ticks and label them with class names
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=CANONICAL_PRIMARY_CLASSES,
            yticklabels=CANONICAL_PRIMARY_CLASSES,
            title=f"{self.model_name} — Internal Test Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )

        fig.tight_layout()
        plt.savefig(fig_dir / f"{self.model_name}_confusion_matrix.png", dpi=150)
        plt.close()

        self.logger.info(f"Evaluation reports successfully written for {self.model_name} under {out_dir}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Functional evaluation interface."""
    evaluator = ModelEvaluator(model=model)
    return evaluator.evaluate(dataloader)
