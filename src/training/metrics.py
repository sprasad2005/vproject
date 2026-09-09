"""Metric calculation and evaluation utilities for RiceGuard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List[int]],
    y_pred: Union[np.ndarray, torch.Tensor, List[int]],
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive macro and per-class classification metrics.

    Args:
        y_true: Ground truth integer class indices.
        y_pred: Predicted integer class indices.
        target_names: Canonical class names.

    Returns:
        Dict[str, Any]: Overall and per-class evaluation metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = target_names or CANONICAL_PRIMARY_CLASSES
    labels = list(range(len(classes)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_dict = {}
    for idx, cname in enumerate(classes):
        per_class_dict[cname] = {
            "precision": round(float(per_class_precision[idx]), 4),
            "recall": round(float(per_class_recall[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(np.sum(y_true == idx)),
        }

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
