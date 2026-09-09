"""Evaluation metrics computation: Accuracy, Macro-F1, Per-Class F1, Confusion Matrix."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: Ground truth class labels (N,).
        y_pred: Predicted class labels (N,).
        class_names: List of class label strings.

    Returns:
        Dict[str, Any]: Macro-F1, accuracy, per-class metrics.
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    accuracy = float(np.mean(y_true == y_pred))

    classes = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    f1_scores = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        c_name = class_names[c] if (class_names and c < len(class_names)) else str(c)
        per_class[c_name] = {"precision": prec, "recall": rec, "f1": f1, "support": int(tp + fn)}
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
