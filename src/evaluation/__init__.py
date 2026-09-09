"""Evaluation engines, metrics, and cross-dataset testing harnesses for RiceGuard."""

from __future__ import annotations

from src.evaluation.evaluate import ModelEvaluator, evaluate_model

__all__ = [
    "ModelEvaluator",
    "evaluate_model",
]
