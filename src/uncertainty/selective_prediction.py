"""ACCEPT / ABSTAIN Selective Prediction Policy for safe clinical decision making."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class SelectivePredictionPolicy:
    """Operational decision layer: ACCEPT when confident, ABSTAIN when uncertain."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.75,
        max_uncertainty_threshold: float = 0.35,
        abstain_label: str = "ABSTAIN_HUMAN_REVIEW",
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.max_uncertainty_threshold = max_uncertainty_threshold
        self.abstain_label = abstain_label

    def decide(
        self,
        prediction_class: str,
        calibrated_confidence: float,
        uncertainty_score: float,
    ) -> Dict[str, Any]:
        """Evaluate if prediction should be accepted or abstained."""
        is_confident = calibrated_confidence >= self.min_confidence_threshold
        is_low_uncertainty = uncertainty_score <= self.max_uncertainty_threshold

        if is_confident and is_low_uncertainty:
            decision = "ACCEPT"
            final_class = prediction_class
        else:
            decision = "ABSTAIN"
            final_class = self.abstain_label

        return {
            "decision": decision,
            "final_class": final_class,
            "raw_class": prediction_class,
            "calibrated_confidence": calibrated_confidence,
            "uncertainty_score": uncertainty_score,
            "passed_confidence": is_confident,
            "passed_uncertainty": is_low_uncertainty,
        }

    def evaluate_risk_coverage(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute coverage and selective risk curves across thresholds."""
        # Placeholder interface for Phase 0
        return {
            "coverage": 1.0,
            "selective_risk": 0.0,
        }
