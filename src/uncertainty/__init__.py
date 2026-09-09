"""Uncertainty quantification module for RiceGuard."""

from src.uncertainty.entropy import compute_predictive_entropy
from src.uncertainty.mc_dropout import estimate_mc_dropout_uncertainty
from src.uncertainty.ood import detect_ood
from src.uncertainty.selective_prediction import SelectivePredictionPolicy

__all__ = [
    "compute_predictive_entropy",
    "estimate_mc_dropout_uncertainty",
    "detect_ood",
    "SelectivePredictionPolicy",
]
