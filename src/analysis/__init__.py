"""Analysis and consolidation package for RiceGuard Phase 5."""

from src.analysis.effect_size import (
    compute_absolute_difference,
    compute_cohens_d_paired,
    compute_rank_biserial_correlation,
    compute_relative_change_percent,
    interpret_cohens_d,
)
from src.analysis.failure_analysis import FailureAnalyzer

__all__ = [
    "compute_absolute_difference",
    "compute_relative_change_percent",
    "compute_cohens_d_paired",
    "compute_rank_biserial_correlation",
    "interpret_cohens_d",
    "FailureAnalyzer",
]
