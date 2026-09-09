"""Out-of-Distribution (OOD) detection using energy scores and Mahalanobis distance."""

from __future__ import annotations

from typing import Any, Dict


def detect_ood(
    features: Any,
    method: str = "energy_score",
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Calculate OOD detection scores and separation AUROC."""
    raise NotImplementedError("OOD detection will be implemented in future phase.")
