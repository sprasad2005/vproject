"""Application Configuration for RiceGuard Localhost Backend.

Discovers immutable Phase 3B model checkpoints, frozen decoding configurations,
and configures localhost-only server settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def get_project_root() -> Path:
    """Resolve project root directory."""
    return Path(__file__).resolve().parent.parent.parent


class Settings:
    """RiceGuard Application Settings."""

    PROJECT_NAME: str = "RiceGuard"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # CORS origins strictly for local development
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Canonical Classes (Order 0..5)
    CANONICAL_CLASSES: List[str] = [
        "Healthy",
        "Blast",
        "Brown Spot",
        "Leaf Smut",
        "Tungro",
        "Sheath Blight",
    ]

    # Model metadata
    MODEL_NAME: str = "RiceGuard Phase 3B"
    MODEL_ARCHITECTURE: str = "EfficientNet-B0 Multi-Task"

    def __init__(self) -> None:
        self.root: Path = get_project_root()

        # Checkpoints
        self.phase3b_checkpoint_path: Path = (
            self.root / "experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/checkpoints/best_model.pt"
        )
        self.phase3b_decoding_config_path: Path = (
            self.root / "experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/calibration/selected_decoding_config.json"
        )
        self.phase2b_checkpoint_path: Path = (
            self.root / "experiments/phase2b_final_baseline/efficientnet_b0/checkpoints/best_model.pt"
        )

        # Device detection
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_device: torch.device = torch.device(self.device)

        # Load frozen decoding configuration
        self.decoding_config: Dict[str, Any] = self._load_frozen_decoding_config()
        self.confidence_threshold: float = float(self.decoding_config.get("confidence_threshold", 0.60))
        self.top_k: int = int(self.decoding_config.get("top_k", 3))

    def _load_frozen_decoding_config(self) -> Dict[str, Any]:
        """Load frozen Phase 3B decoding configuration from disk."""
        if self.phase3b_decoding_config_path.exists():
            with open(self.phase3b_decoding_config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Fallback to verified Phase 3B values if config is absent
        return {"confidence_threshold": 0.60, "top_k": 3}


settings = Settings()
