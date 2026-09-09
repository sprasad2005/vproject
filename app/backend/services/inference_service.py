"""Inference Engine Service for RiceGuard Backend."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from app.backend.config import settings
from app.backend.schemas import (
    ModelInfo,
    PredictionDetail,
    PredictionResponse,
    TopPrediction,
)
from app.backend.services.localization_service import LocalizationService
from app.backend.services.xai_service import XAIService
from app.backend.utils.image_processing import (
    preprocess_image,
    validate_and_load_image,
)
from app.backend.utils.visualization import build_visualizations
from src.models.multitask import RiceMultiTaskEfficientNet

logger = logging.getLogger("riceguard.inference")


class RiceGuardInferenceEngine:
    """Singleton Inference Engine loading Phase 3B model once at startup."""

    def __init__(self) -> None:
        self.device = settings.torch_device
        self.device_str = settings.device
        logger.info("Initializing RiceGuard Inference Engine on %s", self.device_str)

        # 1. Instantiate Phase 3B Model
        self.model = RiceMultiTaskEfficientNet(
            architecture="efficientnet_b0",
            num_classes=6,
            pretrained=False,
        )

        # 2. Load verified Phase 3B Checkpoint
        if not settings.phase3b_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Required Phase 3B checkpoint not found at: {settings.phase3b_checkpoint_path}"
            )

        checkpoint = torch.load(
            settings.phase3b_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Phase 3B model loaded successfully from %s", settings.phase3b_checkpoint_path)

        # 3. Initialize Services with frozen Phase 3B settings
        self.localization_service = LocalizationService(
            confidence_threshold=settings.confidence_threshold,
            top_k=settings.top_k,
        )
        self.xai_service = XAIService(model=self.model)

        logger.info(
            "Localization service configured with frozen thresholds: conf=%.2f, top_k=%d",
            settings.confidence_threshold,
            settings.top_k,
        )

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Execute complete multi-task prediction, localization, and XAI pipeline on image bytes."""
        # 1. Validate & load image
        orig_img, orig_w, orig_h = validate_and_load_image(image_bytes)

        # 2. Preprocess
        input_tensor = preprocess_image(orig_img).to(self.device)

        # 3. Model forward pass
        with torch.no_grad():
            cls_logits, loc_output = self.model(input_tensor)
            probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]

        # 4. Classification outcome
        pred_class_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_class_idx])
        pred_class_name = settings.CANONICAL_CLASSES[pred_class_idx]

        # Top-K predictions list (all 6 classes ranked descending)
        ranked_indices = np.argsort(probs)[::-1]
        top_preds: List[TopPrediction] = [
            TopPrediction(
                class_name=settings.CANONICAL_CLASSES[idx],
                probability=round(float(probs[idx]), 4),
            )
            for idx in ranked_indices
        ]

        # 5. Localization Decoding (with frozen Phase 3B config)
        lesions = self.localization_service.decode_lesions(
            loc_output=loc_output,
            orig_width=orig_w,
            orig_height=orig_h,
        )

        # 6. Grad-CAM Explainability Generation
        heatmap = self.xai_service.generate_heatmap(
            input_tensor=input_tensor,
            target_class=pred_class_idx,
            orig_width=orig_w,
            orig_height=orig_h,
        )

        # 7. Render Base64 Visualizations
        visualizations = build_visualizations(
            orig_img=orig_img,
            heatmap=heatmap,
            lesions=lesions,
        )

        return PredictionResponse(
            prediction=PredictionDetail(
                class_id=pred_class_idx,
                class_name=pred_class_name,
                confidence=round(pred_conf, 4),
            ),
            top_predictions=top_preds,
            lesions=lesions,
            lesion_count=len(lesions),
            visualizations=visualizations,
            model=ModelInfo(
                name=settings.MODEL_NAME,
                architecture=settings.MODEL_ARCHITECTURE,
                device=self.device_str,
            ),
        )


# Global engine placeholder
engine: RiceGuardInferenceEngine | None = None


def get_inference_engine() -> RiceGuardInferenceEngine:
    """Get initialized inference engine singleton."""
    global engine
    if engine is None:
        engine = RiceGuardInferenceEngine()
    return engine
