"""Pydantic Request and Response Schemas for RiceGuard API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status ('healthy')")
    device: str = Field(..., description="Device used for inference ('cuda:0' or 'cpu')")
    model: str = Field(..., description="Active model name")


class PredictionDetail(BaseModel):
    """Predicted top disease class details."""

    class_id: int = Field(..., description="Canonical class index (0..5)")
    class_name: str = Field(..., description="Canonical disease name")
    confidence: float = Field(..., description="Prediction probability (0.0 to 1.0)")


class TopPrediction(BaseModel):
    """Class probability entry."""

    class_name: str = Field(..., description="Disease class name")
    probability: float = Field(..., description="Softmax probability score")


class LesionBox(BaseModel):
    """Predicted lesion bounding box in pixel coordinates."""

    x: int = Field(..., description="Top-left x pixel coordinate")
    y: int = Field(..., description="Top-left y pixel coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")
    confidence: float = Field(..., description="Objectness confidence score")


class Visualizations(BaseModel):
    """Base64 Data URI strings for visual results."""

    original: str = Field(..., description="Original uploaded image")
    localization: str = Field(..., description="Image with predicted lesion bounding boxes")
    gradcam: str = Field(..., description="Grad-CAM explainability heatmap overlay")
    combined: str = Field(..., description="Combined multi-modal visual explanation")


class ModelInfo(BaseModel):
    """Inference model metadata."""

    name: str = Field(..., description="Model identifier")
    architecture: str = Field(..., description="Model backbone architecture")
    device: str = Field(..., description="Execution compute device")


class PredictionResponse(BaseModel):
    """Complete structured prediction response."""

    prediction: PredictionDetail
    top_predictions: List[TopPrediction]
    lesions: List[LesionBox]
    lesion_count: int
    visualizations: Visualizations
    model: ModelInfo
