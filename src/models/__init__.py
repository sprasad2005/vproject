"""Model architecture implementations, baselines, and model factory for RiceGuard."""

from __future__ import annotations

from src.models.efficientnet import RiceEfficientNet
from src.models.model_factory import (
    benchmark_inference_latency,
    build_model,
    count_parameters,
    get_model_size_mb,
)
from src.models.multitask import RiceMultiTaskEfficientNet
from src.models.resnet import RiceResNet
from src.models.vit import RiceViT

__all__ = [
    "RiceResNet",
    "RiceEfficientNet",
    "RiceMultiTaskEfficientNet",
    "RiceViT",
    "build_model",
    "count_parameters",
    "get_model_size_mb",
    "benchmark_inference_latency",
]
