"""Tests for baseline model factory, architectures, and output dimensions."""

from __future__ import annotations

import pytest
import torch

from src.models.model_factory import (
    benchmark_inference_latency,
    build_model,
    count_parameters,
    get_model_size_mb,
)


@pytest.mark.parametrize("model_name", ["resnet50", "efficientnet_b0", "vit_b_16"])
def test_build_model_output_dimensions(model_name: str):
    """Verify all supported baseline models output exactly 6 canonical disease classes."""
    model = build_model(model_name=model_name, num_classes=6, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 6), f"Expected shape (2, 6), got {output.shape} for {model_name}"


def test_count_parameters():
    """Verify parameter counting on ResNet50."""
    model = build_model("resnet50", num_classes=6, pretrained=False)
    params = count_parameters(model)
    assert "total_parameters" in params
    assert "trainable_parameters" in params
    assert params["total_parameters"] > 20_000_000


def test_get_model_size_mb():
    """Verify model size calculation in MB."""
    model = build_model("efficientnet_b0", num_classes=6, pretrained=False)
    size_mb = get_model_size_mb(model)
    assert size_mb > 0.0
    assert size_mb < 50.0  # EfficientNet-B0 is ~15-20 MB


def test_invalid_model_name():
    """Verify invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        build_model("unsupported_supernet_xyz", num_classes=6)


def test_latency_benchmark():
    """Verify CPU latency profiling utility."""
    model = build_model("efficientnet_b0", num_classes=6, pretrained=False)
    metrics = benchmark_inference_latency(model, num_warmup=2, num_iterations=5, device="cpu")
    assert "mean_latency_ms" in metrics
    assert "throughput_fps" in metrics
    assert metrics["mean_latency_ms"] > 0
