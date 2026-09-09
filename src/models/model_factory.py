"""Unified Model Factory and Profiling Engine for RiceGuard Baselines."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.efficientnet import RiceEfficientNet
from src.models.multitask import RiceMultiTaskEfficientNet
from src.models.resnet import RiceResNet
from src.models.vit import RiceViT


def build_model(
    model_name: str,
    num_classes: int = 6,
    pretrained: bool = True,
    dropout_rate: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate and configure baseline deep learning models for RiceGuard.

    Supported model names:
        - 'resnet50', 'resnet'
        - 'efficientnet_b0', 'efficientnet'
        - 'vit_b_16', 'vit'

    Args:
        model_name: Name of the neural network architecture.
        num_classes: Number of canonical disease output classes (default: 6).
        pretrained: Whether to initialize with ImageNet weights.
        dropout_rate: Dropout probability for classification head.

    Returns:
        nn.Module: Configured PyTorch model.
    """
    name = model_name.lower().strip().replace("-", "_")

    if name in ["resnet50", "resnet", "resnet_50"]:
        return RiceResNet(
            architecture="resnet50",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    elif name in ["resnet18", "resnet_18"]:
        return RiceResNet(
            architecture="resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    elif name in ["efficientnet_b0", "efficientnet", "efficientnetb0"]:
        return RiceEfficientNet(
            architecture="efficientnet_b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate if dropout_rate > 0.0 else None,
            **kwargs,
        )
    elif name in [
        "efficientnet_b0_multitask",
        "multitask_efficientnet_b0",
        "multitask_efficientnet",
        "rice_multitask_efficientnet",
    ]:
        return RiceMultiTaskEfficientNet(
            architecture="efficientnet_b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate if dropout_rate > 0.0 else 0.2,
            **kwargs,
        )
    elif name in ["vit_b_16", "vit", "vision_transformer", "vit_base"]:
        return RiceViT(
            architecture="vit_b_16",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Available baselines: ['resnet50', 'efficientnet_b0', 'vit_b_16']"
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate the memory footprint of the model parameters in Megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    return round(size_mb, 2)


def benchmark_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """Profile CPU/GPU inference latency under standardized benchmark conditions."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(*input_size, device=dev)

    # Warm-up cycles
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

    latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(num_iterations):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            start = time.perf_counter()
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "mean_latency_ms": round(float(np.mean(arr)), 2),
        "median_latency_ms": round(float(np.median(arr)), 2),
        "std_latency_ms": round(float(np.std(arr)), 2),
        "min_latency_ms": round(float(np.min(arr)), 2),
        "max_latency_ms": round(float(np.max(arr)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(arr)), 2),
        "device": str(dev),
    }
