"""Profile baseline models: parameter count, size, and CPU inference latency."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.models.model_factory import (
    benchmark_inference_latency,
    build_model,
    count_parameters,
    get_model_size_mb,
)
from src.utils.paths import ensure_dir

matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    models = ["efficientnet_b0", "resnet50", "vit_b_16"]
    display_names = {
        "efficientnet_b0": "EfficientNet-B0",
        "resnet50": "ResNet50",
        "vit_b_16": "Vision Transformer",
    }

    records = []
    print("=" * 80)
    print(" Profiling Baseline Architectures (CPU Environment)")
    print("=" * 80)

    for m in models:
        net = build_model(m, num_classes=6, pretrained=True)
        p_info = count_parameters(net)
        sz = get_model_size_mb(net)
        lat = benchmark_inference_latency(
            net,
            input_size=(1, 3, 224, 224),
            num_iterations=20,
            warmup_iterations=5,
            device="cpu",
        )
        rec = {
            "Model": display_names[m],
            "model_key": m,
            "Parameters": p_info["total_parameters"],
            "Trainable Parameters": p_info["trainable_parameters"],
            "Checkpoint Size": f"{sz:.1f} MB",
            "size_mb": sz,
            "CPU Latency": f"{lat['mean_latency_ms']:.1f} ms",
            "cpu_latency_num": lat["mean_latency_ms"],
        }
        records.append(rec)
        print(f"[{display_names[m]}] Params: {p_info['total_parameters']:,} | Size: {sz:.1f} MB | Latency: {lat['mean_latency_ms']:.1f} ms")

    figures_dir = ensure_dir(PROJECT_ROOT / "results" / "figures" / "phase2")
    df = pd.DataFrame(records)

    colors = ["#2b5c8f", "#2ca02c", "#d96b27"]
    models_list = df["Model"].tolist()

    # 1. Parameter count plot
    plt.figure(figsize=(8, 5))
    params_m = [p / 1e6 for p in df["Parameters"]]
    bars = plt.bar(models_list, params_m, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Baseline Comparison — Parameter Count (Millions)", fontsize=13, fontweight="bold")
    plt.ylabel("Parameters (M)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(params_m) * 0.02), f"{yval:.2f}M", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "parameters_comparison.png", dpi=150)
    plt.close()

    # 2. CPU Latency plot
    plt.figure(figsize=(8, 5))
    latencies = df["cpu_latency_num"].tolist()
    bars = plt.bar(models_list, latencies, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Baseline Comparison — CPU Inference Latency (ms)", fontsize=13, fontweight="bold")
    plt.ylabel("Latency per Sample (ms)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(latencies) * 0.02), f"{yval:.1f} ms", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_comparison.png", dpi=150)
    plt.close()

    reports_dir = ensure_dir(PROJECT_ROOT / "results" / "reports")
    df.to_csv(reports_dir / "phase2_model_profiling.csv", index=False)
    print(f"[*] Profiling reports and figures saved to {reports_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
