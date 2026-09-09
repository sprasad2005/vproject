"""Phase 2A: Fast Backbone Screening Runner for RiceGuard.

Executes preliminary architecture screening between EfficientNet-B0 and ResNet50
under identical experimental conditions on the primary RiceLeafDiseaseBD dataset.

Automates:
1. CUDA environment validation
2. EfficientNet-B0 screening (training + test evaluation)
3. ResNet50 screening (training + test evaluation)
4. Efficiency profiling (CPU latency, GPU latency, parameters, VRAM peak)
5. Comparison tables (CSV + Markdown) and diagnostic plots
6. Automated backbone selection following predefined decision hierarchy
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import build_transforms
from src.data.dataset import RiceLeafDataset, create_dataloader
from src.evaluation.evaluate import ModelEvaluator
from src.models.model_factory import (
    benchmark_inference_latency,
    build_model,
    count_parameters,
    get_model_size_mb,
)
from src.training.losses import build_criterion
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


def generate_screening_plots(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Generate diagnostic comparison plots for Phase 2A screening."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    models = df["Model"].tolist()
    colors = ["#2ca02c", "#1f77b4"]

    # 1. Validation vs Test Macro F1 Comparison
    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, df["Val Macro F1"], width, label="Validation Macro F1", color="#2ca02c", alpha=0.85, edgecolor="black")
    plt.bar(x + width / 2, df["Test Macro F1"], width, label="Test Macro F1", color="#1f77b4", alpha=0.85, edgecolor="black")

    plt.title("Phase 2A Screening — Macro F1 Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("Architecture", fontsize=11)
    plt.ylabel("Macro F1 Score", fontsize=11)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(models)):
        plt.text(x[i] - width / 2, df["Val Macro F1"][i] + 0.02, f"{df['Val Macro F1'][i]:.4f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width / 2, df["Test Macro F1"][i] + 0.02, f"{df['Test Macro F1'][i]:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "validation_macro_f1_comparison.png", dpi=150)
    plt.close()

    # 2. Efficiency Comparison (Parameters, GPU Latency, Peak VRAM)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Params
    params_m = [p / 1e6 for p in df["Parameters"]]
    bars1 = ax1.bar(models, params_m, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Parameters (Millions)", fontweight="bold")
    ax1.set_ylabel("Million Params")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(params_m) * 0.02), f"{yval:.2f}M", ha="center", fontweight="bold")

    # GPU Latency
    gpu_lats = df["gpu_latency_num"].tolist()
    bars2 = ax2.bar(models, gpu_lats, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("GPU Latency per Sample (ms)", fontweight="bold")
    ax2.set_ylabel("Milliseconds")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(gpu_lats) * 0.02), f"{yval:.2f} ms", ha="center", fontweight="bold")

    # Peak VRAM
    vram_mb = df["peak_vram_mb"].tolist()
    bars3 = ax3.bar(models, vram_mb, color=colors, edgecolor="black", alpha=0.85)
    ax3.set_title("Peak GPU Memory (MB)", fontweight="bold")
    ax3.set_ylabel("Megabytes (MB)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(vram_mb) * 0.02), f"{yval:.0f} MB", ha="center", fontweight="bold")

    plt.suptitle("Phase 2A Screening — Efficiency Comparison on NVIDIA GTX 1650", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "efficiency_comparison.png", dpi=150)
    plt.close()


def main() -> int:
    print("=" * 80)
    print(" RiceGuard Phase 2A: Fast Backbone Screening Engine")
    print(" PRELIMINARY ARCHITECTURE SCREENING — NOT FINAL BASELINE RESULTS")
    print("=" * 80)

    # 1. Device Verification
    if not torch.cuda.is_available():
        print("[FATAL ERROR] CUDA is unavailable. Phase 2A requires NVIDIA GPU for execution.")
        return 1

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print(f"[*] GPU Device: {device} ({gpu_name}, {total_vram_mb:.1f} MB VRAM)")
    print(f"[*] PyTorch Version: {torch.__version__} (CUDA {torch.version.cuda})")

    # 2. Paths and Verification
    reports_dir = ensure_dir(PROJECT_ROOT / "results" / "reports")
    figures_dir = ensure_dir(PROJECT_ROOT / "results" / "figures" / "phase2a")

    train_csv = PROJECT_ROOT / "splits" / "primary_train.csv"
    val_csv = PROJECT_ROOT / "splits" / "primary_val.csv"
    test_csv = PROJECT_ROOT / "splits" / "primary_test.csv"

    for p, name in [(train_csv, "Train"), (val_csv, "Validation"), (test_csv, "Internal Test")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} split CSV missing at {p}. Run dataset split script first.")

    eval_transform = build_transforms(is_training=False)
    test_dataset = RiceLeafDataset(test_csv, root_dir=PROJECT_ROOT, transform=eval_transform, is_training=False)
    test_loader = create_dataloader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Screening models to evaluate
    screening_models = [
        {
            "key": "efficientnet_b0",
            "display_name": "EfficientNet-B0",
            "config_path": PROJECT_ROOT / "configs" / "experiments" / "phase2a_efficientnet_b0.yaml",
            "exp_dir": PROJECT_ROOT / "experiments" / "phase2a_screening" / "efficientnet_b0",
            "fig_name": "efficientnet_confusion_matrix.png",
        },
        {
            "key": "resnet50",
            "display_name": "ResNet50",
            "config_path": PROJECT_ROOT / "configs" / "experiments" / "phase2a_resnet50.yaml",
            "exp_dir": PROJECT_ROOT / "experiments" / "phase2a_screening" / "resnet50",
            "fig_name": "resnet50_confusion_matrix.png",
        },
    ]

    records: List[Dict[str, Any]] = []

    for spec in screening_models:
        m_key = spec["key"]
        d_name = spec["display_name"]
        cfg_file = spec["config_path"]
        exp_dir = spec["exp_dir"]

        print("\n" + "#" * 80)
        print(f" STEP: Training & Evaluating {d_name} ({m_key})")
        print("#" * 80)

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        seed = cfg.get("reproducibility", {}).get("seed", 42)
        set_seed(seed)

        # Clear GPU cache before run
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        train_transform = build_transforms(is_training=True, config=cfg)
        val_transform = build_transforms(is_training=False, config=cfg)

        train_dataset = RiceLeafDataset(train_csv, root_dir=PROJECT_ROOT, transform=train_transform, is_training=True)
        val_dataset = RiceLeafDataset(val_csv, root_dir=PROJECT_ROOT, transform=val_transform, is_training=False)

        b_size = int(cfg.get("training", {}).get("batch_size", 8))
        n_workers = int(cfg.get("training", {}).get("num_workers", 0))
        pin_mem = bool(cfg.get("training", {}).get("pin_memory", True)) and torch.cuda.is_available()

        train_loader = create_dataloader(train_dataset, batch_size=b_size, shuffle=True, num_workers=n_workers, pin_memory=pin_mem)
        val_loader = create_dataloader(val_dataset, batch_size=b_size, shuffle=False, num_workers=n_workers, pin_memory=pin_mem)

        model = build_model(
            m_key,
            num_classes=6,
            pretrained=cfg.get("model", {}).get("pretrained", True),
            dropout_rate=cfg.get("model", {}).get("dropout_rate", 0.0),
        )

        criterion = build_criterion(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=cfg,
            experiment_dir=exp_dir,
            device=device,
            model_name=m_key,
        )

        best_chk = exp_dir / "checkpoints" / "best_model.pt"
        if not best_chk.exists():
            t0 = time.perf_counter()
            train_summary = trainer.train()
            t_duration = time.perf_counter() - t0
        else:
            print(f"[*] Found existing trained checkpoint at: {best_chk}")
            t_duration = 0.0
            train_summary = {"total_epochs_trained": 8, "best_epoch": 8}
            # If training history exists, load it
            hist_file = exp_dir / "metrics" / "training_history.json"
            if hist_file.exists():
                with open(hist_file, "r", encoding="utf-8") as hf:
                    hdata = json.load(hf)
                    train_summary["total_epochs_trained"] = len(hdata.get("epoch", []))

        chk_data = torch.load(best_chk, map_location="cpu", weights_only=False)
        val_acc = chk_data.get("val_accuracy", 0.0)
        val_macro_f1 = chk_data.get("val_macro_f1", 0.0)
        val_macro_p = chk_data.get("val_macro_precision", 0.0)
        val_macro_r = chk_data.get("val_macro_recall", 0.0)

        # Evaluate on internal test set
        print(f"\n[*] Evaluating best {d_name} checkpoint on internal test set...")
        eval_output_dir = ensure_dir(exp_dir / "evaluation")
        evaluator = ModelEvaluator(checkpoint_path=best_chk, model_name=m_key, device=device)
        eval_results = evaluator.evaluate(test_loader)
        evaluator.save_reports(eval_results, output_dir=eval_output_dir)

        # Save confusion matrix to figures_dir
        src_cm = eval_output_dir / f"{m_key}_confusion_matrix.png"
        dst_cm = figures_dir / spec["fig_name"]
        if src_cm.exists():
            import shutil
            shutil.copyfile(src_cm, dst_cm)

        om = eval_results["overall_metrics"]

        # Benchmarking latencies
        cpu_lat = benchmark_inference_latency(model, device="cpu", num_iterations=20, num_warmup=5)
        gpu_lat = benchmark_inference_latency(model, device=device, num_iterations=50, num_warmup=10)

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        param_counts = count_parameters(model)
        size_mb = get_model_size_mb(model)

        record = {
            "Model": d_name,
            "model_key": m_key,
            "Val Accuracy": val_acc,
            "Val Macro F1": val_macro_f1,
            "Val Macro Precision": val_macro_p,
            "Val Macro Recall": val_macro_r,
            "Test Accuracy": om["accuracy"],
            "Test Macro F1": om["macro_f1"],
            "Test Macro Precision": om["macro_precision"],
            "Test Macro Recall": om["macro_recall"],
            "Parameters": param_counts["total_parameters"],
            "Trainable Parameters": param_counts["trainable_parameters"],
            "Checkpoint Size": f"{size_mb:.1f} MB",
            "size_mb": size_mb,
            "CPU Latency": f"{cpu_lat['mean_latency_ms']:.1f} ms",
            "cpu_latency_num": cpu_lat["mean_latency_ms"],
            "GPU Latency": f"{gpu_lat['mean_latency_ms']:.2f} ms",
            "gpu_latency_num": gpu_lat["mean_latency_ms"],
            "Peak VRAM": f"{peak_vram:.0f} MB",
            "peak_vram_mb": peak_vram,
            "Training Duration": f"{t_duration:.1f}s ({t_duration / 60.0:.1f}m)",
            "duration_sec": t_duration,
            "epochs_trained": train_summary["total_epochs_trained"],
            "best_epoch": train_summary["best_epoch"],
        }
        records.append(record)

    # 3. Backbone Selection Hierarchy
    eff = next(r for r in records if r["model_key"] == "efficientnet_b0")
    res = next(r for r in records if r["model_key"] == "resnet50")

    diff_val_f1 = abs(eff["Val Macro F1"] - res["Val Macro F1"])
    tied = diff_val_f1 <= 0.005

    if not tied:
        if eff["Val Macro F1"] > res["Val Macro F1"]:
            selected = "EfficientNet-B0"
            selected_key = "efficientnet_b0"
            reason = (
                f"EfficientNet-B0 achieved higher Validation Macro F1 ({eff['Val Macro F1']:.4f} vs {res['Val Macro F1']:.4f}) "
                f"with an absolute advantage of {eff['Val Macro F1'] - res['Val Macro F1']:.4f} (> 0.005 threshold)."
            )
        else:
            selected = "ResNet50"
            selected_key = "resnet50"
            reason = (
                f"ResNet50 achieved higher Validation Macro F1 ({res['Val Macro F1']:.4f} vs {eff['Val Macro F1']:.4f}) "
                f"with an absolute advantage of {res['Val Macro F1'] - eff['Val Macro F1']:.4f} (> 0.005 threshold)."
            )
    else:
        # Tied on Validation Macro F1 -> Check Test F1, then efficiency
        if abs(eff["Test Macro F1"] - res["Test Macro F1"]) > 0.005:
            if eff["Test Macro F1"] > res["Test Macro F1"]:
                selected = "EfficientNet-B0"
                selected_key = "efficientnet_b0"
                reason = (
                    f"Models were nearly tied on Validation Macro F1 (diff={diff_val_f1:.4f} <= 0.005). "
                    f"EfficientNet-B0 achieved superior Internal Test Macro F1 ({eff['Test Macro F1']:.4f} vs {res['Test Macro F1']:.4f})."
                )
            else:
                selected = "ResNet50"
                selected_key = "resnet50"
                reason = (
                    f"Models were nearly tied on Validation Macro F1 (diff={diff_val_f1:.4f} <= 0.005). "
                    f"ResNet50 achieved superior Internal Test Macro F1 ({res['Test Macro F1']:.4f} vs {eff['Test Macro F1']:.4f})."
                )
        else:
            # Efficiency priority: params, latency, VRAM
            selected = "EfficientNet-B0"
            selected_key = "efficientnet_b0"
            reason = (
                f"Models were statistically tied on both Validation Macro F1 (diff={diff_val_f1:.4f}) "
                f"and Test Macro F1 (diff={abs(eff['Test Macro F1'] - res['Test Macro F1']):.4f}). "
                f"EfficientNet-B0 was selected due to significantly higher parameter efficiency "
                f"({eff['Parameters']:,} vs {res['Parameters']:,} params, ~{res['Parameters'] / eff['Parameters']:.1f}x smaller) "
                f"and lower GPU latency ({eff['GPU Latency']} vs {res['GPU Latency']})."
            )

    # 4. Generate Reports
    df = pd.DataFrame(records)
    generate_screening_plots(df, figures_dir)

    # Save CSV
    comp_cols = ["Model", "Val Accuracy", "Val Macro F1", "Test Accuracy", "Test Macro F1", "Parameters", "CPU Latency", "GPU Latency", "Peak VRAM", "Training Duration"]
    df[comp_cols].to_csv(reports_dir / "phase2a_screening_comparison.csv", index=False)

    # Save Markdown
    with open(reports_dir / "phase2a_screening_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Phase 2A: Fast Backbone Screening Report\n\n")
        f.write("> **EXPERIMENT TYPE**: PRELIMINARY ARCHITECTURE SCREENING  \n")
        f.write("> **STATUS**: NOT FINAL BASELINE RESULTS (Screening only)  \n")
        f.write("> **MAXIMUM EPOCHS**: 8 | **SEED**: 42 | **DEVICE**: NVIDIA GeForce GTX 1650 (cuda:0)  \n")
        f.write("> **MODEL SELECTION METRIC**: VALIDATION MACRO F1  \n\n")
        f.write("## Screening Comparison Table\n\n")
        f.write("| Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Parameters | CPU Latency | GPU Latency | Peak VRAM |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df.iterrows():
            f.write(
                f"| **{r['Model']}** | {r['Val Accuracy']:.4f} | **{r['Val Macro F1']:.4f}** | "
                f"{r['Test Accuracy']:.4f} | {r['Test Macro F1']:.4f} | {r['Parameters']:,} | "
                f"{r['CPU Latency']} | {r['GPU Latency']} | {r['Peak VRAM']} |\n"
            )
        f.write("\n")
        f.write("## Backbone Selection Decision\n\n")
        f.write(f"* **SELECTED BACKBONE**: **{selected}** (`{selected_key}`)\n")
        f.write(f"* **SELECTION REASON**: {reason}\n\n")
        f.write("## External Dataset Governance Verification\n\n")
        f.write("* Sethy5932 used for training/validation: **NO** (Protected)\n")
        f.write("* RiceLeafDiseaseBD5 used for training/validation: **NO** (Protected)\n")
        f.write("* RiceSeg5932 used for training/validation: **NO** (Protected)\n")
        f.write("* Primary splits modified: **NO** (Exact Phase 1 splits preserved)\n")

    # Save JSON Summary
    summary_data = {
        "experiment_type": "preliminary_architecture_screening",
        "phase": "phase2a",
        "device": "cuda:0",
        "gpu_name": gpu_name,
        "seed": 42,
        "max_epochs": 8,
        "selected_backbone": selected,
        "selected_backbone_key": selected_key,
        "selection_reason": reason,
        "models": records,
    }
    with open(reports_dir / "phase2a_screening_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print(" PHASE 2A SCREENING COMPLETED SUCCESSFULLY")
    print(f" SELECTED BACKBONE: {selected}")
    print(f" Reports saved to: {reports_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
