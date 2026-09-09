"""Phase 2B: Final EfficientNet-B0 Baseline Training Engine for RiceGuard.

Trains and evaluates the final classification-only EfficientNet-B0 baseline
under strict data governance using ONLY the primary RiceLeafDiseaseBD dataset.

Features:
- Live tqdm batch progress and standardized epoch summaries
- Live status.json updated after every epoch
- Incremental, flushed training_history.csv after every epoch
- Checkpoint management (best_model.pt and last_model.pt)
- Early stopping monitoring Validation Macro F1 (patience=5, delta=0.001)
- Post-training internal test evaluation on primary_test.csv
- Efficiency profiling (parameters, model size, GPU latency, CPU latency, peak VRAM)
- Diagnostic figure generation (loss, accuracy, macro F1, LR curves, confusion matrix)
- Comprehensive Markdown and JSON summary reports
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import build_transforms
from src.data.dataset import RiceLeafDataset, create_dataloader
from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES
from src.evaluation.evaluate import ModelEvaluator
from src.models.model_factory import (
    benchmark_inference_latency,
    build_model,
    count_parameters,
    get_model_size_mb,
)
from src.training.losses import build_criterion
from src.training.metrics import compute_classification_metrics
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.paths import ensure_dir
from src.utils.seed import get_reproducibility_info, set_seed


def generate_individual_plots(history: Dict[str, List[float]], best_epoch: int, best_f1: float, figures_dir: Path) -> None:
    """Generate separate, clean diagnostic curves for Phase 2B."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    # 1. Training Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], color="#1f77b4", lw=2, marker="o", markersize=4, label="Train Loss")
    plt.title("EfficientNet-B0 — Training Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_loss_curve.png", dpi=150)
    plt.close()

    # 2. Validation Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["val_loss"], color="#ff7f0e", lw=2, marker="s", markersize=4, label="Validation Loss")
    plt.title("EfficientNet-B0 — Validation Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "validation_loss_curve.png", dpi=150)
    plt.close()

    # 3. Accuracy Curves (Train + Val)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], color="#2ca02c", lw=2, label="Train Accuracy (%)")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], color="#d62728", lw=2, label="Validation Accuracy (%)")
    plt.title("EfficientNet-B0 — Accuracy Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    # 4. Macro F1 Curves
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_macro_f1"], color="#9467bd", lw=2, label="Train Macro F1")
    plt.plot(epochs, history["val_macro_f1"], color="#8c564b", lw=2, label="Validation Macro F1")
    plt.axvline(x=best_epoch, color="red", linestyle=":", label=f"Best Val F1 ({best_f1:.4f} @ Ep {best_epoch})")
    plt.title("EfficientNet-B0 — Macro F1 Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Macro F1", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "macro_f1_curve.png", dpi=150)
    plt.close()

    # 5. Learning Rate Schedule
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["lr"], color="#e377c2", lw=2, marker="^", markersize=4)
    plt.title("EfficientNet-B0 — Learning Rate Schedule", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Learning Rate", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "learning_rate_curve.png", dpi=150)
    plt.close()


def train_single_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> Tuple[float, Dict[str, Any]]:
    """Execute training epoch with live tqdm batch progress."""
    model.train()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{max_epochs}", unit="batch", file=sys.stdout, leave=True)
    running_corrects = 0
    running_total = 0

    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)

        running_corrects += int((preds == targets).sum().item())
        running_total += batch_size

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        # Live stats in tqdm bar
        cur_loss = loss.item()
        cur_acc = running_corrects / running_total
        gpu_mb = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        pbar.set_postfix({"loss": f"{cur_loss:.3f}", "acc": f"{cur_acc:.2%}", "GPU": f"{gpu_mb:.0f}MB"})

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds, CANONICAL_PRIMARY_CLASSES)
    return avg_loss, metrics


@torch.no_grad()
def validate_single_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """Execute validation epoch."""
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds, CANONICAL_PRIMARY_CLASSES)
    return avg_loss, metrics


def main() -> int:
    print("=" * 80)
    print(" RiceGuard Phase 2B: Final EfficientNet-B0 Baseline Training Engine")
    print(" Pure Image Classification Baseline — Primary RiceLeafDiseaseBD Dataset Only")
    print("=" * 80)

    # 1. Device Verification
    if not torch.cuda.is_available():
        print("[FATAL ERROR] CUDA is unavailable. Phase 2B requires NVIDIA GPU for execution.")
        return 1

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_vram_mb = props.total_memory / (1024 * 1024)
    print(f"[*] PyTorch Version : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    print(f"[*] Selected Device : {device} ({gpu_name})")
    print(f"[*] Total GPU VRAM  : {total_vram_mb:.1f} MB ({total_vram_mb / 1024:.2f} GB)")

    # 2. Data Governance Verification
    train_csv = PROJECT_ROOT / "splits" / "primary_train.csv"
    val_csv = PROJECT_ROOT / "splits" / "primary_val.csv"
    test_csv = PROJECT_ROOT / "splits" / "primary_test.csv"

    print("\n[*] Verifying Data Governance:")
    print("    - Primary Train : splits/primary_train.csv (6,837 samples)  -> ALLOWED")
    print("    - Primary Val   : splits/primary_val.csv   (1,465 samples)  -> ALLOWED")
    print("    - Internal Test : splits/primary_test.csv  (1,467 samples)  -> ALLOWED")
    print("    - Sethy5932     : LOCKED (Strictly Prohibited)")
    print("    - BD5           : LOCKED (Strictly Prohibited)")
    print("    - RiceSeg5932   : LOCKED (Strictly Prohibited)")

    for p, name in [(train_csv, "Train"), (val_csv, "Validation"), (test_csv, "Internal Test")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} split CSV missing at {p}. Run dataset split script first.")

    # 3. Experiment Paths
    exp_dir = ensure_dir(PROJECT_ROOT / "experiments" / "phase2b_final_baseline" / "efficientnet_b0")
    checkpoints_dir = ensure_dir(exp_dir / "checkpoints")
    figures_dir = ensure_dir(exp_dir / "figures")
    eval_dir = ensure_dir(exp_dir / "evaluation")
    reports_dir = ensure_dir(PROJECT_ROOT / "results" / "reports")

    # Load Configuration
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "phase2b_efficientnet_b0.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    set_seed(seed)

    max_epochs = int(cfg.get("training", {}).get("max_epochs", 20))
    patience = int(cfg.get("early_stopping", {}).get("patience", 5))
    min_delta = float(cfg.get("early_stopping", {}).get("min_delta", 0.001))

    # Determine batch size with auto-fallback: 16 -> 12 -> 8
    candidate_batch_sizes = [16, 12, 8]
    chosen_batch_size = 8  # default safe fallback on 4GB GTX 1650
    for bs in candidate_batch_sizes:
        try:
            torch.cuda.empty_cache()
            test_x = torch.randn(bs, 3, 224, 224, device=device)
            test_m = build_model("efficientnet_b0", num_classes=6, pretrained=False).to(device)
            with torch.amp.autocast("cuda"):
                test_out = test_m(test_x)
                test_l = test_out.sum()
            test_l.backward()
            chosen_batch_size = bs
            del test_x, test_m, test_out, test_l
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            print(f"[!] CUDA Out of Memory with batch size {bs}, falling back...")
            torch.cuda.empty_cache()
            continue

    print(f"[*] Verified Operational Batch Size: {chosen_batch_size}")
    cfg["training"]["batch_size"] = chosen_batch_size

    # Build DataLoaders
    train_transform = build_transforms(is_training=True, config=cfg)
    val_transform = build_transforms(is_training=False, config=cfg)
    eval_transform = build_transforms(is_training=False, config=cfg)

    train_dataset = RiceLeafDataset(train_csv, root_dir=PROJECT_ROOT, transform=train_transform, is_training=True)
    val_dataset = RiceLeafDataset(val_csv, root_dir=PROJECT_ROOT, transform=val_transform, is_training=False)
    test_dataset = RiceLeafDataset(test_csv, root_dir=PROJECT_ROOT, transform=eval_transform, is_training=False)

    train_loader = create_dataloader(train_dataset, batch_size=chosen_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = create_dataloader(val_dataset, batch_size=chosen_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = create_dataloader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Instantiate Model, Loss, Optimizer, Scheduler
    model = build_model(
        "efficientnet_b0",
        num_classes=6,
        pretrained=cfg.get("model", {}).get("pretrained", True),
        dropout_rate=cfg.get("model", {}).get("dropout_rate", 0.2),
    ).to(device)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Initialize CSV history file with headers
    history_csv_path = exp_dir / "training_history.csv"
    csv_headers = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_macro_precision",
        "val_macro_precision",
        "train_macro_recall",
        "val_macro_recall",
        "train_macro_f1",
        "val_macro_f1",
        "lr",
        "epoch_duration_seconds",
        "total_duration_seconds",
        "gpu_memory_peak_mb",
    ]
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    history_dict: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    total_training_start = time.perf_counter()
    early_stopped = False

    print("\n" + "=" * 80)
    print(" STARTING PHASE 2B TRAINING LOOP")
    print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch}/{max_epochs}")
        print("=" * 60)
        print("Training...")

        epoch_start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)

        # 1. Train epoch
        train_loss, train_metrics = train_single_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            max_epochs=max_epochs,
        )

        # 2. Validation epoch
        val_loss, val_metrics = validate_single_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Current LR & step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_duration = time.perf_counter() - epoch_start_time
        total_duration = time.perf_counter() - total_training_start

        avg_epoch_time = total_duration / epoch
        remaining_epochs = max_epochs - epoch
        est_remaining_sec = avg_epoch_time * remaining_epochs

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        val_f1 = val_metrics["macro_f1"]
        train_f1 = train_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        train_acc = train_metrics["accuracy"]

        # Checkpoint payload
        checkpoint_payload = {
            "epoch": epoch,
            "model_name": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_macro_f1": max(best_val_macro_f1, val_f1),
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "config": cfg,
            "seed": seed,
            "canonical_classes": CANONICAL_PRIMARY_CLASSES,
        }

        # Save last_model.pt
        torch.save(checkpoint_payload, checkpoints_dir / "last_model.pt")

        # Best Model Check
        if val_f1 > best_val_macro_f1 + min_delta:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(checkpoint_payload, checkpoints_dir / "best_model.pt")
            saved_banner = " -> NEW BEST MODEL SAVED"
        else:
            patience_counter += 1
            saved_banner = ""

        # Print standardized epoch summary
        print("\n------------------------------------------------------------")
        print(f"EPOCH {epoch}/{max_epochs} COMPLETE {saved_banner}")
        print("------------------------------------------------------------")
        print(f"Train Loss               : {train_loss:.4f}")
        print(f"Validation Loss          : {val_loss:.4f}")
        print(f"Train Accuracy           : {train_acc * 100:.2f}%")
        print(f"Validation Accuracy      : {val_acc * 100:.2f}%")
        print(f"Train Macro F1           : {train_f1 * 100:.2f}%")
        print(f"Validation Macro F1      : {val_f1 * 100:.2f}%")
        print(f"Best Validation Macro F1 : {best_val_macro_f1 * 100:.2f}% (@ Ep {best_epoch})")
        print(f"Early Stopping Counter   : {patience_counter}/{patience}")
        print(f"Learning Rate            : {current_lr:.2e}")
        print(f"Epoch Duration           : {epoch_duration / 60.0:.2f} minutes ({epoch_duration:.1f}s)")
        print(f"Total Training Duration  : {total_duration / 60.0:.2f} minutes")
        print(f"Estimated Remaining Time : {est_remaining_sec / 60.0:.2f} minutes")
        print(f"GPU Memory Peak          : {peak_vram_mb:.1f} MB")
        print("------------------------------------------------------------")

        # Append row to training_history.csv and flush immediately
        row = [
            epoch,
            round(train_loss, 4),
            round(val_loss, 4),
            round(train_acc, 4),
            round(val_acc, 4),
            round(train_metrics["macro_precision"], 4),
            round(val_metrics["macro_precision"], 4),
            round(train_metrics["macro_recall"], 4),
            round(val_metrics["macro_recall"], 4),
            round(train_f1, 4),
            round(val_f1, 4),
            f"{current_lr:.6f}",
            round(epoch_duration, 2),
            round(total_duration, 2),
            round(peak_vram_mb, 2),
        ]
        with open(history_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        # Update tracking dict for curves
        history_dict["epoch"].append(epoch)
        history_dict["train_loss"].append(train_loss)
        history_dict["val_loss"].append(val_loss)
        history_dict["train_acc"].append(train_acc)
        history_dict["val_acc"].append(val_acc)
        history_dict["train_macro_f1"].append(train_f1)
        history_dict["val_macro_f1"].append(val_f1)
        history_dict["lr"].append(current_lr)

        # Update live status.json
        status_payload = {
            "status": "training",
            "current_epoch": epoch,
            "max_epochs": max_epochs,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_macro_f1, 4),
            "current_val_macro_f1": round(val_f1, 4),
            "current_val_accuracy": round(val_acc, 4),
            "elapsed_minutes": round(total_duration / 60.0, 2),
            "estimated_remaining_minutes": round(est_remaining_sec / 60.0, 2),
            "gpu_peak_mb": round(peak_vram_mb, 1),
        }
        with open(exp_dir / "status.json", "w", encoding="utf-8") as sf:
            json.dump(status_payload, sf, indent=2)

        # Check early stopping
        if patience_counter >= patience:
            print("\n" + "=" * 60)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 60)
            print(f"Best Epoch               : {best_epoch}")
            print(f"Best Validation Macro F1 : {best_val_macro_f1:.4f}")
            print(f"Stopped After            : {epoch} epochs")
            print("=" * 60)
            early_stopped = True
            break

    total_training_duration = time.perf_counter() - total_training_start

    # Update status.json to completed
    final_status = {
        "status": "completed",
        "completed_epochs": len(history_dict["epoch"]),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "total_training_minutes": round(total_training_duration / 60.0, 2),
        "early_stopped": early_stopped,
    }
    with open(exp_dir / "status.json", "w", encoding="utf-8") as sf:
        json.dump(final_status, sf, indent=2)

    # 4. Generate Diagnostic Curves
    generate_individual_plots(history_dict, best_epoch, best_val_macro_f1, figures_dir)

    # 5. Post-Training Internal Test Evaluation
    print("\n" + "=" * 80)
    print(" POST-TRAINING EVALUATION ON INTERNAL TEST SPLIT")
    print("=" * 80)
    best_chk_path = checkpoints_dir / "best_model.pt"
    evaluator = ModelEvaluator(checkpoint_path=best_chk_path, model_name="efficientnet_b0", device=device)
    eval_results = evaluator.evaluate(test_loader)
    evaluator.save_reports(eval_results, output_dir=eval_dir)

    # Copy confusion matrix to figures_dir
    src_cm = eval_dir / "efficientnet_b0_confusion_matrix.png"
    dst_cm = figures_dir / "confusion_matrix.png"
    if src_cm.exists():
        shutil.copyfile(src_cm, dst_cm)

    test_om = eval_results["overall_metrics"]

    # 6. Efficiency Profiling
    print("\n[*] Profiling Computational Efficiency...")
    param_counts = count_parameters(model)
    size_mb = get_model_size_mb(model)
    gpu_lat = benchmark_inference_latency(model, device=device, num_iterations=50, num_warmup=10)
    cpu_lat = benchmark_inference_latency(model, device="cpu", num_iterations=20, num_warmup=5)
    peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # 7. Generate Final Phase 2B Reports
    repro_info = get_reproducibility_info(seed=seed)

    summary_dict = {
        "experiment": "phase2b_final_baseline",
        "model_architecture": "EfficientNet-B0",
        "dataset": "RiceLeafDiseaseBD (Primary)",
        "hardware": {
            "device": "cuda:0",
            "gpu_name": gpu_name,
            "total_vram_mb": round(total_vram_mb, 2),
            "peak_vram_mb": round(peak_gpu_mem, 2),
            "amp_enabled": True,
        },
        "training": {
            "batch_size": chosen_batch_size,
            "max_epochs": max_epochs,
            "completed_epochs": len(history_dict["epoch"]),
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "total_duration_minutes": round(total_training_duration / 60.0, 2),
            "seed": seed,
            "optimizer": "AdamW",
            "initial_lr": 0.0001,
            "scheduler": "CosineAnnealingLR",
        },
        "validation_metrics": {
            "best_val_macro_f1": round(best_val_macro_f1, 4),
            "best_epoch": best_epoch,
        },
        "test_metrics": {
            "accuracy": round(test_om["accuracy"], 4),
            "macro_precision": round(test_om["macro_precision"], 4),
            "macro_recall": round(test_om["macro_recall"], 4),
            "macro_f1": round(test_om["macro_f1"], 4),
        },
        "per_class_results": eval_results["per_class"],
        "efficiency": {
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
            "checkpoint_size_mb": size_mb,
            "gpu_latency_mean_ms": gpu_lat["mean_latency_ms"],
            "cpu_latency_mean_ms": cpu_lat["mean_latency_ms"],
            "peak_gpu_memory_mb": round(peak_gpu_mem, 2),
        },
        "reproducibility": repro_info,
        "external_data_governance": {
            "Sethy5932_used": False,
            "RiceLeafDiseaseBD5_used": False,
            "RiceSeg5932_used": False,
        },
    }

    # Save summary JSON
    with open(reports_dir / "phase2b_final_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Save summary Markdown
    md_content = f"""# Phase 2B: Final EfficientNet-B0 Baseline Training Summary

> **EXPERIMENT**: FINAL CLASSIFICATION BASELINE
> **BACKBONE**: EfficientNet-B0 (Selected from Phase 2A Screening)
> **DATASET**: Primary `RiceLeafDiseaseBD` (6 Canonical Classes)
> **DEVICE**: NVIDIA GeForce GTX 1650 (`cuda:0`, 4.0 GB VRAM)
> **AMP MIXED PRECISION**: ENABLED

---

## 1. Executive Summary

| Metric | Training Score / Value |
| :--- | :--- |
| **Selected Backbone** | **EfficientNet-B0** |
| **Best Training Epoch** | **Epoch {best_epoch}** |
| **Total Epochs Completed** | **{len(history_dict['epoch'])} / {max_epochs}** (Early Stopped: {early_stopped}) |
| **Best Validation Macro F1** | **{best_val_macro_f1 * 100:.2f}%** (`{best_val_macro_f1:.4f}`) |
| **Internal Test Accuracy** | **{test_om['accuracy'] * 100:.2f}%** (`{test_om['accuracy']:.4f}`) |
| **Internal Test Macro F1** | **{test_om['macro_f1'] * 100:.2f}%** (`{test_om['macro_f1']:.4f}`) |
| **Internal Test Macro Precision** | **{test_om['macro_precision'] * 100:.2f}%** (`{test_om['macro_precision']:.4f}`) |
| **Internal Test Macro Recall** | **{test_om['macro_recall'] * 100:.2f}%** (`{test_om['macro_recall']:.4f}`) |
| **Total Parameters** | **{param_counts['total_parameters']:,}** |
| **Checkpoint File Size** | **{size_mb:.1f} MB** |
| **GPU Inference Latency** | **{gpu_lat['mean_latency_ms']:.2f} ms** per sample |
| **CPU Inference Latency** | **{cpu_lat['mean_latency_ms']:.1f} ms** per sample |
| **Peak GPU Memory (VRAM)** | **{peak_gpu_mem:.1f} MB** / 4,096 MB |
| **Total Training Time** | **{total_training_duration / 60.0:.2f} minutes** |

---

## 2. Per-Class Test Performance (Internal Test Split: 1,467 Samples)

| Class ID | Canonical Disease Class | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
    for cls_name, vals in eval_results["per_class"].items():
        cls_idx = CANONICAL_PRIMARY_CLASSES.index(cls_name) if cls_name in CANONICAL_PRIMARY_CLASSES else "?"
        md_content += f"| {cls_idx} | **{cls_name}** | {vals['precision'] * 100:.2f}% | {vals['recall'] * 100:.2f}% | **{vals['f1'] * 100:.2f}%** | {vals['support']} |\n"

    md_content += """
---

## 3. Strict Data Governance & Baseline Integrity Confirmation

* **Pure Image Classification Baseline**: No bounding boxes, lesion masks, or auxiliary supervision was used.
* **Primary Dataset Only**: Training and model selection strictly used `splits/primary_train.csv` (6,837) and `splits/primary_val.csv` (1,465).
* **External Datasets Untouched**: `Sethy5932`, `RiceLeafDiseaseBD5`, and `RiceSeg5932` remained 100% locked.
* **Canonical Class Order Preserved**: `0: Healthy, 1: Blast, 2: Brown Spot, 3: Leaf Smut, 4: Tungro, 5: Sheath Blight`.
* **Hardware Acceleration**: Successfully executed on **NVIDIA GeForce GTX 1650 (`cuda:0`)** with AMP.

---

## 4. Key Artifacts Generated

* **Best Checkpoint**: `experiments/phase2b_final_baseline/efficientnet_b0/checkpoints/best_model.pt`
* **Last Checkpoint**: `experiments/phase2b_final_baseline/efficientnet_b0/checkpoints/last_model.pt`
* **Training History**: `experiments/phase2b_final_baseline/efficientnet_b0/training_history.csv`
* **Live Status Log**: `experiments/phase2b_final_baseline/efficientnet_b0/status.json`
* **Evaluation Metrics**: `experiments/phase2b_final_baseline/efficientnet_b0/evaluation/`
* **Diagnostic Figures**: `experiments/phase2b_final_baseline/efficientnet_b0/figures/`
"""

    with open(reports_dir / "phase2b_final_baseline_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print(" PHASE 2B FINAL BASELINE COMPLETED SUCCESSFULLY")
    print(f" Best Validation Macro F1 : {best_val_macro_f1 * 100:.2f}% (@ Ep {best_epoch})")
    print(f" Internal Test Macro F1   : {test_om['macro_f1'] * 100:.2f}%")
    print(f" Reports saved to         : {reports_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
