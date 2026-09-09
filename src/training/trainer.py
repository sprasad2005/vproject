"""Comprehensive Training and Validation Engine for RiceGuard Baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir, get_project_root
from src.utils.seed import get_reproducibility_info

matplotlib.use("Agg")


class Trainer:
    """Manages model training, validation, early stopping, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        experiment_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_name: str = "model",
    ) -> None:
        self.config = config or {}
        self.root = get_project_root()
        self.model_name = model_name

        # Setup experiment directory structure
        if experiment_dir:
            self.exp_dir = Path(experiment_dir)
        elif self.config.get("experiment", {}).get("name"):
            self.exp_dir = self.root / "experiments" / self.config["experiment"]["name"]
        elif self.config.get("experiment_name"):
            self.exp_dir = self.root / "experiments" / self.config["experiment_name"]
        else:
            self.exp_dir = self.root / "experiments" / f"{model_name}_baseline"

        self.checkpoints_dir = ensure_dir(self.exp_dir / "checkpoints")
        self.logs_dir = ensure_dir(self.exp_dir / "logs")
        self.metrics_dir = ensure_dir(self.exp_dir / "metrics")
        self.figures_dir = ensure_dir(self.exp_dir / "figures")

        # Setup logger
        self.logger = get_logger(f"trainer.{model_name}")

        # Device handling
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device(use_cuda=self.config.get("training", {}).get("use_cuda", True))

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training hyperparameters
        train_cfg = self.config.get("training", {})
        es_cfg = self.config.get("early_stopping", {})
        self.epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 20)))
        self.patience = int(es_cfg.get("patience", train_cfg.get("early_stopping_patience", 10)))
        self.min_delta = float(es_cfg.get("min_delta", train_cfg.get("early_stopping_min_delta", 1e-4)))
        self.seed = int(self.config.get("reproducibility", {}).get("seed", 42))

        # AMP handling (only when CUDA is available)
        self.use_amp = bool(train_cfg.get("amp", False)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

        # Tracking state
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "lr": [],
            "gpu_memory_allocated_mb": [],
            "gpu_memory_peak_mb": [],
            "gpu_memory_reserved_mb": [],
        }
        self.best_val_macro_f1 = -1.0
        self.best_epoch = 0
        self.early_stopped = False

        # Save experiment environment metadata
        self._save_environment_metadata()

    def _save_environment_metadata(self) -> None:
        """Save environment and configuration reproducibility records."""
        repro_info = get_reproducibility_info(seed=self.seed)
        repro_info["model_name"] = self.model_name
        repro_info["device"] = str(self.device)
        repro_info["canonical_classes"] = CANONICAL_PRIMARY_CLASSES
        repro_info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(self.exp_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(repro_info, f, indent=2)

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        all_targets: List[int] = []
        all_preds: List[int] = []

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.train_loader):
                batch_acc = (preds == targets).float().mean().item()
                self.logger.info(
                    f"Epoch [{epoch}/{self.epochs}] Step [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"- Batch Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.4f}"
                )

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = compute_classification_metrics(all_targets, all_preds)
        return avg_loss, metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Execute one complete validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_targets: List[int] = []
        all_preds: List[int] = []

        for images, targets, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = compute_classification_metrics(all_targets, all_preds)
        return avg_loss, metrics

    def train(self) -> Dict[str, Any]:
        """Execute full training loop with early stopping and checkpoint management."""
        self.logger.info(f"Starting training run for {self.model_name} on device: {self.device}")
        self.logger.info(f"Total epochs: {self.epochs}, Early stopping patience: {self.patience}")

        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()

            # Train and validate
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate_epoch(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("Macro_F1/Train", train_metrics["macro_f1"], epoch)
            self.writer.add_scalar("Macro_F1/Val", val_metrics["macro_f1"], epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            # Record GPU memory if on CUDA
            if self.device.type == "cuda":
                gpu_alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                gpu_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                gpu_res_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                self.writer.add_scalar("GPU/Allocated_MB", gpu_alloc_mb, epoch)
                self.writer.add_scalar("GPU/Peak_MB", gpu_peak_mb, epoch)
                self.writer.add_scalar("GPU/Reserved_MB", gpu_res_mb, epoch)
                self.history["gpu_memory_allocated_mb"].append(round(gpu_alloc_mb, 2))
                self.history["gpu_memory_peak_mb"].append(round(gpu_peak_mb, 2))
                self.history["gpu_memory_reserved_mb"].append(round(gpu_res_mb, 2))
            else:
                self.history["gpu_memory_allocated_mb"].append(0.0)
                self.history["gpu_memory_peak_mb"].append(0.0)
                self.history["gpu_memory_reserved_mb"].append(0.0)

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_macro_f1"].append(train_metrics["macro_f1"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["lr"].append(current_lr)

            duration = time.perf_counter() - epoch_start
            val_f1 = val_metrics["macro_f1"]

            gpu_str = f" | GPU Peak: {gpu_peak_mb:.1f}MB" if self.device.type == "cuda" else ""
            self.logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({duration:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_f1:.4f} | LR: {current_lr:.6f}{gpu_str}"
            )

            # Save latest checkpoint
            self._save_checkpoint("last_model.pt", epoch, val_loss, val_metrics)

            # Early stopping check (based on Validation Macro F1)
            if val_f1 > self.best_val_macro_f1 + self.min_delta:
                self.best_val_macro_f1 = val_f1
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, val_loss, val_metrics)
                self.logger.info(f" -> [BEST] New best Validation Macro F1: {val_f1:.4f} (Saved best_model.pt)")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch} (Patience: {self.patience})")
                    self.early_stopped = True
                    break

        self.writer.close()

        # Save training history and curves
        self._save_history_and_plots()

        def safe_relpath(p: Path) -> str:
            try:
                return str(p.relative_to(self.root)).replace("\\", "/")
            except ValueError:
                return str(p).replace("\\", "/")

        return {
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "total_epochs_trained": len(self.history["epoch"]),
            "early_stopped": self.early_stopped,
            "checkpoints": {
                "best_model": safe_relpath(self.checkpoints_dir / "best_model.pt"),
                "last_model": safe_relpath(self.checkpoints_dir / "last_model.pt"),
            },
        }

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_metrics: Dict[str, Any]) -> None:
        """Save standardized checkpoint file."""
        state = {
            "model_name": self.model_name,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "seed": self.seed,
            "config": self.config,
            "canonical_classes": CANONICAL_PRIMARY_CLASSES,
        }
        torch.save(state, self.checkpoints_dir / filename)

    def _save_history_and_plots(self) -> None:
        """Save history.json and render diagnostic training curves."""
        with open(self.metrics_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Export CSV history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.metrics_dir / "training_history.csv", index=False)
        history_df.to_csv(self.exp_dir / "training_history.csv", index=False)

        epochs = self.history["epoch"]

        # 1. Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(epochs, self.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
        plt.title(f"{self.model_name} — Loss Curves", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "loss_curve.png", dpi=150)
        plt.close()

        # 2. Accuracy Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_acc"], label="Train Accuracy", color="#2ca02c", lw=2)
        plt.plot(epochs, self.history["val_acc"], label="Val Accuracy", color="#d62728", lw=2)
        plt.title(f"{self.model_name} — Accuracy Curves", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "accuracy_curve.png", dpi=150)
        plt.close()

        # 3. Macro F1 Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_macro_f1"], label="Train Macro F1", color="#9467bd", lw=2)
        plt.plot(epochs, self.history["val_macro_f1"], label="Val Macro F1", color="#8c564b", lw=2)
        plt.axvline(x=self.best_epoch, color="red", linestyle=":", label=f"Best Val F1 ({self.best_val_macro_f1:.4f})")
        plt.title(f"{self.model_name} — Macro F1 Curves", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "macro_f1_curve.png", dpi=150)
        plt.close()

        # 4. Learning Rate Curve
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, self.history["lr"], color="#e377c2", lw=2)
        plt.title(f"{self.model_name} — Learning Rate Schedule", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "lr_curve.png", dpi=150)
        plt.close()
