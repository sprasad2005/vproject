"""Model training script for RiceGuard.

Usage:
    python scripts/train.py --config configs/experiments/efficientnet_b0_baseline.yaml
    python scripts/train.py --model resnet50 --epochs 15 --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import build_transforms
from src.data.dataset import RiceLeafDataset, create_dataloader
from src.models.model_factory import build_model
from src.training.losses import build_criterion
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model on RiceLeafDiseaseBD dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config YAML")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cpu, cuda)")
    return parser.parse_args()


def load_config(config_path: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Override with CLI args
    config.setdefault("model", {})
    if args.model:
        config["model"]["name"] = args.model
    config["model"].setdefault("num_classes", 6)
    config["model"].setdefault("pretrained", True)

    config.setdefault("training", {})
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    config.setdefault("optimizer", {})
    if args.lr:
        config["optimizer"]["lr"] = args.lr

    config.setdefault("reproducibility", {})
    config["reproducibility"]["seed"] = args.seed

    return config


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args)
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    model_name = cfg.get("model", {}).get("name", args.model)
    print("=" * 75)
    print(f" RiceGuard Baseline Training Engine — [{model_name.upper()}]")
    print("=" * 75)

    # 1. Transforms & DataLoaders
    train_transform = build_transforms(is_training=True, config=cfg)
    eval_transform = build_transforms(is_training=False, config=cfg)

    train_csv = PROJECT_ROOT / "splits" / "primary_train.csv"
    val_csv = PROJECT_ROOT / "splits" / "primary_val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Primary split files not found in splits/. Run create_primary_split.py first.")

    train_dataset = RiceLeafDataset(train_csv, root_dir=PROJECT_ROOT, transform=train_transform, is_training=True)
    val_dataset = RiceLeafDataset(val_csv, root_dir=PROJECT_ROOT, transform=eval_transform, is_training=False)

    batch_size = int(cfg.get("training", {}).get("batch_size", 16))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))
    pin_memory = bool(cfg.get("training", {}).get("pin_memory", False))

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"[*] Loaded Datasets: Train={len(train_dataset):,} samples, Val={len(val_dataset):,} samples")

    # 2. Build Model
    model = build_model(
        model_name,
        num_classes=cfg.get("model", {}).get("num_classes", 6),
        pretrained=cfg.get("model", {}).get("pretrained", True),
        dropout_rate=cfg.get("model", {}).get("dropout_rate", 0.0),
    )

    # 3. Loss, Optimizer, Scheduler
    class_weights = train_dataset.get_class_weights() if cfg.get("loss", {}).get("use_class_weights", False) else None
    criterion = build_criterion(cfg, class_weights=class_weights)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        model_name=model_name,
        device=args.device,
    )

    results = trainer.train()

    print("\n" + "=" * 75)
    print(f" Training Complete for {model_name}!")
    print(f"   * Best Val Macro F1 : {results['best_val_macro_f1']:.4f} (Epoch {results['best_epoch']})")
    print(f"   * Best Checkpoint   : {results['checkpoints']['best_model']}")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
