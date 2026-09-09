"""Model evaluation script for RiceGuard.

Usage:
    python scripts/evaluate.py --checkpoint experiments/efficientnet_b0_baseline/checkpoints/best_model.pt
    python scripts/evaluate.py --model resnet50 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import build_transforms
from src.data.dataset import RiceLeafDataset, create_dataloader
from src.evaluation.evaluate import ModelEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a baseline model on RiceLeafDiseaseBD internal test set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="Model name if checkpoint not specified")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Compute device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 75)
    print(" RiceGuard Baseline Evaluation Engine")
    print("=" * 75)

    if args.checkpoint:
        chk_path = Path(args.checkpoint)
        if not chk_path.is_absolute():
            chk_path = PROJECT_ROOT / chk_path
        evaluator = ModelEvaluator(checkpoint_path=chk_path, device=args.device)
    else:
        chk_path = PROJECT_ROOT / "experiments" / f"{args.model}_baseline" / "checkpoints" / "best_model.pt"
        if not chk_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at default path: {chk_path}")
        evaluator = ModelEvaluator(checkpoint_path=chk_path, model_name=args.model, device=args.device)

    # 1. Transforms & DataLoader
    eval_transform = build_transforms(is_training=False)
    split_csv = PROJECT_ROOT / "splits" / f"primary_{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv}")

    dataset = RiceLeafDataset(split_csv, root_dir=PROJECT_ROOT, transform=eval_transform, is_training=False)
    loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    print(f"[*] Evaluating {evaluator.model_name} on {args.split.upper()} set ({len(dataset):,} samples)...")

    results = evaluator.evaluate(loader)
    evaluator.save_reports(results)

    om = results["overall_metrics"]
    eff = results["efficiency"]

    print("\n" + "=" * 75)
    print(f" Evaluation Results for [{evaluator.model_name.upper()}] on Internal Test Set:")
    print("=" * 75)
    print(f"  * Accuracy         : {om['accuracy']:.4f} ({om['accuracy']*100:.2f}%)")
    print(f"  * Macro F1         : {om['macro_f1']:.4f}")
    print(f"  * Macro Precision  : {om['macro_precision']:.4f}")
    print(f"  * Macro Recall     : {om['macro_recall']:.4f}")
    print(f"  * Parameters       : {eff['total_parameters']:,}")
    print(f"  * Model Size       : {eff['model_size_mb']:.2f} MB")
    print(f"  * CPU Latency      : {eff['cpu_latency_mean_ms']:.2f} ms (median: {eff['cpu_latency_median_ms']:.2f} ms)")
    print(f"  * Throughput       : {eff['throughput_fps']:.1f} FPS (CPU)")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
