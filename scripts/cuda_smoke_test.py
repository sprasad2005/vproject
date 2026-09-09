"""CUDA Training Pipeline Smoke Test for RiceGuard.

Verifies end-to-end GPU training capability on NVIDIA GTX 1650:
- Device placement (cuda:0)
- Model instantiation (EfficientNet-B0)
- Forward pass with AMP autocast
- Loss calculation (CrossEntropyLoss)
- Scaled backward pass
- Optimizer step
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import build_model
from src.training.optimizer import build_optimizer


def run_smoke_test() -> bool:
    print("=" * 80)
    print(" RiceGuard CUDA Training Pipeline Smoke Test")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available in current PyTorch installation.")
        return False

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print(f"[*] Selected Device: {device} ({gpu_name}, {vram_mb:.1f} MB VRAM)")

    # 1. Build Model on CUDA
    print("[*] Instantiating EfficientNet-B0 on CUDA...")
    model = build_model("efficientnet_b0", num_classes=6, pretrained=False).to(device)
    model.train()
    print(f"[*] Model device verification: {next(model.parameters()).device}")

    # 2. Synthetic Batch on CUDA
    batch_size = 8
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 6, (batch_size,), device=device)
    print(f"[*] Input tensor device: {inputs.device}, shape: {inputs.shape}")
    print(f"[*] Target tensor device: {targets.device}, shape: {targets.shape}")

    # 3. Setup Optimizer and AMP Scaler
    dummy_cfg = {
        "optimizer": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
        "training": {"amp": True},
    }
    optimizer = build_optimizer(model, dummy_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.CrossEntropyLoss()

    # 4. Forward Pass with AMP
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    print(f"[*] Output tensor device: {outputs.device}, shape: {outputs.shape}")
    print(f"[*] Loss successfully calculated: {loss.item():.4f}")

    # 5. Backward Pass
    scaler.scale(loss).backward()
    print("[*] Backward pass successful (gradients computed).")

    # 6. Optimizer Step
    scaler.step(optimizer)
    scaler.update()
    print("[*] Optimizer step successful.")

    # 7. Memory Tracking
    alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"[*] GPU Memory Allocated: {alloc_mb:.2f} MB | Peak: {peak_mb:.2f} MB")
    print("=" * 80)
    print(" CUDA SMOKE TEST RESULT: PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
