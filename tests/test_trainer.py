"""Tests for Trainer execution, early stopping, and checkpointing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.model_factory import build_model
from src.training.trainer import Trainer


class DummyDataset(Dataset):
    def __init__(self, size: int = 16):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        img = torch.randn(3, 224, 224)
        target = idx % 6
        metadata = {"image_id": f"img_{idx}", "canonical_class": "Healthy"}
        return img, target, metadata


def test_trainer_single_epoch(tmp_path):
    """Verify Trainer can execute training and validation steps and save checkpoints."""
    train_ds = DummyDataset(size=12)
    val_ds = DummyDataset(size=6)

    def collate(batch):
        return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), [b[2] for b in batch]

    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

    model = build_model("efficientnet_b0", num_classes=6, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cfg = {
        "training": {"epochs": 2, "early_stopping_patience": 5},
        "reproducibility": {"seed": 42},
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg,
        experiment_dir=tmp_path / "exp_test",
        model_name="test_model",
        device="cpu",
    )

    results = trainer.train()

    assert results["total_epochs_trained"] == 2
    assert (tmp_path / "exp_test" / "checkpoints" / "best_model.pt").exists()
    assert (tmp_path / "exp_test" / "checkpoints" / "last_model.pt").exists()
    assert (tmp_path / "exp_test" / "metrics" / "training_history.json").exists()
