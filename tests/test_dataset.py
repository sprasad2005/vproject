"""Tests for PyTorch dataset and DataLoader pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import RiceLeafDataset, create_dataloader


@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Create a mock manifest and image directory for testing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    classes = ["Healthy", "Blast", "Brown Spot", "Leaf Smut", "Tungro", "Sheath Blight"]

    for i in range(12):
        c = classes[i % len(classes)]
        img_file = img_dir / f"sample_{i}.jpg"
        # Save a 100x100 dummy RGB image
        img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))
        img.save(img_file)

        records.append(
            {
                "image_id": f"sample_{i}",
                "image_path": str(img_file),
                "canonical_class": c,
                "dataset": "RiceLeafDiseaseBD",
                "split": "train" if i < 8 else "val",
            }
        )

    manifest_file = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_file, index=False)
    return manifest_file, tmp_path


def test_rice_leaf_dataset_loading(dummy_dataset_dir):
    """Verify RiceLeafDataset loads samples and returns proper tensor and metadata."""
    manifest_file, root = dummy_dataset_dir
    dataset = RiceLeafDataset(manifest_file, root_dir=root, is_training=True)

    assert len(dataset) == 12
    img_tensor, target, meta = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(target, int)
    assert 0 <= target <= 5
    assert "image_id" in meta
    assert "canonical_class" in meta


def test_rice_leaf_dataset_split_filtering(dummy_dataset_dir):
    """Verify split filtering in RiceLeafDataset."""
    manifest_file, root = dummy_dataset_dir
    train_ds = RiceLeafDataset(manifest_file, root_dir=root, split_name="train", is_training=True)
    val_ds = RiceLeafDataset(manifest_file, root_dir=root, split_name="val", is_training=False)

    assert len(train_ds) == 8
    assert len(val_ds) == 4


def test_dataloader_batch_generation(dummy_dataset_dir):
    """Verify custom DataLoader collate function."""
    manifest_file, root = dummy_dataset_dir
    dataset = RiceLeafDataset(manifest_file, root_dir=root)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)

    for images, targets, metadata in loader:
        assert images.shape == (4, 3, 100, 100)
        assert targets.shape == (4,)
        assert len(metadata) == 4
        break


def test_dataset_external_training_prohibition(tmp_path):
    """Verify RiceLeafDataset blocks training on external datasets."""
    records = [
        {
            "image_id": "ext_1",
            "image_path": "path/1.jpg",
            "canonical_class": "Blast",
            "dataset": "Sethy5932",
        }
    ]
    manifest_file = tmp_path / "ext_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_file, index=False)

    with pytest.raises(PermissionError, match="Training is strictly prohibited"):
        RiceLeafDataset(manifest_file, root_dir=tmp_path, is_training=True)
