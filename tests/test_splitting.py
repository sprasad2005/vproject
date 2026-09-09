"""Tests for stratified primary dataset splitting and leakage prevention."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splitting import PrimaryDatasetSplitter


def test_primary_dataset_splitter_ratios(tmp_path):
    """Verify splitter distributes data according to specified ratios."""
    # Create synthetic manifest DataFrame
    records = []
    classes = ["Healthy", "Blast", "Brown Spot", "Leaf Smut", "Tungro", "Sheath Blight"]
    for i in range(300):
        c = classes[i % len(classes)]
        records.append(
            {
                "image_id": f"img_{i}",
                "image_path": f"data/{i}.jpg",
                "canonical_class": c,
                "dataset": "RiceLeafDiseaseBD",
                "file_hash": f"hash_{i}",
            }
        )
    df = pd.DataFrame(records)

    splitter = PrimaryDatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, root_dir=tmp_path)
    train_df, val_df, test_df, meta = splitter.split(df)

    assert len(train_df) + len(val_df) + len(test_df) == 300
    assert abs(len(train_df) - 210) <= 6
    assert abs(len(val_df) - 45) <= 6
    assert abs(len(test_df) - 45) <= 6
    assert meta["leakage_verification"]["exact_duplicate_leakage_passed"] is True


def test_splitter_invalid_ratios():
    """Verify assertion when split ratios do not sum to 1.0."""
    with pytest.raises(AssertionError):
        PrimaryDatasetSplitter(train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)
