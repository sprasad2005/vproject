"""Unit and integration tests for Phase 5 Scientific Consolidation."""

from __future__ import annotations

import hashlib

import pytest

from src.analysis.effect_size import (
    compute_absolute_difference,
    compute_cohens_d_paired,
    compute_rank_biserial_correlation,
    compute_relative_change_percent,
    interpret_cohens_d,
)
from src.data.dataset_registry import (
    is_training_allowed,
    is_xai_ground_truth,
    validate_dataset_usage,
)
from src.data.label_mapping import CANONICAL_PRIMARY_CLASSES
from src.utils.paths import get_project_root


class TestEffectSizeCalculations:
    """Test standardized effect size calculations."""

    def test_absolute_difference(self):
        assert compute_absolute_difference(0.85, 0.90) == pytest.approx(0.05)
        assert compute_absolute_difference(0.90, 0.85) == pytest.approx(-0.05)

    def test_relative_change_percent(self):
        assert compute_relative_change_percent(0.02, 0.08) == pytest.approx(300.0)
        assert compute_relative_change_percent(0.0, 0.0) == 0.0

    def test_cohens_d_paired(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.2, 0.3, 0.4, 0.5, 0.6]  # Constant diff = +0.1, std = 0 -> 0.0
        assert compute_cohens_d_paired(a, b) == 0.0

        # Variable differences
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.3, 0.5, 0.4, 0.7, 0.9]
        d = compute_cohens_d_paired(a, b)
        assert d > 0.8
        assert "Large" in interpret_cohens_d(d)

    def test_rank_biserial_correlation(self):
        # When Wilcoxon stat is 0 (all positive differences), r_rb = 1.0
        r_max = compute_rank_biserial_correlation(wilcoxon_stat=0.0, n_pairs=10)
        assert r_max == pytest.approx(1.0)

        # When Wilcoxon stat equals total rank sum (all negative), r_rb = -1.0
        total_sum = 10 * 11 / 2.0
        r_min = compute_rank_biserial_correlation(wilcoxon_stat=total_sum, n_pairs=10)
        assert r_min == pytest.approx(-1.0)


class TestCanonicalClassOrder:
    """Verify standard 6-class canonical ordering."""

    def test_canonical_classes(self):
        expected = [
            "Healthy",
            "Blast",
            "Brown Spot",
            "Leaf Smut",
            "Tungro",
            "Sheath Blight",
        ]
        assert list(CANONICAL_PRIMARY_CLASSES) == expected
        assert len(CANONICAL_PRIMARY_CLASSES) == 6


class TestArtifactHashingAndPreservation:
    """Verify integrity hashing utility on existing project artifacts."""

    def test_file_hash_computation(self):
        root = get_project_root()
        config_path = root / "configs" / "base.yaml"
        assert config_path.exists()

        with open(config_path, "rb") as f:
            h1 = hashlib.sha256(f.read()).hexdigest()

        with open(config_path, "rb") as f:
            h2 = hashlib.sha256(f.read()).hexdigest()

        assert h1 == h2
        assert len(h1) == 64


class TestDataGovernancePhase5:
    """Verify that external datasets remain locked during Phase 5 consolidation."""

    def test_governance_locks_preserved(self):
        assert is_training_allowed("sethy") is False
        assert is_training_allowed("bd5") is False
        assert is_training_allowed("riceseg") is False
        assert is_xai_ground_truth("riceseg") is True

        with pytest.raises(PermissionError):
            validate_dataset_usage("sethy", "training")

        with pytest.raises(PermissionError):
            validate_dataset_usage("riceseg", "mask_training")
