"""Tests for image integrity checks, SHA-256 hash calculation, and duplicate auditing."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.data.quality_checks import (
    DuplicateAuditor,
    calculate_sha256,
    compute_dhash,
    hamming_distance,
    validate_image_file,
)


def test_calculate_sha256_bytes():
    """Verify SHA-256 hash computation on known byte string."""
    data = b"RiceGuard Research Foundation"
    hash_str = calculate_sha256(data)
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64


def test_dhash_and_hamming_distance():
    """Verify perceptual difference hashing and distance computation."""
    img1 = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    img2 = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))

    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0


def test_duplicate_auditor_exact():
    """Verify DuplicateAuditor identifies exact duplicate hashes."""
    auditor = DuplicateAuditor()
    auditor.add_record("img_1", "Primary", "Blast", "hash_abc", "dhash_123", "path/1.jpg")
    auditor.add_record("img_2", "Primary", "Blast", "hash_abc", "dhash_123", "path/2.jpg")
    auditor.add_record("img_3", "Sethy", "Tungro", "hash_xyz", "dhash_456", "path/3.jpg")

    exact_groups = auditor.find_exact_duplicates()
    assert "hash_abc" in exact_groups
    assert len(exact_groups["hash_abc"]) == 2
    assert "hash_xyz" not in exact_groups


def test_validate_nonexistent_image():
    """Verify validate_image_file handles missing files gracefully."""
    res = validate_image_file("nonexistent_path_xyz.jpg")
    assert res["is_valid"] is False
    assert "does not exist" in res["error_message"]
