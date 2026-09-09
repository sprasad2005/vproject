# Data Splits Architecture

This directory contains split definitions, partition manifests, and cross-validation folds.

## Strict Leakage-Free Splitting Policy

1. **Original Image Partitioning**: Splitting is strictly performed on original raw image identifiers before any augmentation or synthesis is applied.
2. **Stratification**: All internal splits (`train.csv`, `val.csv`, `test.csv`) maintain class balance across all 6 target classes.
3. **Partition Proportions**:
   - Training: 70%
   - Validation / Calibration: 15%
   - Internal Test: 15%
4. **External Isolation**: External datasets (`Sethy5932`, `RiceLeafDiseaseBD5`) are stored as independent benchmark manifests and are never included in training or validation splits.
5. **Determinism**: All split manifests record the generator seed (default: `42`), creation timestamp, and SHA-256 hashes of image lists.
