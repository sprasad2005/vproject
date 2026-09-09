# Phase 5: Formal Ablation & Evolution Interpretation

## A. Transition 1: Classification-Only to Lesion-Grounded Multi-Task (Phase 2B -> Phase 3)

- **Classification Trade-Off**: Test Macro F1 shifted slightly from `0.8891` to `0.8729` (-0.0162, -1.82% relative drop) as the shared backbone balances classification and spatial grid optimization.
- **New Capability Gained**: Direct spatial lesion localization was introduced from scratch (Mean Matched IoU = `0.6032`), enabling bounding box generation without fine segmentation supervision.
- **Computational Overhead**: Parameters increased from 4.02M to 6.97M (+2.95M) and model size increased from 15.61 MB to 26.86 MB (+11.25 MB). GPU latency remained under 14 ms.

## B. Transition 2: Localization Imbalance Refinement & Calibration (Phase 3 -> Phase 3B)

- **Precision Quadrupled**: Detection precision improved from `0.0223` to `0.0887` (+300% relative improvement) due to positive weight capping (`pos_weight = 10.0`).
- **False Positive Suppression**: Healthy leaf false positive rate dropped from `21.52%` to `10.97%` (-49% reduction).
- **Controlled Density**: Average predicted boxes per image decreased by 75% from `8.81` down to `2.24`.
- **Classification Recovery**: Test Macro F1 recovered from `0.8729` to `0.8751`.

## C. Final Trade-Off Summary (Phase 2B -> Phase 3B)

The multi-task model introduces calibrated spatial localization (`Mean IoU = 0.6423`) and cuts background border attention in half (9.2% vs 18.5%), while trading off 1.40% in Macro F1 relative to the classification-only baseline.
