# RiceSeg5932 Dataset Alignment & Governance Audit Report

**Phase**: Phase 4 Quantitative XAI Validation  
**Date**: 2026-09-09 13:34:52  
**Governance Status**: `xai_ground_truth_only` (Strictly isolated from training/tuning)  

## 1. Summary Statistics

- **Total RiceSeg Image-Mask Pairs**: 5932
- **Eligible Evaluation Samples**: 4348
- **Excluded Samples**: 1584
- **Image-Mask Spatial Pairing Success**: 100%

## 2. Eligible Canonical Class Distribution

| Canonical Class | Eligible Samples | Percentage |
|---|---|---|
| Healthy | 0 | 0.00% |
| Blast | 1440 | 33.12% |
| Brown Spot | 1600 | 36.80% |
| Leaf Smut | 0 | 0.00% |
| Tungro | 1308 | 30.08% |
| Sheath Blight | 0 | 0.00% |

## 3. Exclusion Categories & Governance Audit

| Exclusion Category / Reason | Count |
|---|---|
| Class 'Bacterial Blight' is not among 6 canonical primary classes. | 1584 |

> [!NOTE]
> Bacterial Blight (1,584 samples) is an external non-canonical disease class not included in the primary 6-class taxonomy, and is excluded without label manipulation.
