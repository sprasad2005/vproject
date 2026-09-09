# Phase 5: Scientific Reproducibility & Artifact Integrity Audit

**Audit Timestamp**: 2026-09-09 14:15:31  
**Audit Verdict**: `PASSED_FULLY_REPRODUCIBLE`  

## 1. Hardware & Software Environment

| Component | Verified Specification |
|---|---|
| **Python** | `3.11.9` |
| **PyTorch** | `2.6.0+cu124` |
| **CUDA / cuDNN** | `12.4` / `90100` |
| **Target GPU** | `NVIDIA GeForce GTX 1650` |
| **OS Platform** | `Windows 10 (10.0.26200)` |
| **Random Seed** | `42` |

## 2. Immutable Artifact Integrity Table

| Artifact Path | Size (KB) | SHA-256 Hash | Status |
|---|---|---|---|
| `results/reports/phase2a_screening_comparison.csv` | 0.3 KB | `525fbdf6a8e84d0c...` | `VERIFIED INTACT` |
| `results/reports/phase2a_screening_summary.json` | 2.0 KB | `9e6cc288107a6beb...` | `VERIFIED INTACT` |
| `experiments/phase2b_final_baseline/efficientnet_b0/checkpoints/best_model.pt` | 47532.7 KB | `9f819b9b134d1058...` | `VERIFIED INTACT` |
| `experiments/phase2b_final_baseline/efficientnet_b0/training_history.csv` | 2.0 KB | `f4ca6351bfeb20ae...` | `VERIFIED INTACT` |
| `results/reports/phase2b_final_baseline_summary.json` | 2.4 KB | `b8b745d56932fc44...` | `VERIFIED INTACT` |
| `results/reports/phase2b_final_baseline_summary.md` | 3.0 KB | `f36551d4eeb79b50...` | `VERIFIED INTACT` |
| `experiments/phase3_lesion_aware/efficientnet_b0_multitask/checkpoints/best_model.pt` | 82114.0 KB | `df38e81544506a5e...` | `VERIFIED INTACT` |
| `experiments/phase3_lesion_aware/efficientnet_b0_multitask/training_history.csv` | 1.6 KB | `2350a86d3bf7b82a...` | `VERIFIED INTACT` |
| `results/reports/phase3_lesion_aware_summary.json` | 5.6 KB | `f84e9a6afd1bbefe...` | `VERIFIED INTACT` |
| `results/reports/phase3_lesion_aware_summary.md` | 3.9 KB | `fe4490539509774c...` | `VERIFIED INTACT` |
| `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/checkpoints/best_model.pt` | 27510.5 KB | `1569b1c833068b5b...` | `VERIFIED INTACT` |
| `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/training_history.csv` | 2.6 KB | `82ba147e404f1f70...` | `VERIFIED INTACT` |
| `results/reports/phase3b_localization_refinement_summary.json` | 5.5 KB | `19b4085eeb77a5cd...` | `VERIFIED INTACT` |
| `results/reports/phase3b_localization_refinement_summary.md` | 4.1 KB | `e4a5acb5da70def0...` | `VERIFIED INTACT` |
| `results/reports/phase2b_vs_phase3_vs_phase3b_comparison.csv` | 0.8 KB | `52e21181005ee500...` | `VERIFIED INTACT` |
| `experiments/phase4_xai/phase4_per_sample_grounding.csv` | 2132.5 KB | `147c3b197eb061c9...` | `VERIFIED INTACT` |
| `results/reports/phase4_xai_summary.json` | 11.4 KB | `91609ae642ea8973...` | `VERIFIED INTACT` |
| `results/reports/phase4_xai_summary.md` | 3.1 KB | `02595c85b70aa11a...` | `VERIFIED INTACT` |
| `results/reports/phase4_xai_model_comparison.csv` | 1.7 KB | `abfac8ba3b334124...` | `VERIFIED INTACT` |
| `results/reports/phase4_statistical_comparison.csv` | 2.0 KB | `040c4d153f399199...` | `VERIFIED INTACT` |
| `results/reports/phase4_faithfulness_results.csv` | 0.4 KB | `ac3b6a216c0e294c...` | `VERIFIED INTACT` |
| `results/reports/phase4_per_class_xai_metrics.csv` | 1.0 KB | `a25d870bea4f6de2...` | `VERIFIED INTACT` |
| `results/reports/phase4_dataset_alignment_report.json` | 0.5 KB | `2396b8f3d6cf512a...` | `VERIFIED INTACT` |

## 3. Data Governance Verification

- **Primary Dataset (`RiceLeafDiseaseBD`)**: Approved for training, validation, testing, and bounding-box localization.
- **External Benchmark (`Sethy5932`)**: Strictly locked (`training_allowed=False`).
- **External Field Benchmark (`RiceLeafDiseaseBD5`)**: Strictly locked (`training_allowed=False`).
- **XAI Benchmark (`RiceSeg5932`)**: Strictly locked (`xai_ground_truth_only`). Never used for training, validation, checkpoint selection, or threshold tuning.
