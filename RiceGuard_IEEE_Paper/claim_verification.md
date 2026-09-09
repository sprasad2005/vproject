# RiceGuard IEEE Paper — Strict Claim Verification Audit

This document provides a line-by-line verification of every numerical, architectural, statistical, and dataset claim presented in the manuscript against verified project artifacts.

---

## 🔍 Claim-by-Claim Verification Matrix

| Claim ID | Paper Section | Manuscript Claim | Exact Source File | Source Metric / Row | Verified Value | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C-001** | Abstract / Results | Phase 2B Test Accuracy = 90.12% | `results/reports/phase5_master_results_table.csv` | Phase 2B, Accuracy | `0.9012` | **VERIFIED** |
| **C-002** | Abstract / Results | Phase 2B Macro F1 = 0.8891 | `results/reports/phase2b_final_baseline_summary.json` | test_metrics.macro_f1 | `0.8891` | **VERIFIED** |
| **C-003** | Abstract / Results | Phase 2B Balanced Accuracy = 89.28% | `results/reports/phase5_master_results_table.csv` | Phase 2B, Balanced Acc | `0.8928` | **VERIFIED** |
| **C-004** | Abstract / Results | Phase 2B Parameters = 4,015,234 | `results/reports/phase5_reproducibility_audit.json` | phase2b params | `4015234` | **VERIFIED** |
| **C-005** | Abstract / Results | Phase 2B Model Size = 15.61 MB | `experiments/phase2b_final_baseline/` | checkpoint file size | `15.61 MB` | **VERIFIED** |
| **C-006** | Results | Phase 3 Test Accuracy = 88.75% | `results/reports/phase5_master_results_table.csv` | Phase 3, Accuracy | `0.8875` | **VERIFIED** |
| **C-007** | Results | Phase 3 Macro F1 = 0.8729 | `results/reports/phase3_lesion_aware_summary.json` | test_metrics.macro_f1 | `0.8729` | **VERIFIED** |
| **C-008** | Results | Phase 3 Loc Precision = 0.0223 | `results/reports/phase3_lesion_aware_summary.json` | localization.precision | `0.0223` | **VERIFIED** |
| **C-009** | Results | Phase 3 Loc Recall = 0.0914 | `results/reports/phase3_lesion_aware_summary.json` | localization.recall | `0.0914` | **VERIFIED** |
| **C-010** | Results | Phase 3 Loc F1 = 0.0358 | `results/reports/phase3_lesion_aware_summary.json` | localization.f1 | `0.0358` | **VERIFIED** |
| **C-011** | Results | Phase 3 Mean Matched IoU = 0.6032 | `results/reports/phase3_lesion_aware_summary.json` | localization.mean_matched_iou | `0.6032` | **VERIFIED** |
| **C-012** | Results | Phase 3 Healthy FPR = 21.52% | `results/reports/phase3_lesion_aware_summary.json` | localization.healthy_fpr | `21.52%` | **VERIFIED** |
| **C-013** | Results | Phase 3 Boxes / Image = 8.81 | `results/reports/phase5_effect_size_analysis.csv` | Avg Boxes / Image | `8.81` | **VERIFIED** |
| **C-014** | Abstract / Results | Phase 3B Test Accuracy = 88.96% | `results/reports/phase5_master_results_table.csv` | Phase 3B, Accuracy | `0.8896` | **VERIFIED** |
| **C-015** | Abstract / Results | Phase 3B Macro F1 = 0.8751 | `results/reports/phase2b_vs_phase3_vs_phase3b_comparison.csv` | Macro F1 row | `0.8751` | **VERIFIED** |
| **C-016** | Abstract / Results | Phase 3B Balanced Acc = 87.70% | `results/reports/phase5_master_results_table.csv` | Phase 3B, Balanced Acc | `0.8770` | **VERIFIED** |
| **C-017** | Abstract / Results | Phase 3B Loc Precision = 0.0887 | `results/reports/phase3b_localization_refinement_summary.json` | loc_precision | `0.0887` | **VERIFIED** |
| **C-018** | Abstract / Results | Phase 3B Loc Recall = 0.0924 | `results/reports/phase3b_localization_refinement_summary.json` | loc_recall | `0.0924` | **VERIFIED** |
| **C-019** | Abstract / Results | Phase 3B Loc F1 = 0.0905 | `results/reports/phase3b_localization_refinement_summary.json` | loc_f1 | `0.0905` | **VERIFIED** |
| **C-020** | Abstract / Results | Phase 3B Mean Matched IoU = 0.6423 | `results/reports/phase3b_localization_refinement_summary.json` | mean_matched_iou | `0.6423` | **VERIFIED** |
| **C-021** | Abstract / Results | Phase 3B Healthy FPR = 10.97% | `results/reports/phase3b_localization_refinement_summary.json` | healthy_fpr | `10.97%` | **VERIFIED** |
| **C-022** | Results | Phase 3B Boxes / Image = 2.24 | `results/reports/phase5_effect_size_analysis.csv` | Avg Boxes / Image | `2.24` | **VERIFIED** |
| **C-023** | Methodology | Phase 3B Calibration: $	au=0.60$, Top-3 | `results/reports/phase3b_localization_refinement_summary.json` | optimal_threshold, top_k | `0.60, 3` | **VERIFIED** |
| **C-024** | Results | Phase 3B Parameters = 6,966,151 | `results/reports/phase5_reproducibility_audit.json` | phase3b params | `6966151` | **VERIFIED** |
| **C-025** | Results | Phase 3B Model Size = 26.86 MB | `experiments/phase3b_localization_refinement/` | checkpoint file size | `26.86 MB` | **VERIFIED** |
| **C-026** | Results | Phase 3B GPU Latency = 10.46 ms | `results/reports/phase5_master_results_table.csv` | Phase 3B latency | `10.46 ms` | **VERIFIED** |
| **C-027** | XAI Results | RiceSeg Eligible Samples = 4,348 | `results/reports/phase4_xai_summary.json` | total_evaluated_samples | `4348` | **VERIFIED** |
| **C-028** | XAI Results | Phase 2B All-Sample Energy In = 9.64% | `results/reports/phase4_xai_model_comparison.csv` | all_samples, Phase 2B Energy | `0.0964` | **VERIFIED** |
| **C-029** | XAI Results | Phase 3B All-Sample Energy In = 6.81% | `results/reports/phase4_xai_model_comparison.csv` | all_samples, Phase 3B Energy | `0.0681` | **VERIFIED** |
| **C-030** | XAI Results | Phase 2B All-Sample Attr IoU = 0.0864 | `results/reports/phase4_xai_model_comparison.csv` | all_samples, Phase 2B IoU | `0.0864` | **VERIFIED** |
| **C-031** | XAI Results | Phase 3B All-Sample Attr IoU = 0.0724 | `results/reports/phase4_xai_model_comparison.csv` | all_samples, Phase 3B IoU | `0.0724` | **VERIFIED** |
| **C-032** | XAI Results | Phase 2B Pointing (All) = 29.58% | `results/reports/phase4_xai_model_comparison.csv` | all_samples, Pointing 2B | `0.2958` | **VERIFIED** |
| **C-033** | XAI Results | Phase 3B Pointing (All) = 21.37% | `results/reports/phase4_xai_model_comparison.csv` | all_samples, Pointing 3B | `0.2137` | **VERIFIED** |
| **C-034** | XAI Results | Mutually Correct Samples $N=280$ | `results/reports/phase4_xai_model_comparison.csv` | correct_both stratum count | `280` | **VERIFIED** |
| **C-035** | XAI Results | Pointing (Mutually Correct) = 38.21% vs 36.79% | `results/reports/phase4_xai_model_comparison.csv` | correct_both Pointing | `0.3821 vs 0.3679` | **VERIFIED** |
| **C-036** | XAI Results | Healthy Border Attention: 18.5% to 9.2% | `results/reports/phase5_effect_size_analysis.csv` | Border Attention Ratio | `0.185 vs 0.092` | **VERIFIED** |
| **C-037** | XAI Results | Attribution Entropy: 0.824 to 0.612 | `results/reports/phase4_xai_summary.json` | healthy_entropy | `0.824 vs 0.612` | **VERIFIED** |
| **C-038** | Faithfulness | Deletion AUC: $0.1387 \pm 0.2086$ vs $0.1367 \pm 0.1482$ | `results/reports/phase4_faithfulness_results.csv` | Deletion AUC row | `0.1387 vs 0.1367` | **VERIFIED** |
| **C-039** | Faithfulness | Insertion AUC: $0.2939 \pm 0.2979$ vs $0.1486 \pm 0.1682$ | `results/reports/phase4_faithfulness_results.csv` | Insertion AUC row | `0.2939 vs 0.1486` | **VERIFIED** |
| **C-040** | Statistics | McNemar Classif: $\chi^2=1.18, p=0.2774$ | `results/reports/phase4_statistical_comparison.csv` | classification invariance | `chi2=1.18, p=0.2774` | **VERIFIED** |
| **C-041** | Statistics | Wilcoxon Energy In: $p=5.12 	imes 10^{-285}$ | `results/reports/phase4_statistical_comparison.csv` | Energy In, all samples | `p=5.12e-285` | **VERIFIED** |
| **C-042** | Statistics | Fisher's Exact Healthy FPR: $p < 10^{-6}$ | `results/reports/phase5_effect_size_analysis.csv` | Healthy FP Rate test | `p < 1e-6` | **VERIFIED** |
| **C-043** | Setup | Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4 | `results/reports/phase5_reproducibility_audit.json` | environment block | `Verified` | **VERIFIED** |
| **C-044** | Setup | Random Seed = 42 | `results/reports/phase5_reproducibility_audit.json` | random_seed | `42` | **VERIFIED** |

---

## 🎯 Audit Summary
- **Total Audited Claims**: 44
- **Verified Concordant**: 44 (100%)
- **Unverified / Fabricated Claims**: 0 (0%)
- **Negative Trade-Offs Reported Honestly**: Yes (Phase 2B to Phase 3B Macro F1: 0.8891 to 0.8751, Insertion AUC favored Phase 2B, all-sample XAI overlap reflected domain shift).
- **Governance Integrity**: RiceSeg5932 strictly labeled as XAI ground-truth only; Sethy5932 & RiceLeafDiseaseBD5 labeled as locked external datasets.
