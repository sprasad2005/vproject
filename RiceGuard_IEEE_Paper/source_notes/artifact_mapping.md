# RiceGuard Scientific Artifact Traceability Mapping

This document provides full scientific provenance linking every claim, metric, and figure in the IEEE research paper directly to immutable research artifacts stored in the RiceGuard repository.

---

## 1. Manuscript Metadata & Architecture
| Paper Section / Element | Source Repository Artifact | Exact Verified Metric / Value |
| :--- | :--- | :--- |
| **Title & Methodology** | `src/models/multitask.py` | `RiceMultiTaskEfficientNet(backbone="efficientnet_b0", num_classes=6)` |
| **Backbone Screening** | `results/reports/phase2a_screening_comparison.csv` | EfficientNet-B0 (Val F1: 0.8653, Size: 15.61 MB) vs ResNet-50 (0.8642, 90.25 MB) vs ViT-B/16 (0.8512, 328.40 MB) |
| **Phase 2B Baseline** | `results/reports/phase2b_final_baseline_summary.json` | Accuracy: **90.12%**, Macro F1: **0.8891**, Balanced Acc: **89.28%**, Latency: **10.36 ms**, Params: **4,015,234** |
| **Phase 3 Multi-Task** | `results/reports/phase3_lesion_aware_summary.json` | Accuracy: **88.75%**, Macro F1: **0.8729**, Loc F1: **0.0358**, Mean IoU: **0.6032**, Healthy FPR: **21.52%**, Boxes/img: **8.81** |
| **Phase 3B Calibrated Multi-Task** | `results/reports/phase3b_localization_refinement_summary.json` | Accuracy: **88.96%**, Macro F1: **0.8751**, Balanced Acc: **87.70%**, Loc Prec: **0.0887**, Loc Recall: **0.0924**, Loc F1: **0.0905**, Mean Matched IoU: **0.6423**, Healthy FPR: **10.97%**, Boxes/img: **2.24**, Latency: **10.46 ms**, Params: **6,966,151** |
| **Frozen Calibration Config** | `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/calibration/selected_decoding_config.json` | `confidence_threshold: 0.60`, `top_k: 3` |

---

## 2. Dataset Partitioning & Governance
| Dataset Entity | Source Artifact / Manifest | Verified Statistics | Governance Access Role |
| :--- | :--- | :--- | :--- |
| **Primary Benchmark** | `splits/primary_split_metadata.json` | 7,060 images total across 6 canonical classes (Train: 2,824, Val: 706, Test: 706) | Primary training, validation, and testing |
| **RiceSeg5932** | `data/processed/manifests/riceseg_manifest.csv` | 5,932 raw pairs; **4,348 eligible canonical samples** (Blast: 1,440, Brown Spot: 1,600, Tungro: 1,308) | `xai_ground_truth_only` (Zero training access) |
| **Sethy5932** | `data/processed/manifests/sethy_manifest.csv` | 5,932 images across 4 classes | Locked external domain evaluation |
| **RiceLeafDiseaseBD5**| `data/processed/manifests/bd5_manifest.csv` | 5,932 images across 5 classes | Locked external domain evaluation |

---

## 3. Quantitative Explainability (XAI) & Faithfulness
| Metric / Stratum | Source Artifact | Phase 2B Baseline | Phase 3B Refined | Statistical Test & Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Energy in Mask (All $N=4,348$)** | `results/reports/phase4_xai_model_comparison.csv` | **9.64%** | 6.81% | Paired Wilcoxon: $W=1.74\times 10^6, p=5.12\times 10^{-285}$ |
| **Attribution IoU (All $N=4,348$)** | `results/reports/phase4_xai_model_comparison.csv` | **0.0864** | 0.0724 | Paired Wilcoxon: $W=1.96\times 10^6, p=9.59\times 10^{-95}$ |
| **Pointing Game (All $N=4,348$)** | `results/reports/phase4_xai_model_comparison.csv` | **29.58%** | 21.37% | McNemar Test: $p < 10^{-20}$ |
| **Energy in Mask (Mutually Correct $N=280$)** | `results/reports/phase4_xai_model_comparison.csv` | **9.73%** | 8.13% | Paired Wilcoxon: $W=2,724, p=7.88\times 10^{-36}$ |
| **Attribution IoU (Mutually Correct $N=280$)** | `results/reports/phase4_xai_model_comparison.csv` | **0.0921** | 0.0864 | Paired Wilcoxon: $p=3.23\times 10^{-6}$ |
| **Pointing Game (Mutually Correct $N=280$)** | `results/reports/phase4_xai_model_comparison.csv` | 36.79% | **38.21%** | McNemar Test: $\chi^2=0.155, p=0.694$ (Directional, Not Sig.) |
| **Pointing Game (3B Correct Only $N=468$)** | `results/reports/phase4_xai_model_comparison.csv` | 32.69% | **36.32%** | McNemar Test: $\chi^2=1.984, p=0.159$ (Directional, Not Sig.) |
| **Healthy Border Attention** | `results/reports/phase4_xai_summary.json` | 18.5% | **9.2%** | **50.3% relative reduction** |
| **Attribution Entropy** | `results/reports/phase4_xai_summary.json` | 0.824 | **0.612** | **25.7% relative concentration** |
| **Deletion AUC ($\downarrow$)** | `results/reports/phase4_faithfulness_results.csv` | $0.1387 \pm 0.2086$ | $\mathbf{0.1367 \pm 0.1482}$ | $p=0.187$ (Not Significant) |
| **Insertion AUC ($\uparrow$)** | `results/reports/phase4_faithfulness_results.csv` | $\mathbf{0.2939 \pm 0.2979}$ | $0.1486 \pm 0.1682$ | $p=3.58\times 10^{-9}$ (Significant) |

---

## 4. Figures & Scientific Visualizations
| Figure Label | Paper Title | Underlying Artifact / Generator Script | Verified Purpose |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | Model Evolution & Architecture | `results/figures/phase5/model_evolution_overview.png` | Illustrates shared EfficientNet-B0 backbone bifurcation into classification and spatial localization heads. |
| **Fig. 2** | Classification vs. Localization Trade-off | `results/figures/phase5/classification_localization_tradeoff.png` | Quantifies Pareto frontier across Macro F1, Loc F1, Mean IoU, and Latency. |
| **Fig. 3** | Quantitative XAI Grounding Summary | `results/figures/phase5/xai_grounding_summary.png` | Shows four-panel metric distribution: EIM, Attribution IoU, Pointing Game, and Healthy Border Attention. |

---

## 5. Statistical Hypothesis Testing
| Hypothesis / Test | Metric / Comparison | Verified Statistic | Exact $p$-value | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Classification Invariance** | Phase 2B vs Phase 3B | $\chi^2 = 1.18$ | $p = 0.2774$ | Fail to Reject $H_0$ (Invariance supported) |
| **Healthy FPR Reduction** | Phase 3 vs Phase 3B | $\text{Odds Ratio} = 0.118$ | $p < 10^{-6}$ | Reject $H_0$ (Highly Significant) |
| **Energy in Mask (All)** | Phase 2B vs Phase 3B | $W = 1.74 \times 10^6$ | $p = 5.12 \times 10^{-285}$ | Reject $H_0$ (Highly Significant) |
| **Attribution IoU (All)** | Phase 2B vs Phase 3B | $W = 1.96 \times 10^6$ | $p = 9.59 \times 10^{-95}$ | Reject $H_0$ (Highly Significant) |
