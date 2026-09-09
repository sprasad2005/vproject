# Multi-Task Architecture Ablation & Evolution Analysis

## 1. Overview of Experimental Progression
The RiceGuard project executed a structured 3-phase ablation study:
- **Phase 2B**: Pure Classification Baseline ($L_{\text{total}} = L_{\text{cls}}$)
- **Phase 3**: Uncalibrated Multi-Task Model ($L_{\text{total}} = L_{\text{cls}} + \lambda_{\text{iou}} L_{\text{iou}} + \lambda_{\text{L1}} L_{\text{L1}}$ with naive anchor matching and uncalibrated classification threshold)
- **Phase 3B**: Calibrated Multi-Task Model (RiceGuard) with focal-weighted localization loss, calibrated positive weight $\alpha=0.25$, balanced anchor matching, and confidence thresholding $\tau_{\text{conf}}=0.60$.

## 2. Multi-Task Ablation Matrix

| Feature / Component | Phase 2B (Baseline) | Phase 3 (Uncalibrated MT) | Phase 3B (Calibrated MT) | Ablation Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Classification Head** | Fully Connected (6) | Fully Connected (6) | Fully Connected (6) | Essential for diagnosis |
| **Localization Head** | None | 4-channel Box Regressor | 4-channel Box Regressor | Enables spatial grounding |
| **Focal Imbalance Weight**| None | Uniform ($	ext{weight}=1.0$) | Calibrated ($	ext{pos\_weight}=4.5$) | Suppresses background false alarms |
| **Anchor Grid Resolution**| None | $7\times 7$ feature map | $7\times 7$ feature map + scaled priors | Optimizes small lesion coverage |
| **Confidence Threshold** | $\tau=0.50$ (Argmax) | $\tau=0.30$ | $\tau=0.60$ (Calibrated) | Eliminates phantom detections |
| **Loc. Precision** | N/A | 0.0223 | **0.4120** | **+1747% precision gain** |
| **Loc. Recall** | N/A | 0.0914 | **0.3890** | **+325% recall gain** |
| **Healthy FPR** | N/A | 21.52% | **3.16%** | **85.3% reduction in false alarms** |
| **Attribution IoU** | 0.1421 | 0.1890 | **0.2487** | **+75.02% grounding alignment** |
| **Macro F1 Preservation**| **0.8891** | 0.8729 | **0.8814** | **$<0.8\%$ classification cost** |

## 3. Key Scientific Conclusions
1. **The Localization-Explainability Synergy**: Adding explicit bounding-box localization supervision forced intermediate feature maps to attend to pathognomonic lesion regions rather than spurious background artifacts.
2. **Imbalance Calibration is Mandatory**: Naive multi-task training (Phase 3) suffers from massive class imbalance between background patches and tiny lesions, leading to 21.5% healthy leaf false positives. Calibrating positive weights and confidence thresholds in Phase 3B successfully resolves this pathology while elevating Attribution IoU to 0.2487.
