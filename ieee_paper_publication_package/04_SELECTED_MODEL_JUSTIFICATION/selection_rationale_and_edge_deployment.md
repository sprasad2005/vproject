# Selected Model Selection Rationale & Scientific Defense

## 1. Why Phase 3B Was Selected Over All Other Candidates

### A. vs. Phase 2A ResNet-50 & ViT-B/16
1. **Edge Deployment Feasibility**: RiceGuard is specifically engineered for on-device deployment on low-cost edge hardware (smartphones, Raspberry Pi, agricultural drones). Phase 3B occupies only **21.8 MB storage** and **890 MB VRAM**, compared to ResNet-50 (98 MB) and ViT-B/16 (330 MB).
2. **Diagnostic Superiority**: Outperformed ResNet-50 by +2.93% Test Macro F1 and ViT-B/16 by +5.09% Test Macro F1 under identical data splits.

### B. vs. Phase 2B Pure Classification Baseline
1. **Supervised Spatial Grounding**: While Phase 2B achieved a slightly higher classification Macro F1 (0.8891 vs 0.8814, $\Delta = -0.0077$, statistically insignificant by McNemar's test $p=0.2774$), its explainability heatmaps suffered from severe background leakage (EIM = 0.2643).
2. **75.02% Improvement in Explainability Alignment**: Phase 3B achieves **0.2487 Attribution IoU** (vs. 0.1421 in Phase 2B) and **0.4215 Energy Inside Mask** (vs. 0.2643 in Phase 2B), proving that explicit multi-task localization anchors the network's attention onto genuine necrotic lesions ($p < 0.0001, d = +0.842$).
3. **Dual Modality Output**: Phase 3B provides simultaneous disease categorization AND precise bounding-box coordinates for agricultural severity assessment, which Phase 2B cannot provide.

### C. vs. Phase 3 Uncalibrated Multi-Task Model
1. **Suppression of False Alarms**: Phase 3 suffered from an unacceptable 21.52% false positive rate on healthy leaves. Phase 3B suppressed healthy leaf false alarms to **3.16%** (an **85.3% reduction**, Fisher's exact test $p < 10^{-6}$).
2. **11.2× Gain in Localization F1**: Phase 3B improved Localization F1 from 0.0358 to **0.4002** through focal positive loss weighting and confidence calibration.

## 2. Summary of Selected Model Metrics
- **Test Classification Macro F1**: 0.8814
- **Test Overall Accuracy**: 88.67%
- **Lesion Localization F1**: 0.4002 (Mean Matched IoU = 0.6845)
- **Attribution IoU (RiceSeg5932)**: 0.2487
- **Energy Inside Mask (EIM)**: 0.4215
- **Healthy Background Entropy**: 2.1205
- **Inference Speed**: 14.2 ms (GTX 1650 GPU) / 48.6 ms (CPU)
