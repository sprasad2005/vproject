# Section IV: Results & Statistical Discussion

## A. Architectural Backbone Screening Analysis
As detailed in **Table I**, EfficientNet-B0 outperformed both ResNet-50 and ViT-B/16 across validation and test Macro F1-scores. ResNet-50 required 1.66× higher VRAM and 1.71× longer training epochs, while ViT-B/16 suffered from over-parameterization and lack of spatial inductive bias, consuming 3,210 MB VRAM with a subpar test Macro F1 of 0.8305. EfficientNet-B0 established the optimal trade-off with a 5.29M parameter footprint, 890 MB VRAM usage, and a test Macro F1 of 0.8891.

## B. Master Architecture Evolution
As detailed in **Table II**, transitioning from pure classification (Phase 2B) to uncalibrated multi-task learning (Phase 3) caused severe false-alarm over-prediction (21.52% false positive rate on healthy leaves) due to severe foreground-background lesion imbalance. 

In Phase 3B (RiceGuard), focal positive weighting and confidence calibration resolved this imbalance, suppressing healthy false alarms to **3.16%** ($p < 10^{-6}$) and driving Localization F1 from 0.0358 to **0.4002** (Mean Matched IoU = 0.6845). Importantly, this spatial grounding was achieved with negligible classification degradation (0.8814 vs. 0.8891 Macro F1), confirmed to be statistically invariant by McNemar's test ($\chi^2 = 1.18, p = 0.2774$).

## C. Per-Class Diagnostic Performance
As shown in **Table III**, RiceGuard exhibits robust classification across all 6 pathology categories:
- **Healthy**: Precision = 0.9828, Recall = 0.9661, F1 = **0.9744**
- **Leaf Smut**: Precision = 0.9009, Recall = 0.9174, F1 = **0.9091**
- **Blast**: Precision = 0.8621, Recall = 0.8772, F1 = **0.8696**
- **Tungro**: Precision = 0.8696, Recall = 0.8451, F1 = **0.8571**
- **Brown Spot**: Precision = 0.8594, Recall = 0.8333, F1 = **0.8462**
- **Sheath Blight**: Precision = 0.8306, Recall = 0.8333, F1 = **0.8320**
