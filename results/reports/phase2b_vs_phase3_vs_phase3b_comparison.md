# Phase 2B vs. Phase 3 vs. Phase 3B Comparison Report

| Metric | Phase 2B Baseline | Phase 3 Multi-Task | Phase 3B Refined | Phase 3 $\rightarrow$ 3B Delta | Phase 2B $\rightarrow$ 3B Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test Accuracy** | 90.12% | 88.75% | 88.96% | +0.21% | -1.16% |
| **Test Macro F1** | 0.8891 | 0.8729 | 0.8751 | +0.0022 | -0.0140 |
| **Balanced Accuracy** | 89.28% | 86.97% | 87.70% | +0.73% | -1.58% |
| **Localization Precision** | N/A | 0.0223 | 0.0887 | +0.0664 | New Capability |
| **Localization Recall** | N/A | 0.0914 | 0.0924 | +0.0010 | New Capability |
| **Localization F1** | N/A | 0.0358 | 0.0905 | +0.0547 | New Capability |
| **Mean Matched IoU** | N/A | 0.6032 | 0.6423 | +0.0391 | New Capability |
| **Healthy FP Rate** | N/A | 21.52% | 10.97% | -10.55% | Safe Backgrounding |
| **Avg Pred Boxes/Image** | N/A | 8.81 | 2.24 | -6.57 | Reduced overprediction |
| **Parameters** | 4,015,234 | 6,966,151 | 6,966,151 | 0 | +2,950,917 |
| **Model Size (MB)** | 15.61 | 26.86 | 26.86 | 0.00 MB | +11.25 MB |
| **GPU Latency (ms)** | 11.20 | 13.52 | 10.46 | -3.06 ms | +-0.74 ms |
