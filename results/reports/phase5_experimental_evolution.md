# Phase 5: Experimental Evolution & Multi-Dimensional Comparison

| Dimension | Metric | Phase 2B Baseline | Phase 3 Multi-Task | Phase 3B Refined |
|---|---|---|---|---|
| Classification | Test Accuracy | 90.12% | 88.75% | 88.96% |
| Classification | Balanced Accuracy | 89.28% | 86.97% | 87.70% |
| Classification | Macro Precision | 0.8871 | 0.8712 | 0.8737 |
| Classification | Macro Recall | 0.8928 | 0.8697 | 0.8770 |
| Classification | Macro F1 | 0.8891 | 0.8729 | 0.8751 |
| Classification | Weighted F1 | 0.9011 | 0.8864 | 0.8901 |
| Localization | Detection Precision | N/A | 0.0223 | 0.0887 |
| Localization | Detection Recall | N/A | 0.0914 | 0.0924 |
| Localization | Detection F1 | N/A | 0.0358 | 0.0905 |
| Localization | Mean Matched IoU | N/A | 0.6032 | 0.6423 |
| Localization | Healthy False Pos. Rate | N/A | 21.52% | 10.97% |
| Localization | Avg Pred Boxes / Image | N/A | 8.81 | 2.24 |
| Efficiency | Total Parameters | 4,015,234 | 6,966,151 | 6,966,151 |
| Efficiency | Model Size | 15.61 MB | 26.86 MB | 26.86 MB |
| Efficiency | GPU Latency (Mean) | 10.36 ms | 13.52 ms | 10.46 ms |
| Efficiency | CPU Latency (Mean) | 34.87 ms | N/A | 48.35 ms |
| Efficiency | Peak VRAM | 814.5 MB | 458.8 MB | 458.8 MB |
| Explainability | Energy Inside Mask (All) | 9.64% | N/A | 6.81% |
| Explainability | Attribution IoU (Top 20%) | 0.0864 | N/A | 0.0724 |
| Explainability | Pointing Game (Mut. Correct) | 36.79% | N/A | 38.21% |
| Explainability | Deletion AUC | 0.1387 | N/A | 0.1367 |
| Explainability | Insertion AUC | 0.2939 | N/A | 0.1486 |
| Explainability | Border Attention (Healthy) | 18.5% | N/A | 9.2% |
| Explainability | Normalized Entropy | 0.824 | N/A | 0.612 |
