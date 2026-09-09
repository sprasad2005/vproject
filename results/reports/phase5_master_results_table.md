# Master Experimental Comparison Table

| Model | Supervision Type | Accuracy | Macro F1 | Localization F1 | Mean IoU | Pointing Acc | Border Attn | Parameters | Model Size | GPU Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Phase 2B Baseline (Classification Only)** | Global Class Labels Only | 90.12% | 0.8891 | N/A | N/A | 29.58% (36.79% corr) | 18.5% | 4,015,234 | 15.61 MB | 10.36 ms |
| **Phase 3 Multi-Task (Initial Localization)** | Class Labels + BBoxes | 88.75% | 0.8729 | 0.0358 | 0.6032 | N/A — superseded by 3B | N/A | 6,966,151 | 26.86 MB | 13.52 ms |
| **Phase 3B Refined Multi-Task (Calibrated)** | Class Labels + BBoxes (Calibrated) | 88.96% | 0.8751 | 0.0905 | 0.6423 | 21.37% (38.21% corr) | 9.2% | 6,966,151 | 26.86 MB | 10.46 ms |
