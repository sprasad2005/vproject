# TABLE I: Architectural Backbone Screening (Phase 2A)

| Architecture | Parameters (M) | VRAM Footprint (MB) | Epoch Time (s) | Val Macro F1 | Test Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | 25.56 | 1480 | 42.1 | 0.8654 | 0.8521 | Evaluated Baseline |
| **ViT-B/16** | 86.57 | 3210 | 118.4 | 0.8420 | 0.8305 | Resource Prohibitive |
| **EfficientNet-B0** | **5.29** | **890** | **24.6** | **0.8987** | **0.8891** | **Selected Backbone** |
