# TABLE II: Master Experimental Evolution (Phases 2B, 3, 3B)

| Phase | Model Architecture | Macro F1 | Accuracy | Loc. Prec. | Loc. Recall | Loc. F1 | Healthy FPR | Mean IoU | Attr. IoU | EIM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 2B** | EfficientNet-B0 Baseline | **0.8891** | **0.8942** | — | — | — | — | — | 0.1421 | 0.2643 |
| **Phase 3** | Uncalibrated Multi-Task | 0.8729 | 0.8783 | 0.0223 | 0.0914 | 0.0358 | 21.52% | 0.6032 | 0.1890 | 0.3312 |
| **Phase 3B** | **RiceGuard Refined (Selected)** | 0.8814 | 0.8867 | **0.4120** | **0.3890** | **0.4002** | **3.16%** | **0.6845** | **0.2487** | **0.4215** |
