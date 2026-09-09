# TABLE IV: Quantitative Explainability (XAI) Grounding Benchmark (on RiceSeg5932)

| Explainability Grounding Metric | Phase 2B Baseline | Phase 3B Proposed | Delta | Relative Gain | Significance ($p$) | Cohen's $d$ Effect Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Attribution IoU (mean)** | 0.1421 | **0.2487** | +0.1066 | **+75.02%** | $p < 0.0001$ | $d = +0.842$ (Large) |
| **Energy Inside Mask (EIM)** | 0.2643 | **0.4215** | +0.1572 | **+59.48%** | $p < 0.0001$ | $d = +0.915$ (Large) |
| **Pointing Game Accuracy** | 0.5824 | **0.7641** | +0.1817 | **+31.20%** | $p < 0.001$ | $d = +0.780$ (Medium-Large) |
| **Healthy Background Entropy** | 3.8412 | **2.1205** | -1.7207 | **-44.80%** | $p < 0.0001$ | $d = -1.042$ (Large) |
