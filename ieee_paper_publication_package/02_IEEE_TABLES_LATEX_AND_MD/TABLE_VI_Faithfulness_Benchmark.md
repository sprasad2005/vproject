# TABLE VI: Explainability Faithfulness Benchmark (Deletion & Insertion AUC)

| Model Architecture | Deletion AUC ($\downarrow$ lower is better) | Insertion AUC ($\uparrow$ higher is better) | Faithfulness Score ($\text{AUC}_{\text{ins}} - \text{AUC}_{\text{del}}$) |
| :--- | :--- | :--- | :--- |
| **Phase 2B Baseline** | 0.3842 | 0.6128 | 0.2286 |
| **Phase 3 Multi-Task** | 0.3210 | 0.6740 | 0.3530 |
| **Phase 3B Proposed (RiceGuard)** | **0.2451** | **0.7684** | **0.5233 (+128.9% gain)** |
