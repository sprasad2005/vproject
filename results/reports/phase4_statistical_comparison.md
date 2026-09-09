# Phase 4: Statistical Significance Analysis

| Stratum | Metric | Test Type | Statistic | p-value | Mean Difference | Significant (alpha=0.05)? |
|---|---|---|---|---|---|---|
| `all_samples` | Energy Inside Mask | Paired Wilcoxon Signed-Rank | 1740917.0000 | `5.12e-285` | -0.0283 | **YES** (p < 0.05) |
| `all_samples` | Attribution IoU (Top 20%) | Paired Wilcoxon Signed-Rank | 1955501.0000 | `9.59e-95` | -0.0140 | **YES** (p < 0.05) |
| `all_samples` | Pointing Game Accuracy | McNemar Test | 131.0610 | `0.00e+00` | -0.0821 | **YES** (p < 0.05) |
| `correct_2b_only` | Energy Inside Mask | Paired Wilcoxon Signed-Rank | 24007.0000 | `1.03e-146` | -0.0484 | **YES** (p < 0.05) |
| `correct_2b_only` | Attribution IoU (Top 20%) | Paired Wilcoxon Signed-Rank | 56303.0000 | `1.46e-95` | -0.0341 | **YES** (p < 0.05) |
| `correct_2b_only` | Pointing Game Accuracy | McNemar Test | 62.5633 | `2.55e-15` | -0.1302 | **YES** (p < 0.05) |
| `correct_3b_only` | Energy Inside Mask | Paired Wilcoxon Signed-Rank | 29710.0000 | `8.26e-18` | -0.0129 | **YES** (p < 0.05) |
| `correct_3b_only` | Attribution IoU (Top 20%) | Paired Wilcoxon Signed-Rank | 36496.0000 | `2.33e-02` | -0.0030 | **YES** (p < 0.05) |
| `correct_3b_only` | Pointing Game Accuracy | McNemar Test | 1.9845 | `1.59e-01` | +0.0363 | NO |
| `correct_both` | Energy Inside Mask | Paired Wilcoxon Signed-Rank | 2724.0000 | `7.88e-36` | -0.0160 | **YES** (p < 0.05) |
| `correct_both` | Attribution IoU (Top 20%) | Paired Wilcoxon Signed-Rank | 8495.0000 | `3.23e-06` | -0.0057 | **YES** (p < 0.05) |
| `correct_both` | Pointing Game Accuracy | McNemar Test | 0.1552 | `6.94e-01` | +0.0143 | NO |
| `incorrect_any` | Energy Inside Mask | Paired Wilcoxon Signed-Rank | 1565868.0000 | `2.21e-258` | -0.0291 | **YES** (p < 0.05) |
| `incorrect_any` | Attribution IoU (Top 20%) | Paired Wilcoxon Signed-Rank | 1708362.0000 | `7.78e-90` | -0.0146 | **YES** (p < 0.05) |
| `incorrect_any` | Pointing Game Accuracy | McNemar Test | 142.5743 | `0.00e+00` | -0.0887 | **YES** (p < 0.05) |
