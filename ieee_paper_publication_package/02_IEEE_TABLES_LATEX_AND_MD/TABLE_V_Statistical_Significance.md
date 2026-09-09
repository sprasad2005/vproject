# TABLE V: Statistical Significance Hypothesis Tests

| Hypothesis / Comparison | Statistical Test | Test Statistic | p-value | Scientific Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Attribution IoU: P3B > P2B** | Paired Student's $t$-test | $t = 14.82$ | $p < 10^{-12}$ | Reject $H_0$ ($p < 0.001$, highly significant) |
| **Energy Inside Mask: P3B > P2B** | Wilcoxon Signed-Rank | $W = 4210.5$ | $p < 10^{-10}$ | Reject $H_0$ ($p < 0.001$, highly significant) |
| **Classification Preservation: P3B $\approx$ P2B** | McNemar's $\chi^2$ Test | $\chi^2 = 1.18$ | $p = 0.2774$ | Fail to Reject $H_0$ (Diagnostic Invariance) |
| **Healthy FPR Suppression: P3B < P3** | Fisher's Exact Test | $\text{OR} = 0.118$ | $p < 10^{-6}$ | Reject $H_0$ (93% suppression confirmed) |
