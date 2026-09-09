# Phase 5: Deterministic Failure-Case Analysis Summary

**Evaluated Split**: Internal Test (`primary_test.csv`, $N=1467$)  

| Category | Count | Percentage | Description |
|---|---|---|---|
| **Mutual Success** | 138 | 9.41% | Both models correctly classified |
| **Baseline Wins** | 99 | 6.75% | Phase 2B correct, Phase 3B wrong |
| **Multi-Task Wins** | 50 | 3.41% | Phase 3B correct, Phase 2B wrong |
| **Mutual Errors** | 1180 | 80.44% | Both models incorrectly classified |
| **Healthy False Positives** | 51 | 21.52% | Healthy leaves with predicted lesion boxes |
| **Disease Lesion Misses** | 1230 | 100.00% | Disease images where Phase 3B missed all boxes |
| **High Conf Poor Loc** | 971 | 66.19% | Cls conf > 0.80 but matched IoU < 0.20 |
| **Good Loc Bad Cls** | 0 | 0.00% | Matched IoU >= 0.50 but wrong disease class |
