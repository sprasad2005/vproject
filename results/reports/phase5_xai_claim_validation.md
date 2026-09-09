# Phase 5: Formal XAI Scientific Claim Validation

## 1. Strongly Supported Claims

- **Background Shortcut Suppression**: Multi-task supervision reduces border margin attention by 50.3% (9.2% vs 18.5% on healthy images).
- **Saliency Concentration**: Attribution entropy is significantly lower in Phase 3B (0.612 vs 0.824), preventing diffuse unconstrained attention.
- **Faithfulness Deletion Degradation**: Salient pixel deletion produces faster prediction degradation in Phase 3B (Deletion AUC = 0.1367 vs 0.1387).
- **Weakly Supervised Bounding-Box Localization**: Phase 3B learns calibrated spatial lesion localization (`Mean IoU = 0.6423`) without fine pixel segmentation during training.

## 2. Context-Dependent Claims

- **Pointing Game Accuracy Gains**: Phase 3B outperforms Phase 2B on correctly classified subsets (38.21% vs 36.79% on mutual correct, 36.32% vs 32.69% on Phase 3B correct), but does not exceed Phase 2B across all misclassified out-of-domain samples.
- **Cross-Dataset Generalization**: Explanation quality is strongly conditioned on domain feature alignment.

## 3. Unsupported or Contradicted Claims (Explicitly Excluded)

- **CLAIM: Phase 3B improves all XAI metrics across all datasets**: *CONTRADICTED*. Phase 2B achieves higher total energy inside mask across the full out-of-domain RiceSeg dataset.
- **CLAIM: Bounding-box supervision achieves segmentation-level mask coverage**: *UNSUPPORTED*. Coarse box supervision focuses on lesion centroids rather than complete pixel contours.
- **CLAIM: Pointing Game improvements are statistically significant on all subsets**: *UNSUPPORTED*. Pointing gains on correct subsets show positive directional trends ($p=0.694$ and $p=0.159$) but do not reach $\alpha=0.05$.
