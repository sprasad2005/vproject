# Phase 5: Cross-Dataset Domain-Shift Analysis

## 1. Dataset Role Separation

- **`RiceLeafDiseaseBD` (In-Domain)**: Used strictly for training, validation, internal testing, and bounding-box localization. Test Macro F1 reaches 0.8891 (2B) and 0.8751 (3B).
- **`RiceSeg5932` (Out-of-Domain XAI Benchmark)**: Used strictly as `xai_ground_truth_only`. It was never seen during training or tuning.

## 2. Impact on Quantitative Explainability Metrics

- **Domain Gap**: RiceSeg originates from an external dataset distribution with differing background lighting, leaf angles, and lesion colorations.
- **Attribution Behavior**: On the full out-of-domain dataset ($N=4,348$), overall classification accuracy drops across both models. Phase 2B generates diffuse visual heatmaps covering 9.64% of mask area, while Phase 3B produces more focused attributions (6.81%).
- **Stratified Correctness Insights**: When evaluating samples correctly classified by both models ($N=280$), Phase 3B achieves higher Pointing Game Accuracy (38.21% vs 36.79%), proving that multi-task supervision aligns the saliency peak more tightly on true lesions when domain features are recognized.
