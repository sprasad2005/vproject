# RiceGuard Phase 3B: Localization Refinement Summary Report

## Executive Summary
Phase 3B implemented a controlled refinement over Phase 3 to solve objectness over-sensitivity by:
1. Capping the effective training positive weight `effective_pos_weight = min(22.90, 10.0) = 10.0`.
2. Initializing from Phase 3 best checkpoint with reinitialized optimizer and scheduler.
3. Conducting validation-only confidence threshold and Top-K decoding calibration.

## Key Scientific Questions Answered

1. **Did localization precision improve?**
   - **Yes.** Localization Precision shifted from `0.0223` to **`0.0887`** (+0.0664).
2. **Did localization recall improve?**
   - Precision-Recall tradeoff was rebalanced: Recall is `0.0924` vs Phase 3 `0.0914`.
3. **Did localization F1 improve?**
   - Localization F1 reached **`0.0905`** vs Phase 3 `0.0358` (+0.0547).
4. **Did Healthy false positive rate decrease?**
   - **Yes.** Healthy False Positive Rate dropped from `21.52%` down to **`10.97%`** (-10.55%).
5. **Did predicted boxes per image decrease?**
   - **Yes.** Average predicted boxes per image drastically reduced from `8.81` down to **`2.24`** boxes/image.
6. **Did classification Macro F1 recover or improve?**
   - Classification Test Macro F1 is **`0.8751`** (Accuracy: **`88.96%`**).
7. **Which model should proceed to Phase 4 XAI?**
   - **`Phase 3B EfficientNet-B0 Refined Multi-Task Model`** (`best_model.pt`) with frozen decoding configuration (Threshold = `0.60`, Top-K = `3`).
8. **What is the final frozen localization decoding configuration?**
   - `confidence_threshold`: `0.60`
   - `top_k`: `3`

## 3-Way Comparative Overview

| Metric | Phase 2B Baseline | Phase 3 Multi-Task | Phase 3B Refined |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 90.12% | 88.75% | **88.96%** |
| **Test Macro F1** | 0.8891 | 0.8729 | **0.8751** |
| **Localization Precision** | N/A | 0.0223 | **0.0887** |
| **Localization F1** | N/A | 0.0358 | **0.0905** |
| **Mean Matched IoU** | N/A | 0.6032 | **0.6423** |
| **Healthy False Pos. Rate** | N/A | 21.52% | **10.97%** |
| **Avg Predicted Boxes/Img** | N/A | 8.81 | **2.24** |

## Per-Class Disease Classification Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Healthy** | 0.9283 | 0.9565 | 0.9422 | 230 |
| **Blast** | 0.8794 | 0.8621 | 0.8706 | 203 |
| **Brown Spot** | 0.8257 | 0.8464 | 0.8359 | 319 |
| **Leaf Smut** | 0.7156 | 0.7573 | 0.7358 | 103 |
| **Tungro** | 0.9318 | 0.9075 | 0.9195 | 346 |
| **Sheath Blight** | 0.9612 | 0.9323 | 0.9466 | 266 |

## Per-Class Lesion Localization Performance (IoU $\ge 0.5$)

| Class | GT Boxes | Pred Boxes | TP | FP | FN | Precision | Recall | Localization F1 | Mean IoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Healthy** | 0 | 46 | 0 | 46 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Blast** | 703 | 539 | 13 | 526 | 690 | 0.0241 | 0.0185 | 0.0209 | 0.5515 |
| **Brown Spot** | 1384 | 946 | 16 | 930 | 1368 | 0.0169 | 0.0116 | 0.0137 | 0.5873 |
| **Leaf Smut** | 203 | 244 | 6 | 238 | 197 | 0.0246 | 0.0296 | 0.0268 | 0.6853 |
| **Tungro** | 483 | 840 | 147 | 693 | 336 | 0.1750 | 0.3043 | 0.2222 | 0.6490 |
| **Sheath Blight** | 377 | 664 | 109 | 555 | 268 | 0.1642 | 0.2891 | 0.2094 | 0.6498 |

## Artifacts Generated
- **Best Model Checkpoint**: `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/checkpoints/best_model.pt`
- **Training History CSV**: `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/training_history.csv`
- **Threshold Sweep CSV**: `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/calibration/threshold_sweep.csv`
- **Joint Sweep CSV**: `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/calibration/joint_threshold_topk_sweep.csv`
- **Selected Decoding Config**: `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/calibration/selected_decoding_config.json`
- **Test Overlay Visualization**: `results/figures/phase3b/test_localization_visualizations.png`
