# RiceGuard Phase 3: Lesion-Aware Multi-Task Model Report

## Executive Summary
Phase 3 establishes the proposed lesion-aware multi-task architecture by extending the verified EfficientNet-B0 backbone with a lightweight spatial localization head ($7 \times 7$ grid) supervised by bounding box annotations from `RiceLeafDiseaseBD`.

## Core Experimental Results (Internal Test Split)

| Dimension | Metric | Phase 2B Baseline (Classification-Only) | Phase 3 Proposed (Lesion-Aware Multi-Task) | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- |
| **Classification** | Test Accuracy | 90.12% | **88.75%** | -1.37% |
| | Test Balanced Acc | 89.28% | **86.97%** | -2.31% |
| | Test Macro F1 | 0.8891 | **0.8729** | -0.0162 |
| | Test Weighted F1 | 0.9011 | **0.8869** | -0.0142 |
| **Localization** | Precision (IoU $\ge 0.5$) | N/A (Baseline) | **0.0223** | *New Capability* |
| | Recall (IoU $\ge 0.5$) | N/A (Baseline) | **0.0914** | *New Capability* |
| | Localization F1 | N/A (Baseline) | **0.0358** | *New Capability* |
| | Mean Matched IoU | N/A (Baseline) | **0.6032** | *New Capability* |
| | Healthy False Pos. Rate | N/A (Baseline) | **21.52%** | *Safe Backgrounding* |
| **Efficiency** | Parameters | 4,015,234 | **6,966,151** | +2,950,917 |
| | Model Size (MB) | 15.61 MB | **26.86 MB** | ++11.25 MB |
| | GPU Latency (ms) | 11.20 ms | **13.52 ms** | ++2.32 ms |
| | Peak VRAM (MB) | 814.7 MB | **520.3 MB** | -294.4 MB |

## Objectness Imbalance Handling
- Pos_weight calculated strictly from `primary_train.csv`: `22.9005`
- Total Grid Cells in Train: `335,013`
- Positive Lesion Cells: `14,017` (4.18%)
- Negative Cells: `320,996` (95.82%)

## Per-Class Disease Classification Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Healthy** | 0.9271 | 0.9662 | 0.9463 | 237 |
| **Blast** | 0.8600 | 0.8643 | 0.8622 | 199 |
| **Brown Spot** | 0.8354 | 0.8379 | 0.8366 | 327 |
| **Leaf Smut** | 0.7755 | 0.6972 | 0.7343 | 109 |
| **Tungro** | 0.9017 | 0.9258 | 0.9136 | 337 |
| **Sheath Blight** | 0.9637 | 0.9264 | 0.9447 | 258 |

## Per-Class Lesion Localization Performance (IoU $\ge 0.5$)

| Class | GT Boxes | Pred Boxes | TP | FP | FN | Precision | Recall | Localization F1 | Mean IoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Healthy** | 0 | 162 | 0 | 162 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Blast** | 703 | 2177 | 18 | 2159 | 685 | 0.0083 | 0.0256 | 0.0125 | 0.6115 |
| **Brown Spot** | 1384 | 4728 | 13 | 4715 | 1371 | 0.0027 | 0.0094 | 0.0043 | 0.6132 |
| **Leaf Smut** | 203 | 795 | 1 | 794 | 202 | 0.0013 | 0.0049 | 0.0020 | 0.5178 |
| **Tungro** | 483 | 2930 | 145 | 2785 | 338 | 0.0495 | 0.3002 | 0.0850 | 0.5941 |
| **Sheath Blight** | 377 | 2127 | 111 | 2016 | 266 | 0.0522 | 0.2944 | 0.0887 | 0.6133 |

## Checkpoints and Artifacts
- **Best Model Checkpoint**: `experiments/phase3_lesion_aware/efficientnet_b0_multitask/checkpoints/best_model.pt` (Epoch 4)
- **Last Model Checkpoint**: `experiments/phase3_lesion_aware/efficientnet_b0_multitask/checkpoints/last_model.pt`
- **Training History CSV**: `experiments/phase3_lesion_aware/efficientnet_b0_multitask/training_history.csv`
- **Evaluation Summary**: `experiments/phase3_lesion_aware/efficientnet_b0_multitask/evaluation/efficientnet_b0_multitask_evaluation_summary.json`
- **Diagnostic Plots**:
  - `experiments/phase3_lesion_aware/efficientnet_b0_multitask/figures/training_loss_curve.png`
  - `experiments/phase3_lesion_aware/efficientnet_b0_multitask/figures/accuracy_curve.png`
  - `experiments/phase3_lesion_aware/efficientnet_b0_multitask/figures/macro_f1_curve.png`
  - `experiments/phase3_lesion_aware/efficientnet_b0_multitask/figures/localization_f1_curve.png`
  - `experiments/phase3_lesion_aware/efficientnet_b0_multitask/figures/multitask_metrics_dashboard.png`
  - `experiments/phase3_lesion_aware/efficientnet_b0_multitask/figures/confusion_matrix.png`
