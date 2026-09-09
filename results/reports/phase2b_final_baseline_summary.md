# Phase 2B: Final EfficientNet-B0 Baseline Training Summary

> **EXPERIMENT**: FINAL CLASSIFICATION BASELINE
> **BACKBONE**: EfficientNet-B0 (Selected from Phase 2A Screening)
> **DATASET**: Primary `RiceLeafDiseaseBD` (6 Canonical Classes)
> **DEVICE**: NVIDIA GeForce GTX 1650 (`cuda:0`, 4.0 GB VRAM)
> **AMP MIXED PRECISION**: ENABLED

---

## 1. Executive Summary

| Metric | Training Score / Value |
| :--- | :--- |
| **Selected Backbone** | **EfficientNet-B0** |
| **Best Training Epoch** | **Epoch 13** |
| **Total Epochs Completed** | **18 / 20** (Early Stopped: True) |
| **Best Validation Macro F1** | **87.76%** (`0.8776`) |
| **Internal Test Accuracy** | **90.12%** (`0.9012`) |
| **Internal Test Macro F1** | **88.91%** (`0.8891`) |
| **Internal Test Macro Precision** | **88.71%** (`0.8871`) |
| **Internal Test Macro Recall** | **89.28%** (`0.8928`) |
| **Total Parameters** | **4,015,234** |
| **Checkpoint File Size** | **15.6 MB** |
| **GPU Inference Latency** | **10.36 ms** per sample |
| **CPU Inference Latency** | **35.4 ms** per sample |
| **Peak GPU Memory (VRAM)** | **814.5 MB** / 4,096 MB |
| **Total Training Time** | **81.92 minutes** |

---

## 2. Per-Class Test Performance (Internal Test Split: 1,467 Samples)

| Class ID | Canonical Disease Class | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **Healthy** | 92.62% | 95.36% | **93.97%** | 237 |
| 1 | **Blast** | 85.65% | 89.95% | **87.75%** | 199 |
| 2 | **Brown Spot** | 92.15% | 82.57% | **87.10%** | 327 |
| 3 | **Leaf Smut** | 75.65% | 79.82% | **77.68%** | 109 |
| 4 | **Tungro** | 89.39% | 94.96% | **92.09%** | 337 |
| 5 | **Sheath Blight** | 96.77% | 93.02% | **94.86%** | 258 |

---

## 3. Strict Data Governance & Baseline Integrity Confirmation

* **Pure Image Classification Baseline**: No bounding boxes, lesion masks, or auxiliary supervision was used.
* **Primary Dataset Only**: Training and model selection strictly used `splits/primary_train.csv` (6,837) and `splits/primary_val.csv` (1,465).
* **External Datasets Untouched**: `Sethy5932`, `RiceLeafDiseaseBD5`, and `RiceSeg5932` remained 100% locked.
* **Canonical Class Order Preserved**: `0: Healthy, 1: Blast, 2: Brown Spot, 3: Leaf Smut, 4: Tungro, 5: Sheath Blight`.
* **Hardware Acceleration**: Successfully executed on **NVIDIA GeForce GTX 1650 (`cuda:0`)** with AMP.

---

## 4. Key Artifacts Generated

* **Best Checkpoint**: `experiments/phase2b_final_baseline/efficientnet_b0/checkpoints/best_model.pt`
* **Last Checkpoint**: `experiments/phase2b_final_baseline/efficientnet_b0/checkpoints/last_model.pt`
* **Training History**: `experiments/phase2b_final_baseline/efficientnet_b0/training_history.csv`
* **Live Status Log**: `experiments/phase2b_final_baseline/efficientnet_b0/status.json`
* **Evaluation Metrics**: `experiments/phase2b_final_baseline/efficientnet_b0/evaluation/`
* **Diagnostic Figures**: `experiments/phase2b_final_baseline/efficientnet_b0/figures/`
