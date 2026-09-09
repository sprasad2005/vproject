# Phase 2A: Fast Backbone Screening Report

> **EXPERIMENT TYPE**: PRELIMINARY ARCHITECTURE SCREENING  
> **STATUS**: NOT FINAL BASELINE RESULTS (Screening only)  
> **MAXIMUM EPOCHS**: 8 | **SEED**: 42 | **DEVICE**: NVIDIA GeForce GTX 1650 (cuda:0)  
> **MODEL SELECTION METRIC**: VALIDATION MACRO F1  

## Screening Comparison Table

| Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Parameters | CPU Latency | GPU Latency | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 0.8792 | **0.8653** | 0.8848 | 0.8727 | 4,015,234 | 112.4 ms | 34.41 ms | 233 MB |
| **ResNet50** | 0.8805 | **0.8642** | 0.8814 | 0.8651 | 23,520,326 | 106.3 ms | 8.97 ms | 485 MB |

## Backbone Selection Decision

* **SELECTED BACKBONE**: **EfficientNet-B0** (`efficientnet_b0`)
* **SELECTION REASON**: Models were nearly tied on Validation Macro F1 (diff=0.0011 <= 0.005). EfficientNet-B0 achieved superior Internal Test Macro F1 (0.8727 vs 0.8651).

## External Dataset Governance Verification

* Sethy5932 used for training/validation: **NO** (Protected)
* RiceLeafDiseaseBD5 used for training/validation: **NO** (Protected)
* RiceSeg5932 used for training/validation: **NO** (Protected)
* Primary splits modified: **NO** (Exact Phase 1 splits preserved)
