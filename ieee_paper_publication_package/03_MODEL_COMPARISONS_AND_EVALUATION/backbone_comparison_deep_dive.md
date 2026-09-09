# Architectural Backbone Comparative Analysis

## 1. Executive Summary
To identify the most viable backbone for lesion-grounded edge deployment, three distinct paradigm architectures were screened during Phase 2A under identical training protocol (AdamW, lr=1e-4, batch=16, cosine decay):
1. **ResNet-50** (Standard Residual CNN)
2. **Vision Transformer (ViT-B/16)** (Self-Attention Transformer)
3. **EfficientNet-B0** (Compound Scaled Lightweight CNN)

## 2. Quantitative Comparison

| Metric | ResNet-50 | Vision Transformer (ViT-B/16) | EfficientNet-B0 | Optimal Choice |
| :--- | :--- | :--- | :--- | :--- |
| **Parameter Count** | 25.56 M | 86.57 M | **5.29 M** | **EfficientNet-B0 (16.4× smaller than ViT)** |
| **VRAM Footprint** | 1,480 MB | 3,210 MB | **890 MB** | **EfficientNet-B0 (72.3% lower VRAM)** |
| **Epoch Time (GTX 1650)**| 42.1 s | 118.4 s | **24.6 s** | **EfficientNet-B0 (4.8× faster than ViT)** |
| **Validation Macro F1** | 0.8654 | 0.8420 | **0.8987** | **EfficientNet-B0 (+3.33% over ResNet)** |
| **Test Macro F1** | 0.8521 | 0.8305 | **0.8891** | **EfficientNet-B0 (+3.70% over ResNet)** |
| **Test Accuracy** | 86.01% | 84.12% | **89.42%** | **EfficientNet-B0 (+3.41% over ResNet)** |

## 3. Scientific Justification for Rejecting ViT-B/16 and ResNet-50
- **ViT-B/16 Overfitting on Limited Agronomic Datasets**: Without massive pre-training (e.g. JFT-300M) or extensive data augmentation, ViT lacks inductive spatial bias (translation invariance and locality). On the 7,000-image RiceLeafDiseaseBD dataset, ViT suffered severe attention dispersion and overfitted rapidly, yielding the lowest Test Macro F1 (0.8305) while consuming 3,210 MB VRAM (approaching the 4GB hardware limit).
- **ResNet-50 Redundancy**: ResNet-50 contains 25.56M parameters with dense $3\times 3$ convolutions across 50 layers. It consumed 1.66× more VRAM and 1.71× more compute time than EfficientNet-B0, while yielding lower classification discrimination across small punctate lesions (Brown Spot F1: 0.812 vs 0.852).
- **EfficientNet-B0 Superiority**: Employs Mobile Inverted Bottleneck Convolutions (MBConv) with squeeze-and-excitation optimization. Compound scaling of depth, width, and resolution ensures optimal feature extraction for heterogeneous pathology symptoms at a fraction of the computational burden.
