# Deep Learning Inference Sequence

## Overview
This document models the internal execution lifecycle of the **RiceGuard Deep Learning Inference Pipeline**. It details the synchronous tensor transformations occurring within the neural network and decoding modules during a single forward evaluation pass, excluding external network or web application layers.

---

## Inference Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Input as Raw Input Image (RGB)
    participant Preproc as Preprocessing Pipeline
    participant Backbone as EfficientNet-B0 (features.8)
    participant ClsHead as Classification Head
    participant LocHead as Spatial Localization Head
    participant Decoder as Calibrated Decoder (Phase 3B)
    participant Output as Final Multi-Task Output

    Input->>Preproc: Ingest raw image (H x W x 3)
    activate Preproc
    Preproc->>Preproc: Bilinear Resize to 224 x 224 x 3
    Preproc->>Preproc: Convert to FloatTensor [1, 3, 224, 224]
    Preproc->>Preproc: Apply ImageNet Normalization (mu, sigma)
    Preproc-->>Backbone: Standardized Tensor X in R^(1x3x224x224)
    deactivate Preproc

    activate Backbone
    Backbone->>Backbone: Forward pass through Stages 1-8 MBConv Blocks
    Backbone-->>ClsHead: Shared Feature Tensor F in R^(1x1280x7x7)
    Backbone-->>LocHead: Shared Feature Tensor F in R^(1x1280x7x7)
    deactivate Backbone

    par Classification Branch
        activate ClsHead
        ClsHead->>ClsHead: AdaptiveAvgPool2d(1x1) -> [1, 1280, 1, 1]
        ClsHead->>ClsHead: Flatten to [1, 1280] & Dropout(0.20)
        ClsHead->>ClsHead: Linear Projection (1280 -> 6) -> Logits y_hat
        ClsHead->>ClsHead: Softmax Normalization -> Class Probabilities P(c|x)
        ClsHead-->>Output: Top-1 Disease Class & Probability Distribution
        deactivate ClsHead
    and Localization Branch
        activate LocHead
        LocHead->>LocHead: Conv2d(1280->256, 3x3) + BatchNorm2d + SiLU
        LocHead->>LocHead: Conv2d(256->5, 1x1) Projection
        LocHead-->>Decoder: Raw Spatial Grid Tensor T_loc in R^(1x5x7x7)
        deactivate LocHead
        
        activate Decoder
        Decoder->>Decoder: Sigmoid Objectness s_obj = sigmoid(l_obj)
        Decoder->>Decoder: Confidence Filter: Retain cells where s_obj >= 0.60
        Decoder->>Decoder: Decode Coordinates (bx, by, bw, bh) & Clip to [0, 1]
        Decoder->>Decoder: Rank descending by s_obj & Truncate to Top-3 (K=3)
        Decoder-->>Output: Calibrated Lesion Bounding Boxes List
        deactivate Decoder
    end

    activate Output
    Output->>Output: Aggregate Categorical Class & Spatial Bounding Boxes
    Output-->>Input: Return Unified Prediction (Class, Confidence, Bounding Boxes, Latency)
    deactivate Output
```

---

## Detailed Step-by-Step Execution Lifecycle

1. **Image Ingestion**: The inference engine receives an RGB image tensor.
2. **Preprocessing**: The raw image is resized to $224 \times 224$ via bilinear interpolation, converted to a PyTorch tensor, and normalized using channel means $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and standard deviations $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.
3. **Shared Feature Extraction**: The input tensor passes through Stages 1–8 of the EfficientNet-B0 backbone, producing the shared spatial feature map $\mathbf{F} \in \mathbb{R}^{1 \times 1280 \times 7 \times 7}$ at `features.8`.
4. **Parallel Prediction Streams**:
   - **Classification Stream**: $\mathbf{F}$ is spatially pooled (`AdaptiveAvgPool2d(1)`), flattened, regularized (`Dropout(0.20)`), projected via a linear classifier (`Linear(1280, 6)`), and normalized via softmax to generate 6-class posterior probabilities.
   - **Localization Stream**: $\mathbf{F}$ passes through a convolutional bottleneck (`Conv2d(1280, 256, 3, pad=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `SiLU` $\rightarrow$ `Conv2d(256, 5, 1)`) yielding a spatial grid tensor $\mathbf{T}_{\text{loc}} \in \mathbb{R}^{1 \times 5 \times 7 \times 7}$.
5. **Calibrated Box Decoding**:
   - Objectness logits and spatial offsets are mapped via sigmoid functions.
   - Grid cells with objectness scores $s_{\text{obj}} < 0.60$ are filtered out.
   - Centroid coordinates and dimensions are decoded, normalized relative to grid cell indices, and clipped to the interval $[0.0, 1.0]$.
   - Remaining candidates are sorted descending by confidence and truncated to the Top-$3$ most salient detections.
6. **Output Assembly**: The diagnostic class prediction, confidence percentage, probability distribution, calibrated bounding boxes, and model forward execution latency ($10.46\text{ ms}$) are aggregated into a unified output payload.
