# Literature Survey

## Introduction
Automated plant disease diagnosis using computer vision and deep learning has evolved rapidly over the past decade. To contextualize the methodological and architectural innovations in the **RiceGuard** project, this survey reviews twenty approved, peer-reviewed IEEE publications (`[1]`–`[20]`) spanning five key technological paradigms:
1. Deep Convolutional Neural Networks & Transfer Learning Baselines
2. Lightweight, Edge-Optimized Architectures
3. Attention Mechanisms & Hybrid Vision Transformers
4. Object Detection & Spatial Lesion Localization
5. Explainable AI (XAI) & Visual Attribution Frameworks

```mermaid
mindmap
  root((Rice Disease Literature Survey [1]-[20]))
    Deep CNN & Transfer Learning
      Singh et al. 2026 [2] - Hybrid DenseNet & KerasTuner
      Kaur et al. 2025 [10] - Transfer Learning Benchmark
      Albashish et al. 2021 [15] - Multi-Feature Segmentation CNN
      Sultana et al. 2022 [19] - Deep CNN Baseline
      Upadhyay & Kumar 2022 [20] - Pretrained Transfer Learning
    Lightweight & Edge Architectures
      Tasci 2026 [3] - DBLA-MobileNetV2
      Ismail et al. 2024 [6] - Lightweight Edge CNN
      Sood et al. 2023 [8] - Embedded Hardware Deployment
      Hasan et al. 2024 [11] - Low-Compute Diagnostic Network
      Rahman et al. 2020 [18] - Compact Two-Stage Classifier
    Attention & Vision Transformers
      Nawer et al. 2025 [5] - Vision Transformer & Grad-CAM
      Bhattacharya et al. 2024 [7] - Spatial Attention CNN
      Patil & Bodhe 2024 [9] - Cross-Layer Attention Network
      Verma et al. 2025 [12] - Swin Transformer Backbone
      Chen et al. 2021 [17] - Channel Attention Mechanism
    Object Detection & Localization
      Nguyen et al. 2026 [1] - RiceLDD-YOLO
      Joardar et al. 2023 [13] - Modified Faster R-CNN
      Bari et al. 2021 [16] - Faster R-CNN Real-Time Detection
    Explainable AI Frameworks
      Mahmud et al. 2026 [4] - Multi-Model XAI & Grad-CAM
      Nawer et al. 2025 [5] - Saliency-Guided Diagnosis
      Rangarajan & Purushothaman 2023 [14] - Visual Attribution Saliency
```

---

## Detailed Literature Categorization

### 1. Deep Convolutional Networks and Transfer Learning Baselines
Early and contemporary deep learning literature heavily utilizes standard deep CNN architectures for whole-image disease classification:
- **Singh et al. (2026) [2]** evaluated hybrid DenseNet architectures combined with KerasTuner hyperparameter optimization, demonstrating that deep dense connectivity captures multi-scale foliar lesion patterns effectively. However, the resulting models have large parameter footprints and provide no spatial lesion localization.
- **Kaur et al. (2025) [10]** benchmarked diverse pretrained backbones (ResNet-50, VGG-16, Inception-V3) across standard rice datasets, confirming the value of transfer learning on agrarian tasks with limited sample sizes.
- **Albashish et al. (2021) [15]** proposed a multi-feature segmentation and classification framework combining color-texture descriptors with CNN features, improving separation between necrotic tissue and healthy background.
- **Sultana et al. (2022) [19]** and **Upadhyay & Kumar (2022) [20]** developed baseline CNN classifiers, reporting high validation accuracy on controlled lab datasets while highlighting vulnerability to out-of-distribution field artifacts.

### 2. Lightweight and Edge-Deployable Architectures
To address the computational constraints of rural agricultural deployment, recent literature focuses on parameter efficiency:
- **Taşcı (2026) [3]** introduced *DBLA-MobileNetV2*, incorporating a Dual-Branch Lightweight Attention module into MobileNetV2. The model achieved fast inference on edge devices while maintaining competitive classification accuracy.
- **Ismail et al. (2024) [6]** and **Hasan et al. (2024) [11]** designed compact convolutional networks using depthwise separable convolutions to minimize floating-point operations (FLOPs).
- **Sood et al. (2023) [8]** implemented embedded hardware disease diagnosis on Raspberry Pi platforms, emphasizing power consumption and real-time execution.
- **Rahman et al. (2020) [18]** demonstrated a two-stage lightweight pipeline optimizing feature extraction for resource-constrained field deployments.

### 3. Attention Mechanisms and Hybrid Vision Transformers
To enhance focus on subtle lesion regions, attention-guided architectures have gained prominence:
- **Nawer et al. (2025) [5]** explored Vision Transformers (ViT) combined with Grad-CAM, showing that global self-attention captures long-range contextual relationships across leaf blades.
- **Bhattacharya et al. (2024) [7]** and **Chen et al. (2021) [17]** integrated spatial and channel attention mechanisms (such as Squeeze-and-Excitation and CBAM), enabling networks to selectively amplify symptom features while suppressing background clutter.
- **Patil & Bodhe (2024) [9]** and **Verma et al. (2025) [12]** evaluated cross-layer attention networks and Swin Transformers, reporting superior boundary delineation for complex multi-infection symptoms at the expense of increased computational latency.

### 4. Object Detection and Spatial Lesion Localization
To advance beyond whole-image labels, researchers have explored object detection frameworks:
- **Nguyen et al. (2026) [1]** proposed *RiceLDD-YOLO*, an optimized single-stage YOLO detector targeting real-time foliar disease detection with low latency ($1.7\text{ ms}$). However, detection precision drops on small, diffuse lesions, and spatial false-positive rates on healthy leaves were not calibrated.
- **Joardar et al. (2023) [13]** and **Bari et al. (2021) [16]** implemented two-stage Faster R-CNN frameworks for disease localization. While two-stage detectors provide high localization accuracy on distinct lesions, their heavy parameter counts ($>40\text{M}$ parameters) and long inference latencies ($>80\text{ ms}$) limit edge deployment.

### 5. Explainable AI (XAI) and Visual Attribution
To interpret deep network predictions, post-hoc visual attribution has become a major research direction:
- **Mahmud et al. (2026) [4]** applied Grad-CAM, Grad-CAM++, and Score-CAM across various deep backbones, demonstrating that attribution heatmaps provide visual verification for agricultural users.
- **Rangarajan & Purushothaman (2023) [14]** evaluated visual attribution saliency maps across tomato and rice disease models, observing that unregularized classifiers frequently attend to image borders and background soil rather than pathological lesions.
- However, existing XAI studies rely almost exclusively on qualitative visual inspection on cherry-picked samples, lacking quantitative spatial grounding against independent pixel-level biological lesion masks.

---

## Comprehensive Literature Comparison Matrix

The table below summarizes the 20 approved literature references across key technical dimensions:

| Ref | Author & Year | Primary Architecture / Approach | Primary Task | Spatial Localization | Quantitative XAI | Edge Feasibility | Key Limitation Addressed by RiceGuard |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `[1]` | Nguyen et al. (2026) | RiceLDD-YOLO (Optimized YOLO) | Detection | Yes (Bounding Box) | No | High ($1.7\text{ ms}$) | High false-positive rate on small lesions; no post-hoc calibration. |
| `[2]` | Singh et al. (2026) | Hybrid DenseNet + KerasTuner | Classification | No | No | Moderate | Heavy parameter count; no spatial localization or XAI. |
| `[3]` | Taşcı (2026) | DBLA-MobileNetV2 (Dual Attention) | Classification | No | No | High (Mobile) | Classification-only; cannot localize specific lesion coordinates. |
| `[4]` | Mahmud et al. (2026) | Multi-Model CNN + Grad-CAM | Classification | No | Qualitative Only | Moderate | XAI is purely visual; lacks quantitative ground-truth evaluation. |
| `[5]` | Nawer et al. (2025) | Vision Transformer + Grad-CAM | Classification | No | Qualitative Only | Low | High compute requirement; ungrounded attribution maps. |
| `[6]` | Ismail et al. (2024) | Lightweight Edge-CNN | Classification | No | No | High | Classification-only; vulnerable to background shortcut learning. |
| `[7]` | Bhattacharya et al. (2024) | Spatial Attention CNN | Classification | No | No | Moderate | Lacks spatial bounding boxes and explainability benchmarking. |
| `[8]` | Sood et al. (2023) | Embedded CNN (Raspberry Pi) | Classification | No | No | High (Edge) | Low diagnostic resolution; no lesion localization output. |
| `[9]` | Patil & Bodhe (2024) | Cross-Layer Attention Network | Classification | No | No | Moderate | High architectural complexity without spatial verification. |
| `[10]` | Kaur et al. (2025) | Pretrained Transfer Learning Bench | Classification | No | No | Moderate | Generic benchmark; lacks multi-task learning and XAI. |
| `[11]` | Hasan et al. (2024) | Low-Compute Diagnostic Network | Classification | No | No | High | Pure classification; does not suppress background artifacts. |
| `[12]` | Verma et al. (2025) | Swin Transformer Backbone | Classification | No | Qualitative Only | Low | High memory footprint; uncalibrated decision boundaries. |
| `[13]` | Joardar et al. (2023) | Modified Faster R-CNN | Detection | Yes (Two-Stage) | No | Low ($>80\text{ ms}$) | High latency and heavy parameter footprint ($>40\text{M}$ params). |
| `[14]` | Rangarajan & Puru. (2023) | Visual Attribution Saliency | Classification | No | Qualitative Only | Moderate | Saliency maps exhibit heavy border and background noise. |
| `[15]` | Albashish et al. (2021) | Multi-Feature Segmentation CNN | Seg + Class | Yes (Pixel Mask) | No | Moderate | Multi-stage pipeline requires dense annotation during training. |
| `[16]` | Bari et al. (2021) | Real-Time Faster R-CNN | Detection | Yes (Two-Stage) | No | Low | Heavy computational cost prevents lightweight edge deployment. |
| `[17]` | Chen et al. (2021) | Channel Attention ResNet | Classification | No | No | Moderate | Pure classification; no spatial evidence or calibration. |
| `[18]` | Rahman et al. (2020) | Two-Stage Compact Classifier | Classification | No | No | High | Heuristic two-stage pipeline lacking unified end-to-end training. |
| `[19]` | Sultana et al. (2022) | Deep CNN Baseline | Classification | No | No | Moderate | Tested on clean lab backgrounds; fails on complex field imagery. |
| `[20]` | Upadhyay & Kumar (2022) | Pretrained Transfer Learning | Classification | No | No | Moderate | No localization, calibration, or explainability analysis. |

---

## Comparison With Existing Systems
The architectural and methodological distinctions of **RiceGuard** in comparison to existing systems in the literature are summarized below:

1. **Lightweight Multi-Task Learning vs. Heavy Detectors**: In contrast to two-stage object detectors (`[13]`, `[16]`) that demand substantial memory and latency, RiceGuard integrates a shared EfficientNet-B0 backbone ($4.02\text{M}$ parameters) with a lightweight $7 \times 7$ grid localization head, maintaining edge-friendly latency ($10.46\text{ ms}$).
2. **Calibrated Localization vs. Uncalibrated Detection**: Unlike standard single-stage detection frameworks (`[1]`) that over-predict bounding boxes on healthy foliage, RiceGuard introduces Phase 3B positive-weight loss optimization ($w_{\text{pos}}=10.0$), confidence threshold calibration ($\tau=0.60$), and Top-$K$ candidate decoding ($K=3$), cutting healthy leaf false positives by $49.0\%$ ($p < 10^{-6}$).
3. **Quantitative XAI Grounding vs. Qualitative Cherry-Picking**: Unlike previous XAI literature (`[4]`, `[5]`, `[14]`), RiceGuard introduces an automated quantitative evaluation pipeline benchmarking Grad-CAM attributions against $N=4,348$ independent biological lesion masks from the RiceSeg5932 dataset.
4. **Feature Regularization via Multi-Task Co-Training**: RiceGuard demonstrates that auxiliary localization co-training acts as an effective spatial regularizer, reducing spurious border attention on healthy leaves by $50.3\%$ and significantly improving Energy-Inside-Mask ($+41.6\%$).

---

## Summary of Literature Positioning
The RiceGuard framework bridges the gap between lightweight classification models and heavy object detection systems. By coupling a shared convolutional backbone with calibrated spatial grid regression and quantitative XAI benchmarking, RiceGuard achieves dual-output disease diagnosis (classification and localization) while remaining computationally efficient and scientifically interpretable.
