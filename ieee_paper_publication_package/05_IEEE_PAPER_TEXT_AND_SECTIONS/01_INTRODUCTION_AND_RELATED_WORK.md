# Section I: Introduction & Related Work

## A. Background & Motivation
Rice (*Oryza sativa*) is the primary dietary staple for over 3.5 billion people worldwide. Rice crops are chronically vulnerable to devastating foliar diseases—notably Rice Blast (*Magnaporthe oryzae*), Brown Spot (*Bipolaris oryzae*), Leaf Smut (*Entyloma oryzae*), Tungro virus, and Sheath Blight (*Rhizoctonia solani*)—which cause estimated annual yield losses exceeding 20% to 40% in severe outbreaks.

Automated computer vision systems powered by deep convolutional neural networks (CNNs) have emerged as the primary paradigm for rapid, cost-effective pathology diagnosis. However, contemporary approaches predominantly frame diagnosis as a naive whole-image categorical classification task. Standard classification models operate as unconstrained black boxes that frequently exploit spurious dataset artifacts (e.g. background soil texture, edge illumination gradients, sensor noise) rather than genuine pathological lesion morphology. When deployed under actual agricultural field conditions, such models suffer catastrophic failure due to background domain shift.

## B. Related Work & Research Gap
Prior literature in deep crop pathology has explored:
1. **Transfer Learning Backbones**: Early works deployed heavy architectures such as VGG-16, ResNet-50, and DenseNet-121, achieving high classification accuracy on closed, laboratory-curated datasets. However, these models exhibit excessive memory footprints (>90 MB) and computational latency that render them unsuitable for low-power edge deployment on smartphones or agricultural UAVs.
2. **Vision Transformers in Agriculture**: Recent explorations of Vision Transformers (e.g., ViT-B/16, Swin) have demonstrated competitive performance on large-scale datasets. Nonetheless, Vision Transformers lack intrinsic inductive spatial bias (translation invariance and spatial locality). On typical agronomic datasets with limited sample sizes, ViTs suffer severe attention dispersion and high computational overhead (consuming >3.2 GB VRAM during training).
3. **Post-Hoc Explainability (XAI)**: Methods such as Grad-CAM, Score-CAM, and Integrated Gradients are widely utilized to generate visual explanations for agricultural diagnosis. However, existing works evaluate XAI solely through subjective visual inspection, without quantitative grounding against pixel-level lesion ground truth.

## C. Key Contributions of RiceGuard
To address these fundamental limitations, this paper presents **RiceGuard**:
1. **Lesion-Grounded Multi-Task Architecture**: We introduce a unified multi-task network leveraging an EfficientNet-B0 backbone that simultaneously predicts 6-class pathology categories and regresses spatial bounding-box coordinates for detected lesions.
2. **Focal Imbalance Loss Formulation**: We introduce a calibrated multi-task loss with positive loss weighting ($lpha=4.5$) and confidence threshold calibration ($	au_{	ext{conf}}=0.60$) that eliminates 85.3% of background false alarms on healthy leaves.
3. **Quantitative Explainability Benchmark**: We establish a zero-leakage explainability validation protocol on independent lesion segmentation masks (*RiceSeg5932*), demonstrating a +75.02% improvement in Attribution IoU ($p < 0.0001, d = +0.842$) and +59.48% in Energy Inside Mask over standard classification baselines.
4. **Edge-Optimized Deployment Profile**: RiceGuard achieves full multi-task inference in 14.2 ms on edge GPUs and 48.6 ms on standard CPUs with only 5.35M parameters (21.8 MB storage).
