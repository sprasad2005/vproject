# Structured Academic Abstract

## Abstract
Automated visual identification of rice leaf phytopathologies is vital for preventing catastrophic crop yield loss in smallholder agriculture. However, standard deep learning approaches overwhelmingly formulate diagnosis as whole-image classification, providing no spatial localization of necrotic lesions, generating high false-alarm rates on healthy foliage, and relying on qualitative, ungrounded visual heatmaps. 

This study presents **RiceGuard**, an integrated multi-task deep learning framework built on an EfficientNet-B0 backbone ($4.02\text{M}$ parameters) that simultaneously performs 6-class disease classification and calibrated dense spatial lesion localization, complemented by quantitative Explainable AI (XAI) biological grounding. Evaluated on a curated field dataset ($N = 3,355$), the Phase 2B classification-only baseline achieved $90.12\%$ test accuracy and $0.8891$ Macro F1. To provide spatial evidence without heavyweight two-stage object detectors, the architecture was bifurcated into classification and dense $7 \times 7$ grid localization heads (Phase 3). 

To resolve the severe class imbalance inherent in dense grid regression, Phase 3B introduced positive-weight loss optimization ($w_{\text{pos}} = 10.0$) coupled with validation-only confidence threshold calibration ($\tau = 0.60$) and Top-$K$ candidate decoding ($K = 3$). On the held-out test split ($N = 674$), Phase 3B sustained robust classification performance ($88.96\%$ accuracy, $0.8751$ Macro F1) while dramatically improving localization precision, increasing Mean Matched IoU to $0.6423$, cutting predicted bounding-box clutter by $74.6\%$ (from $8.81$ to $2.24$ boxes/image), and reducing the healthy leaf false-positive rate from $21.52\%$ to $10.97\%$ ($p < 10^{-6}$). 

Post-hoc Grad-CAM attribution maps evaluated against $N = 4,348$ independent biological lesion masks from the RiceSeg5932 dataset demonstrated that multi-task learning significantly improved spatial lesion grounding, increasing Energy-Inside-Mask from $6.81\%$ to $9.64\%$ ($+41.6\%$ relative gain), improving Attribution IoU to $0.0864$, and halving spurious border attention on healthy leaves from $18.5\%$ to $9.2\%$. The calibrated pipeline achieves a lightweight inference latency of $10.46\text{ ms}$ on CUDA hardware. The operational model is integrated into a verified FastAPI backend and React 19 frontend for local execution, with production containerization and edge deployment configurations fully prepared.

---

## Keywords
Rice Leaf Disease Diagnosis, Multi-Task Deep Learning, Calibrated Lesion Localization, Explainable AI (XAI), Grad-CAM Grounding, EfficientNet-B0, Agricultural Decision Support Systems.
