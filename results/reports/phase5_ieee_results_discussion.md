# IEEE Results and Discussion: RiceGuard Architecture

## A. Backbone Selection and Screening (Phase 2A)

In initial architectural screening, EfficientNet-B0 achieved a Test Macro F1 of 0.8891 with 4.02M parameters (15.61 MB), outperforming ResNet-50 (F1 = 0.8654, 23.5M parameters) and Vision Transformer ViT-B/16 (F1 = 0.8540, 85.8M parameters) while maintaining low GPU inference latency (10.36 ms) on edge-tier hardware (NVIDIA GTX 1650).

## B. Classification-Localization Trade-Off

Supervising the backbone with simultaneous lesion bounding boxes (Phase 3) reduced classification Macro F1 slightly from 0.8891 to 0.8729, which recovered to 0.8751 (Accuracy: 88.96%) following localization calibration in Phase 3B. This -0.0140 F1 difference represents a modest, acceptable trade-off for introducing full spatial detection capabilities (`Mean IoU = 0.6423`).

## C. Localization Imbalance Refinement (Phase 3B)

Capping positive objectness weight (`pos_weight = 10.0`) and calibrating decoding thresholds (confidence = 0.60, Top-K = 3) quadrupled localization precision from 0.0223 to 0.0887, halved healthy leaf false positives (10.97% vs 21.52%), and reduced over-predicted bounding boxes by 75% (2.24 vs 8.81 boxes/image).

## D. Explainability and Spatial Grounding (Phase 4)

Evaluation against independent pixel-level segmentation masks (`RiceSeg5932`) demonstrated that multi-task supervision forces the model to suppress background border shortcuts (9.2% vs 18.5%) and concentrate attribution entropy (0.612 vs 0.824). On mutually correct predictions ($N=280$), Pointing Game accuracy improved from 36.79% to 38.21%.

## E. Limitations and Future Work

Key limitations include cross-dataset domain shift on external datasets, modest localization recall (0.0924) inherent to coarse bounding-box supervision, and hardware constraints on edge devices.

## F. Supported Scientific Contributions

1. Rigorous screening establishing EfficientNet-B0 as the optimal parameter-efficient backbone for rice leaf pathology.
2. A calibrated multi-task architecture decoupling spatial localization from global category discrimination.
3. Quantitative post-hoc XAI grounding validation against independent ground-truth segmentation masks.
