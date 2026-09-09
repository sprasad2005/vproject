# Section II: Proposed Methodology & Loss Formulation

## A. Architecture Overview
RiceGuard employs a modified **EfficientNet-B0** backbone optimized with compound scaling across depth $d=1.0$, width $w=1.0$, and input resolution $r=224	imes 224	imes 3$. The architectural pipeline bifurcates at the terminal feature map $F \in \mathbb{R}^{7	imes 7	imes 1280}$:
1. **Classification Stream**: Global Average Pooling (GAP) aggregates spatial dimensions to a 1280-dimensional feature vector, followed by Dropout ($p=0.2$) and a linear classification head producing unnormalized logits $\hat{y} \in \mathbb{R}^6$.
2. **Lesion Localization Stream**: A $3	imes 3$ convolutional adaptation layer reduces channel dimensionality to 256 with Batch Normalization and ReLU, followed by a $1	imes 1$ convolutional prediction layer producing a 5-channel tensor $T_{	ext{loc}} \in \mathbb{R}^{7	imes 7	imes 5}$ encoding objectness probability $p_{	ext{obj}}$ and normalized bounding box coordinates $(c_x, c_y, w, h)$.

## B. Multi-Task Joint Loss Function
The composite loss function $\mathcal{L}_{	ext{total}}$ jointly penalizes classification errors and lesion localization discrepancies:
$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{cls}}(\hat{y}, y) + \lambda_{	ext{obj}} \mathcal{L}_{	ext{obj}}(p_{	ext{obj}}, p^*) + \lambda_{	ext{box}} \mathcal{L}_{	ext{box}}(b, b^*)$$

Where:
- $\mathcal{L}_{	ext{cls}}$ is standard Multi-Class Cross-Entropy Loss over the 6 pathology classes.
- $\mathcal{L}_{	ext{obj}}$ is Focal Binary Cross-Entropy Loss with positive weight $lpha=4.5$ and focusing parameter $\gamma=2.0$ to mitigate severe background patch dominance:
  $$\mathcal{L}_{	ext{obj}} = -lpha p^* (1 - p_{	ext{obj}})^\gamma \log(p_{	ext{obj}}) - (1 - p^*) p_{	ext{obj}}^\gamma \log(1 - p_{	ext{obj}})$$
- $\mathcal{L}_{	ext{box}}$ is a linear combination of Complete Intersection-over-Union (CIoU) Loss and Smooth L1 Loss applied strictly on grid cells containing positive ground-truth lesion annotations ($p^* = 1$):
  $$\mathcal{L}_{	ext{box}} = \mathcal{L}_{	ext{CIoU}}(b, b^*) + eta \mathcal{L}_{	ext{Smooth-L1}}(b, b^*)$$
- Hyperparameters are empirically optimized: $\lambda_{	ext{obj}} = 1.0$, $\lambda_{	ext{box}} = 2.5$, $eta = 1.0$.

## C. Confidence-Calibrated Lesion Decoding
During inference, candidate bounding boxes are decoded across the $7	imes 7$ grid. Detections are filtered using a calibrated confidence threshold $	au_{	ext{conf}} = 0.60$ followed by Non-Maximum Suppression (NMS) with IoU threshold $	au_{	ext{iou}} = 0.45$, retaining at most $K = 3$ high-confidence lesion proposals.
