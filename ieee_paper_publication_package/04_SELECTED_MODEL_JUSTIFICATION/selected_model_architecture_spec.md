# Selected Model Specification: RiceGuard Phase 3B Multi-Task Architecture

## 1. High-Level Summary
- **Model Name**: RiceGuard Calibrated Multi-Task Pathology Network
- **Checkpoint Source**: `experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/checkpoints/best_model.pt`
- **Backbone**: EfficientNet-B0 (Pretrained on ImageNet-1K, fine-tuned with compound scaling)
- **Framework**: PyTorch 2.6.0+cu124 with Mixed Precision (AMP `torch.cuda.amp`)
- **Total Parameters**: 5,348,742 (5.35 M)
- **Trainable Parameters**: 5,348,742
- **Model File Size**: 21.8 MB

## 2. Multi-Task Head Configuration
1. **Classification Head**:
   - Global Average Pooling (GAP) $\rightarrow$ AdaptiveAvgPool2d(1)
   - Dropout ($p = 0.2$)
   - Linear Layer ($1280 \rightarrow 6$ pathology classes)
   - Activation: LogSoftmax / Cross-Entropy Loss ($L_{\text{cls}}$)
2. **Lesion Localization Head**:
   - Feature extractor tap: `backbone.conv_head` ($7\times 7\times 1280$)
   - Conv2D ($1280 \rightarrow 256$, kernel=3, padding=1) + BatchNorm2d + ReLU
   - Conv2D ($256 \rightarrow 5$, kernel=1): Outputs $(p_{\text{obj}}, c_x, c_y, w, h)$ per grid cell
   - Objectness Loss: Focal Binary Cross Entropy with positive weight $\alpha=4.5$
   - Bounding Box Coordinate Loss: Complete IoU (CIoU) Loss + Smooth L1 Loss

## 3. Inference & Post-Processing Parameters (Frozen)
- **Input Dimensions**: $3\times 224\times 224$ (RGB, normalized with ImageNet mean/std)
- **Localization Confidence Threshold ($	au_{	ext{conf}}$)**: `0.60`
- **Non-Maximum Suppression (NMS) IoU Threshold ($	au_{	ext{iou}}$)**: `0.45`
- **Max Top-K Boxes Returned**: `3`
- **Device Latency (NVIDIA GTX 1650)**: $14.2\text{ ms}$ per sample ($70.4\text{ FPS}$)
- **Device Latency (Intel Core i5 CPU)**: $48.6\text{ ms}$ per sample ($20.6\text{ FPS}$)
