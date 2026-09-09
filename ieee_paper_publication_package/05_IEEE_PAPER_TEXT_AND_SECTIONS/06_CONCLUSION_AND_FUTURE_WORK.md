# Section VI: Conclusion & Future Work

## A. Conclusion
This paper presented **RiceGuard**, an interpretable, lesion-grounded multi-task deep learning architecture for accurate and explainable rice leaf disease diagnosis. By coupling an EfficientNet-B0 backbone with a calibrated multi-task loss, RiceGuard achieves high 6-class diagnostic accuracy (Macro F1 = 0.8814, Accuracy = 88.67%) while simultaneously localizing pathological lesions (Localization F1 = 0.4002, Mean IoU = 0.6845). Quantitative explainability benchmarks on *RiceSeg5932* demonstrate that explicit multi-task localization supervision yields a +75.02% increase in Attribution IoU, +59.48% increase in Energy Inside Mask, and +128.9% increase in saliency faithfulness, while reducing healthy leaf false positive proposals by 85.3%. Operating at 14.2 ms latency with a 21.8 MB memory footprint, RiceGuard provides a robust foundation for trustworthy agronomic edge AI.

## B. Future Work
Future extensions include:
1. Extending multi-task supervision to multi-scale feature pyramids (FPN) for dense sub-millimeter fungal spore detection.
2. Integrating self-supervised pre-training on large unlabelled agricultural imagery.
3. Conducting multi-season field trials with automated drone-mounted multispectral cameras across diverse climatic zones.
