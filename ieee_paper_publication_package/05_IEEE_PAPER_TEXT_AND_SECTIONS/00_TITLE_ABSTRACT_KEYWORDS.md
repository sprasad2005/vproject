# RiceGuard: Lesion-Grounded Multi-Task Deep Learning for Interpretable Rice Leaf Pathology and Edge Diagnosis

**Authors**:
- **Varad Hiraman Parate** (PRN: 22311382, Roll No: BTech3A431078, `varad.22311382@viit.ac.in`)
- **Dhruv A. Gholap** (PRN: 22311579, Roll No: BTech3A431002, `dhruv.22311579@viit.ac.in`)
- **Hrushikesh Suryawanshi** (PRN: 22310354, Roll No: BTech3A431083, `hrushikesh.22310354@viit.ac.in`)
- **Arnav Kulkarni** (PRN: 22310597, Roll No: BTech3A431034, `arnav.22310597@viit.ac.in`)
  *Department of Information Technology, Vishwakarma Institute of Information Technology, Pune, India*
- **Dr. Shalini Wankhade** (`Shalini.wankhade@vit.edu`)
  *Department of Information Technology, Vishwakarma Institute of Technology, Pune, India*

**Target Venue**: IEEE Transactions on Agri-Food Electronics / IEEE Access / IEEE ICIP

---

### Abstract
Deep convolutional neural networks (CNNs) have shown remarkable accuracy in automated crop pathology classification. However, standard classification models often act as black boxes that exploit spurious background correlations (such as soil textures, lighting variations, or border artifacts) rather than bona fide pathological lesions. This vulnerability undermines trust and leads to severe performance degradation under field domain shifts. In this work, we propose **RiceGuard**, an interpretable, lesion-grounded multi-task deep architecture built upon an optimized EfficientNet-B0 backbone. RiceGuard jointly optimizes 6-class rice leaf disease classification and spatial bounding-box lesion localization using a calibrated multi-task loss with focal class-imbalance weighting. Evaluated on the standardized *RiceLeafDiseaseBD* benchmark (706 test samples across Healthy, Blast, Brown Spot, Leaf Smut, Tungro, and Sheath Blight), RiceGuard achieves a classification Macro F1-score of **0.8814** (Accuracy: **88.67%**) and a lesion localization F1-score of **0.4002** (Mean Matched IoU: **0.6845**), while operating with only **5.35M parameters** (21.8 MB) at **14.2 ms inference latency** on an NVIDIA GTX 1650. To rigorously validate interpretability, we benchmark Grad-CAM saliency heatmaps against independent ground-truth lesion masks from *RiceSeg5932*. RiceGuard demonstrates a **+75.02% increase in Attribution IoU** (0.2487 vs. 0.1421, $p < 0.0001, d = +0.842$) and a **+59.48% increase in Energy Inside Mask** (0.4215 vs. 0.2643) compared to pure classification baselines, while suppressing false positive lesion proposals on healthy leaves by 85.3% (FPR = 3.16%). Deletion and insertion faithfulness benchmarks confirm that RiceGuard's decision pathways are strictly grounded in pathognomonic lesion features.

**IEEE Keywords**: Agriculture pathology, crop disease diagnosis, multi-task learning, explainable artificial intelligence (XAI), lesion localization, Grad-CAM grounding, edge computing, EfficientNet.
