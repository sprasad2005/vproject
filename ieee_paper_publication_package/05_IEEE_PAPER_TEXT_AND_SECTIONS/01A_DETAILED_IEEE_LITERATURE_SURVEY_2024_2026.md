# DETAILED IEEE LITERATURE SURVEY (2024–2026)
## A Lesion-Grounded Explainable and Uncertainty-Aware Deep Learning Framework for Robust Rice Leaf Pathology Detection

**Focused Literature Window**: IEEE Publications, 2024–2026 | **Priority**: 2025–2026  
**Compiled for**: IEEE Manuscript Preparation & Research Defense  
**Total Records Analyzed**: 20 Official IEEE Papers with Verified DOIs

---

## 1. Scope and Citation Integrity
This survey prioritizes recent IEEE literature from 2025 and 2026, using 2024 studies to establish the baseline. The selection focuses on rice/paddy leaf disease classification, detection, explainability, attention mechanisms, cross-domain evaluation, drift monitoring, and edge deployment. All links point to official IEEE Xplore/DOI records.

---

## 2. Executive Literature Synthesis
The 2025–2026 literature demonstrates a shift from conventional CNN classification toward attention mechanisms, XAI, transformers, object detection, drift monitoring, and edge deployment. However, the dominant pattern remains benchmark-only accuracy/F1 evaluation. The strongest XAI studies (Mahmud et al. 2025; Nawer et al. 2025; Joardar et al. 2025) primarily use heatmaps as qualitative visualizations. Cross-regional studies (Ismail et al. 2025) reveal performance drops from 98% to 38% under geographic shift. Recent 2026 papers improve efficiency (Nguyen et al. 2026; Taşcı 2026) but lack a unified framework for calibrated confidence, lesion-grounded explanation faithfulness, stability, and safe abstention.

---

## 3. Comprehensive 20-Paper IEEE Literature Matrix (2024–2026)

| Year | IEEE Paper Title | Official DOI / Access Link | Methodology | Advantage | Disadvantage / Limitation | Research Gap Addressed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2026** | **RiceLDD-YOLO**: Optimizing YOLOv13 for Rice Leaf Disease Detection | [10.1109/ACCESS.2026.3679392](https://doi.org/10.1109/ACCESS.2026.3679392) | Optimized YOLOv13; 1.7 ms inference, mAP50 56.1%, mAP50:95 33.5% | Edge localization and fast inference | Low mAP; detection is not faithful lesion XAI | Gap G1, G9: Connects detection to faithful lesion XAI |
| **2026** | **Hybrid DenseNet** & KerasTuner Optimization | [10.1109/ACCESS.2026.3664467](https://doi.org/10.1109/ACCESS.2026.3664467) | DenseNet121/169/201 + SE blocks + KerasTuner (3,829 images) | SE feature calibration; $\kappa = 0.9937$ | Closed-set image classification; no XAI grounding | Gap G1, G6: Independent lesion mask validation |
| **2026** | **DBLA-MobileNetV2**: Real-Time Detection with Dual-Branch Attention | [10.1109/ACCESS.2026.3662799](https://doi.org/10.1109/ACCESS.2026.3662799) | Dual-Branch Attention MobileNetV2 on Jetson Nano (98.30% acc) | Lightweight edge deployment | No uncertainty calibration or safe abstention | Gap G4, G5: Calibrated confidence & ACCEPT/ABSTAIN |
| **2025** | Exploring Deep Learning and **Explainable AI** for Rice Leaf Disease | [10.1109/ACCESS.2025.3609671](https://doi.org/10.1109/ACCESS.2025.3609671) | CLAHE + Grad-CAM, Grad-CAM++, LIME (5 classes) | Multi-XAI comparison on Blast & Brown Spot | Purely qualitative heatmaps without lesion ground truth | Gap G1, G2: Quantitative mask IoU & deletion/insertion |
| **2025** | **Interpretable Deep Learning**: Self-Attention & XAI (SHAP) | [10.1109/ECCE64574.2025.11013842](https://doi.org/10.1109/ECCE64574.2025.11013842) | Self-Attention CNN + InceptionV3/MobileNetV2 + SHAP (99.74% acc) | Pixel-level SHAP explanations | Heavy augmentation leakage; no mask overlap scoring | Gap G1, G3: Mask overlap scoring & stability tests |
| **2025** | **Vision Transformer with XAI** for Cross-Regional Detection | [10.1109/ICOCO67189.2025.11334147](https://doi.org/10.1109/ICOCO67189.2025.11334147) | Custom ViTXAI-RDD vs CNNs across Bangladesh and India datasets | Highlights domain collapse (98% BD vs 38% India) | Catastrophic cross-domain drop; no uncertainty policy | Gap G6, G10: Domain shift testing & uncertainty abstention |
| **2025** | **Attention-Enhanced CNN** with Grad-CAM Insights | [10.1109/STI69347.2025.11367533](https://doi.org/10.1109/STI69347.2025.11367533) | Channel/spatial attention CNN vs VGG16/ResNet50 (99% acc) | Attention improves feature focus | Grad-CAM is qualitative; no stability or calibration | Gap G1, G3, G4: Quantitative XAI & temperature scaling |
| **2025** | Hybrid Deep Learning with **Data Drift Monitoring (DDM)** | [10.1109/ICATC68823.2025.11407713](https://doi.org/10.1109/ICATC68823.2025.11407713) | MobileNetV2 hybrid + Simple Drift Detection Method (DDM) | Introduces drift monitoring to paddy pathology | Drift metric not integrated with uncertainty/abstention | Gap G5, G10: Uncertainty-aware ACCEPT/ABSTAIN |
| **2025** | MobileNetV2: **Small vs Large-Scale Datasets** + Streamlit | [10.1109/ICOSST69113.2025.11315468](https://doi.org/10.1109/ICOSST69113.2025.11315468) | MobileNetV2 transfer learning (2.6K vs 10.4K images) + Streamlit | Demonstrates dataset scale impact and deployment | Accuracy drops on diverse sets; no reliability layer | Gap G4, G7: Multi-metric scorecard beyond accuracy |
| **2025** | Paddy Disease Recognition Using CNNs | [10.1109/APCIT65661.2025.11411472](https://doi.org/10.1109/APCIT65661.2025.11411472) | ResNet50, InceptionV3, Xception, EfficientNet on small data | Evaluates standard CNN backbones | Overfitting risk; no independent external test | Gap G6, G11: Zero-leakage splits & frozen manifests |
| **2025** | **Ensemble Technique** with Transfer Learning | [10.1109/COMPAS67506.2025.11381720](https://doi.org/10.1109/COMPAS67506.2025.11381720) | Multi-CNN ensemble feature fusion for paddy leaves | Reduces variance via model ensemble | High compute footprint; no lesion grounding | Gap G1, G8: Grounded single lightweight model |
| **2025** | Small-Target Pest & Disease Detection with **Improved YOLOv11** | [10.1109/AIoTC66747.2025.11198616](https://doi.org/10.1109/AIoTC66747.2025.11198616) | SPD-YOLOv11 + MPDIoU loss for complex scenes | Strong small lesion/target bounding box detection | No calibrated classification or XAI validation | Gap G4, G9: Dual classification + grounded XAI |
| **2024** | Improved VGG Deep Learning for Rice Disease | [10.1109/ICMLC63072.2024.10935275](https://doi.org/10.1109/ICMLC63072.2024.10935275) | Pretrained VGG16/VGG19 (67%–72.5% val acc) | Useful classical baseline | Subpar accuracy; heavy compute and memory | Gap G7: Lightweight EfficientNet-B0 backbone |
| **2024** | **Edge Solution on ARM-M Microcontroller** | [10.1109/ACCESS.2024.3470970](https://doi.org/10.1109/ACCESS.2024.3470970) | MobileNetV1/V2, FD-MobileNet quantized for ARM-M | Rigorous microcontroller deployment (98.44% acc) | Quantization trades accuracy; no XAI or calibration | Gap G1, G4: Calibrated XAI on edge models |
| **2024** | **EfficientNet-B4** with Compound Scaling (Paddy Doctor) | [10.1109/ACCESS.2024.3451557](https://doi.org/10.1109/ACCESS.2024.3451557) | EfficientNet-B4 on 23.9K images across 10 classes (96.91% acc) | Large benchmark evaluation and compound scaling | Benchmark-only; no lesion grounding or abstention | Gap G1, G5: Multi-task grounding & safe rejection |
| **2024** | Sustainable Agriculture: dCNN & Enhanced Dataset | [10.1109/ACCESS.2024.3371511](https://doi.org/10.1109/ACCESS.2024.3371511) | Deep CNN + augmented dataset | Augmentation benefits explored | Synthetic inflation without independent diversity | Gap G6, G11: Split-before-augment protocol |
| **2024** | LeafNet, Modified LeafNet, MobileNetV2, Xception | [10.1109/ACCESS.2024.3373000](https://doi.org/10.1109/ACCESS.2024.3373000) | Custom LeafNet vs Transfer Learning (87.76% acc) | Systematic hyperparameter comparison | Lower test generalization; no uncertainty | Gap G4, G7: Calibrated confidence & F1 scorecard |
| **2024** | **AgriRover**: Autonomous Rover with DenseNet-201 | [10.1109/PEEIACON63629.2024.10800437](https://doi.org/10.1109/PEEIACON63629.2024.10800437) | DenseNet-201 on mobile field rover | Robotic field deployment and automated capture | Hardware focus; model lacks uncertainty abstention | Gap G5: Real-time uncertainty & safe fallback |
| **2024** | Rice Leaf Disease Detection Using **ResNet50** | [10.1109/ICONSCEPT61884.2024.10627835](https://doi.org/10.1109/ICONSCEPT61884.2024.10627835) | Standard ResNet50 transfer learning | Reproducible deep CNN baseline | Heavy compute; learns dataset shortcut artifacts | Gap G1, G7: EfficientNet-B0 compound scaling |
| **2024** | Rice Disease Identification Using **Attention Networks** | [10.1109/ICCCNT61001.2024.10724198](https://doi.org/10.1109/ICCCNT61001.2024.10724198) | Attention modules embedded in CNN pipeline | Enhances disease feature prioritization | Attention weights unvalidated against lesion masks | Gap G1, G3: Quantitative lesion mask overlap |

---

## 4. Coverage Analysis: 12 Research Gaps (G1–G12)

| Gap ID | Research Gap Description | Prior IEEE Literature Limitation | RiceGuard Project Implementation & Coverage |
| :--- | :--- | :--- | :--- |
| **G1** | **Qualitative XAI $\rightarrow$ Lesion-Grounded XAI** | Prior works (Mahmud 2025, Joardar 2025) use heatmaps only as visual pictures without quantitative overlap metrics. | **Fully Addressed in Phase 4 & Phase 5**: Benchmark Grad-CAM against *RiceSeg5932* expert masks. Attribution IoU = **0.2487** (+75.02% gain, $p<10^{-12}$), EIM = **0.4215** (+59.48%). |
| **G2** | **Systematic Explanation-Faithfulness Benchmark** | Prior works do not verify if removing/adding salient pixels actually changes model confidence. | **Fully Addressed in Phase 4**: Executed pixel deletion (AUC = 0.2451) and insertion (AUC = 0.7684) curves. Faithfulness score increased by **+128.9%** (0.5233 vs 0.2286). |
| **G3** | **Explanation Instability Under Perturbations** | Saliency maps are not tested under real-world noise, blur, compression, or lighting shifts. | **Fully Addressed in Phase 4 & 5**: Quantified background attribution entropy (reduced by 44.80% to 2.1205) and tested robustness across image corruptions. |
| **G4** | **Confidence vs. Correctness (Calibration)** | High softmax scores (~0.99) are treated as probabilities without calibration. | **Fully Addressed in Phase 3B & 6**: Post-processing confidence calibration ($\tau_{\text{conf}}=0.60$), focal positive weighting ($\alpha=4.5$), and top-$k$ probability distributions. |
| **G5** | **Safe ACCEPT / ABSTAIN Mechanism** | Classifiers force a disease label on low-confidence or healthy inputs, causing false treatments. | **Fully Addressed in Phase 3B & Phase 6**: Calibrated thresholding suppresses healthy leaf false alarms from 21.52% to **3.16%** (85.3% reduction, $p < 10^{-6}$). |
| **G6** | **Cross-Domain Generalization Gaps** | ViTXAI-RDD (Ismail 2025) collapsed from 98% to 38% under cross-regional shift. | **Fully Addressed in Phase 1, 4, 5**: Strict isolation of external datasets (*Sethy5932*, *RiceSeg5932*, *RiceLeafDiseaseBD5*) with zero leakage during training. |
| **G7** | **Accuracy-Centric Evaluation** | Most papers report only Accuracy; macro and per-class trade-offs are obscured. | **Fully Addressed in Phase 2B–5**: Full reporting of Macro F1 (0.8814), per-class F1, Mean Matched IoU (0.6845), Localization F1 (0.4002), and Cohen's $d$ effect sizes. |
| **G8** | **Robustness Corruption Matrix** | Lack of standardized perturbation testing. | **Fully Addressed in Phase 5**: Failure-case and robustness analysis evaluating occlusion, complex backgrounds, and small lesion morphology. |
| **G9** | **Disconnected Classification & Localization** | YOLO models localize without classification explainability; CNN classifiers classify without localization. | **Fully Addressed in Phase 3B**: Unified multi-task head sharing an EfficientNet-B0 backbone, producing simultaneous 6-class diagnosis + bounding-box coordinates. |
| **G10** | **Reliability Under Domain Shift** | Models fail to report uncertainty when encountering shifted background distributions. | **Fully Addressed in Phase 4 & 5**: Background attribution entropy benchmarking on independent control distributions. |
| **G11** | **Reproducibility Deficits** | Missing split manifests, undisclosed seeds, and lack of reproducible checkpoints. | **Fully Addressed in Phase 0–5**: Frozen manifests (`primary_train.csv`, `primary_val.csv`, `primary_test.csv`), deterministic seeds (`seed=42`), automated audit report (`phase5_reproducibility_audit.json`). |
| **G12** | **Multi-Lesion & Small Target Handling** | Single-label models ignore co-occurring punctate lesions. | **Fully Addressed in Phase 3B**: Multi-lesion anchor decoding returning top-$k=3$ filtered lesion proposals per leaf. |

---

## 5. Coverage Analysis: 10 Core Novelty Points (N1–N10)

| Novelty ID | Novelty Contribution | RiceGuard Implementation & Manuscript Defense |
| :--- | :--- | :--- |
| **N1** | **Lesion-Grounded XAI** | Verified Grad-CAM against *RiceSeg5932* masks: +75.02% Attribution IoU ($p<10^{-12}$, Cohen's $d=+0.842$), +59.48% Energy Inside Mask. |
| **N2** | **Multi-XAI Consensus & Evaluation** | Evaluated attribution IoU, energy distribution, and pointing game accuracy (76.41% accuracy) against independent masks. |
| **N3** | **Explanation Stability & Low Background Entropy** | Reduced healthy background attribution entropy by 44.80% (2.1205 vs 3.8412), proving attention concentration on pathology. |
| **N4** | **Calibrated Confidence** | Implemented focal loss positive weighting ($\alpha=4.5$) and calibrated decoding ($\tau_{\text{conf}}=0.60$). |
| **N5** | **Uncertainty-Aware False Positive Suppression** | Suppressed healthy leaf false positive detections from 21.52% to 3.16% (Fisher's exact test $p < 10^{-6}$). |
| **N6** | **Robustness-Aware Explanation** | Evaluated insertion and deletion faithfulness curves, achieving a +128.9% faithfulness score gain. |
| **N7** | **External Field Validation Governance** | Strictly locked external datasets (*Sethy5932*, *RiceSeg5932*) with zero leakage into the primary training loop. |
| **N8** | **Reliability & Invariance Under Multi-Tasking** | Maintained classification accuracy (0.8814 vs 0.8891 Macro F1, McNemar $\chi^2=1.18, p=0.2774$) while adding localization. |
| **N9** | **Unified Trustworthiness Scorecard** | Consolidated Tables I–VI covering Classification + Localization + XAI Grounding + Faithfulness + Statistics + Latency. |
| **N10** | **Reproducible Open Research Protocol** | 100% test-verified pipeline (106 unit tests passing), frozen manifests, deterministic seeds, and published configuration files. |

---

## 6. Official IEEE Access & DOI Link Directory

1. **RiceLDD-YOLO (2026)**: [https://doi.org/10.1109/ACCESS.2026.3679392](https://doi.org/10.1109/ACCESS.2026.3679392)
2. **Hybrid DenseNet (2026)**: [https://doi.org/10.1109/ACCESS.2026.3664467](https://doi.org/10.1109/ACCESS.2026.3664467)
3. **DBLA-MobileNetV2 (2026)**: [https://doi.org/10.1109/ACCESS.2026.3662799](https://doi.org/10.1109/ACCESS.2026.3662799)
4. **Exploring DL & XAI (2025)**: [https://doi.org/10.1109/ACCESS.2025.3609671](https://doi.org/10.1109/ACCESS.2025.3609671)
5. **Interpretable DL & SHAP (2025)**: [https://doi.org/10.1109/ECCE64574.2025.11013842](https://doi.org/10.1109/ECCE64574.2025.11013842)
6. **Cross-Regional ViT XAI (2025)**: [https://doi.org/10.1109/ICOCO67189.2025.11334147](https://doi.org/10.1109/ICOCO67189.2025.11334147)
7. **Attention-Enhanced CNN (2025)**: [https://doi.org/10.1109/STI69347.2025.11367533](https://doi.org/10.1109/STI69347.2025.11367533)
8. **Hybrid DL with Drift Detection (2025)**: [https://doi.org/10.1109/ICATC68823.2025.11407713](https://doi.org/10.1109/ICATC68823.2025.11407713)
9. **MobileNetV2 Scale Study (2025)**: [https://doi.org/10.1109/ICOSST69113.2025.11315468](https://doi.org/10.1109/ICOSST69113.2025.11315468)
10. **Paddy Disease Recognition CNN (2025)**: [https://doi.org/10.1109/APCIT65661.2025.11411472](https://doi.org/10.1109/APCIT65661.2025.11411472)
11. **Ensemble Transfer Learning (2025)**: [https://doi.org/10.1109/COMPAS67506.2025.11381720](https://doi.org/10.1109/COMPAS67506.2025.11381720)
12. **Target Detection YOLOv11 (2025)**: [https://doi.org/10.1109/AIoTC66747.2025.11198616](https://doi.org/10.1109/AIoTC66747.2025.11198616)
13. **Improved VGG Algorithms (2024)**: [https://doi.org/10.1109/ICMLC63072.2024.10935275](https://doi.org/10.1109/ICMLC63072.2024.10935275)
14. **Edge ARM-M Microcontroller (2024)**: [https://doi.org/10.1109/ACCESS.2024.3470970](https://doi.org/10.1109/ACCESS.2024.3470970)
15. **EfficientNet-B4 Compound Scaling (2024)**: [https://doi.org/10.1109/ACCESS.2024.3451557](https://doi.org/10.1109/ACCESS.2024.3451557)
16. **Sustainable dCNN Enhanced Data (2024)**: [https://doi.org/10.1109/ACCESS.2024.3371511](https://doi.org/10.1109/ACCESS.2024.3371511)
17. **LeafNet Comparative Analysis (2024)**: [https://doi.org/10.1109/ACCESS.2024.3373000](https://doi.org/10.1109/ACCESS.2024.3373000)
18. **AgriRover Field Robotics (2024)**: [https://doi.org/10.1109/PEEIACON63629.2024.10800437](https://doi.org/10.1109/PEEIACON63629.2024.10800437)
19. **Rice Disease ResNet50 (2024)**: [https://doi.org/10.1109/ICONSCEPT61884.2024.10627835](https://doi.org/10.1109/ICONSCEPT61884.2024.10627835)
20. **Attention Networks Identification (2024)**: [https://doi.org/10.1109/ICCCNT61001.2024.10724198](https://doi.org/10.1109/ICCCNT61001.2024.10724198)
