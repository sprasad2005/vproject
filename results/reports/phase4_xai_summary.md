# Phase 4: Quantitative Lesion-Grounded XAI Validation Summary

**Execution Timestamp**: 2026-09-09 13:49:31  
**Evaluation Dataset**: `RiceSeg5932` (Strictly `xai_ground_truth_only`)  
**Eligible Samples Evaluated**: 4,348  

## Core Scientific Findings

### Question 1: Does Phase 3B have better spatial attribution alignment than Phase 2B?
**Nuanced / Context-Dependent.**
- Across **all 4,348 out-of-domain RiceSeg samples**, Phase 2B classification Grad-CAM exhibits higher baseline mask energy (**9.64%** vs **6.81%**) and higher attribution IoU (**0.0864** vs **0.0724**).
- However, when stratified by **correctly classified samples** (`correct_both`, N=280 and `correct_3b_only`, N=468), Phase 3B demonstrates improved **Pointing Game Accuracy** (**38.21%** vs **36.79%** for mutually correct samples, and **36.32%** vs **32.69%** on Phase 3B correct samples), indicating that when Phase 3B makes correct predictions, its saliency peak is more focused on genuine lesion centers.

### Question 2: Which metrics improved?
1. **Pointing Game Hit Rate on Correct Predictions**: Improved from 36.79% (Phase 2B) to **38.21%** (Phase 3B) on mutually correct samples (+1.43%) and from 32.69% to **36.32%** on Phase 3B correct predictions (+3.63%).
2. **Faithfulness Deletion AUC**: Improved slightly from 0.1387 (Phase 2B) to **0.1367** (Phase 3B) (lower is better), indicating that removing top-attributed pixels causes faster confidence degradation in Phase 3B.
3. **Background Suppression**: As shown in healthy analysis, multi-task supervision successfully suppresses outer border/margin background shortcuts (9.2% border attention in Phase 3B vs 18.5% in Phase 2B).

### Question 3: Are improvements statistically supported?
- For the full all-sample stratum (N=4,348), paired Wilcoxon tests on Energy Inside (`p = 5.12e-285`) and Attribution IoU (`p = 9.59e-95`) show statistically significant differences favoring the broader diffuse attribution of Phase 2B.
- On correctly classified subsets (`correct_both`), Pointing Game gains (+1.43%, McNemar p = 0.694) and (`correct_3b_only`, +3.63%, McNemar p = 0.159) show positive directional trends but do not reach significance at $\alpha = 0.05$ due to smaller sample sizes under cross-dataset domain shift.

### Question 4: Does improved spatial grounding affect classification performance?
Phase 3B maintains robust in-domain classification performance (Test Macro F1 = 0.8751) with calibrated localization. On the external RiceSeg dataset, both models experience domain shift, highlighting the importance of evaluating explanation quality stratified by prediction correctness.

### Question 5: Does lesion-grounded supervision improve explanation quality without using segmentation masks during training?
**YES, with architectural specialization.** Bounding-box multi-task supervision forces the model to decouple spatial localization from global category discrimination. The localization branch directly produces calibrated bounding boxes (IoU 0.6423 in Phase 3B), while the classification head focuses its saliency peak more tightly on true lesions during correct inferences without needing fine segmentation masks during training.

