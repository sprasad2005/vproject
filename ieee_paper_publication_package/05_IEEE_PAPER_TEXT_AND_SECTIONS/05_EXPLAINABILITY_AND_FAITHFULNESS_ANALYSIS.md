# Section V: Explainability & Faithfulness Analysis

## A. Quantitative Grounding Benchmark on RiceSeg5932
To scientifically validate whether RiceGuard's internal representations attend to true pathological lesions, we benchmarked Grad-CAM saliency heatmaps against independent pixel-level lesion masks from *RiceSeg5932* (**Table IV**):
- **Attribution IoU**: RiceGuard achieved **0.2487**, representing a **+75.02% relative gain** over the Phase 2B classification baseline (0.1421), with extreme statistical significance ($t = 14.82, p < 10^{-12}$, Cohen's $d = +0.842$).
- **Energy Inside Mask (EIM)**: RiceGuard concentrated **42.15%** of its total gradient energy strictly inside true lesion boundaries (vs. 26.43% for Phase 2B, Wilcoxon $W = 4210.5, p < 10^{-10}, d = +0.915$).
- **Pointing Game Accuracy**: The spatial location of maximum Grad-CAM activation fell within the true lesion boundary in **76.41%** of test samples for RiceGuard (vs. 58.24% for Phase 2B, +31.2% relative improvement).
- **Healthy Background Entropy**: On healthy control leaves, background saliency dispersion was reduced by **44.80%** (2.1205 vs. 3.8412), demonstrating dramatic suppression of spurious hallucinated saliency.

## B. Faithfulness via Deletion and Insertion Benchmarks
To confirm that saliency heatmaps reflect actual model decision dynamics rather than visualization artifacts, we evaluated pixel deletion and insertion curves (**Table VI**, Fig. 7a, 7b):
- **Deletion AUC**: Progressive removal of salient pixels induced significantly faster confidence degradation in RiceGuard ($	ext{AUC} = 0.2451$) than in the baseline ($	ext{AUC} = 0.3842$). Lower deletion AUC proves that highlighted pixels are truly necessary for the diagnostic decision.
- **Insertion AUC**: Progressive re-introduction of salient pixels into a blank canvas restored diagnostic confidence significantly faster in RiceGuard ($	ext{AUC} = 0.7684$) than in the baseline ($	ext{AUC} = 0.6128$). Higher insertion AUC proves that highlighted pixels are sufficient for correct classification.
- **Overall Faithfulness Metric**: RiceGuard achieved a Faithfulness Score ($	ext{AUC}_{	ext{ins}} - 	ext{AUC}_{	ext{del}}$) of **0.5233**, representing a **+128.9% gain** over the baseline (0.2286).
