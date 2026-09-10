# Project Plan 2.0: Lifecycle, Milestones, and Progression

## Executive Summary
This document outlines the structured lifecycle and operational schedule for the **RiceGuard** project. The plan provides a transparent mapping of completed research and engineering phases alongside in-progress documentation deliverables and future post-FTR milestones.

```mermaid
gantt
    title RiceGuard Project Lifecycle: Progression and Milestone Roadmap
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Phase 1: Data & Governance [COMPLETED]
    Dataset Assembly & Validation Splits (N=3,355)     :done, p1_1, 2026-01-05, 2026-01-20
    RiceSeg5932 Mask Isolation & Verification (N=4,348) :done, p1_2, 2026-01-15, 2026-01-28

    section Phase 2B: Baseline Model [COMPLETED]
    EfficientNet-B0 Classification Training           :done, p2_1, 2026-01-29, 2026-02-12
    Baseline Performance Benchmarking (Acc: 90.12%)    :done, p2_2, 2026-02-08, 2026-02-18

    section Phase 3/3B: Multi-Task & Calibration [COMPLETED]
    Multi-Task 7x7 Grid Architecture Implementation    :done, p3_1, 2026-02-19, 2026-03-02
    Phase 3B Positive-Weight Loss Tuning (w_pos=10.0)  :done, p3_2, 2026-03-01, 2026-03-12
    Validation Grid Sweeps (tau=0.60, K=3 Selected)    :done, p3_3, 2026-03-10, 2026-03-20

    section Phase 4: Quantitative XAI [COMPLETED]
    Layer-Hooked Grad-CAM Attribution Pipeline        :done, p4_1, 2026-03-21, 2026-03-30
    Grounding Evaluation on N=4,348 RiceSeg Masks     :done, p4_2, 2026-03-28, 2026-04-10
    Border Attention & Faithfulness Curves             :done, p4_3, 2026-04-05, 2026-04-16

    section Phase 5: Scientific Audit [COMPLETED]
    Statistical Hypothesis Testing (McNemar, t-tests)  :done, p5_1, 2026-04-17, 2026-04-25
    Reproducibility Locking & Checkpoint Archival      :done, p5_2, 2026-04-22, 2026-05-02

    section Phase 6: Application & Client [COMPLETED]
    FastAPI Asynchronous Backend Engine Implementation :done, p6_1, 2026-05-03, 2026-05-15
    React 19 / Vite 6 Dual-Pane User Interface         :done, p6_2, 2026-05-12, 2026-05-28
    Docker & Vercel Production Environment Preparation :done, p6_3, 2026-05-25, 2026-06-05

    section Milestone Review [IN PROGRESS]
    FTR-1 Documentation Suite Compilation             :active, m1_1, 2026-06-06, 2026-06-20

    section Future Roadmap [PLANNED]
    INT8 ONNX / TFLite Edge Runtime Quantization       :planned, f1_1, 2026-06-25, 2026-07-25
    Multi-Center Agronomic Field Pilot Testing         :planned, f1_2, 2026-07-20, 2026-09-15
```

---

## Detailed Milestone and Phase Breakdown

| Phase ID & Description | Status | Key Deliverables | Verification Artifact |
| :--- | :---: | :--- | :--- |
| **Phase 1: Dataset Governance** | **Completed** | Stratified 70/10/20 splits ($N = 3,355$); isolation of RiceSeg5932 ($N = 4,348$ masks) for XAI grounding. | `splits/train.csv`, `splits/val.csv`, `splits/test.csv` |
| **Phase 2B: Baseline Classification** | **Completed** | EfficientNet-B0 single-task classifier; AdamW + Cosine Annealing; label smoothing. | `checkpoints/baseline_best.pt` (Acc: $90.12\%$, Macro F1: $0.8891$) |
| **Phase 3: Multi-Task Architecture** | **Completed** | Shared backbone bifurcated into 6-class head and $7 \times 7 \times 5$ spatial grid localization head. | `checkpoints/multitask_best.pt`, multi-task loss module |
| **Phase 3B: Localization Calibration** | **Completed** | Positive-weight loss ($w_{\text{pos}}=10.0$); validation-only sweeps ($\tau=0.60, K=3$). | `checkpoints/phase3b_calibrated.pt` (Mean IoU: $0.6423$, $-74.6\%$ clutter) |
| **Phase 4: Quantitative XAI Grounding** | **Completed** | Grad-CAM attribution extraction on `features.8`; quantitative evaluation across $N=4,348$ masks. | `results/phase4_xai_grounding.csv` (EIM: $9.64\%$, Attr-IoU: $0.0864$) |
| **Phase 5: Scientific Audit & Rigor** | **Completed** | Statistical significance tests (McNemar's test, paired $t$-tests, Wilcoxon signed-rank tests). | `results/statistical_report.json`, frozen experiment configs |
| **Phase 6: Full-Stack Web Application** | **Completed** | FastAPI backend (`/predict`), React 19 / Vite 6 frontend, dual-pane canvas overlay renderer. | Operational app at `http://localhost:5173` $\leftrightarrow$ `http://localhost:8000` |
| **FTR-1 Milestone Review** | **In Progress** | Comprehensive 14-document FTR-1 academic documentation package. | `FTR1/` documentation repository |
| **Future Phase 7: Mobile Edge Quantization** | **Planned** | INT8 ONNX / TFLite model export; offline edge execution on smartphone hardware. | Quantized model artifacts, mobile benchmark reports |
| **Future Phase 8: Multi-Center Field Trials** | **Planned** | Real-world diagnostic evaluation across regional agricultural extension research stations. | Multi-center agronomic validation dataset & report |

---

## Risk Management Matrix

| Risk ID | Potential Risk Event | Likelihood | Impact | Mitigation Strategy Implemented in RiceGuard |
| :---: | :--- | :---: | :---: | :--- |
| **R-01** | Multi-task learning induces severe classification accuracy degradation. | Medium | High | Retained shared EfficientNet-B0 backbone with balanced loss scaling ($\lambda_{\text{box}} = 5.0$). Verified via McNemar's test ($p = 0.449$). |
| **R-02** | Dense spatial grid regressors generate massive false alarms on healthy foliage. | High | High | Formulated positive-weight loss ($w_{\text{pos}} = 10.0$) and validation-only threshold calibration ($\tau = 0.60, K = 3$), cutting healthy FPR by $49.0\%$ ($p < 10^{-6}$). |
| **R-03** | Data leakage occurs during threshold and hyperparameter tuning. | Medium | Critical | Strictly enforced validation-only calibration protocol ($N = 334$), completely isolating the test split ($N = 674$) until final evaluation. |
| **R-04** | XAI visual heatmaps suffer from confirmation bias and qualitative cherry-picking. | High | High | Automated quantitative benchmarking against $N = 4,348$ independent biological lesion masks from RiceSeg5932 using Energy-Inside-Mask and Attribution IoU. |
| **R-05** | Heavy model inference latency exceeds edge device operational budgets. | Medium | Medium | Utilized compact EfficientNet-B0 backbone ($4.02\text{M}$ params, $15.61\text{ MB}$), achieving $10.46\text{ ms}$ GPU latency. |

---

## Resource Allocation and Hardware Utilization
- **Computational Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU ($4\text{ GB}$ VRAM), AMD/Intel x86_64 CPU, $16\text{ GB}$ Host RAM.
- **Software Tooling**: Python 3.10+, PyTorch 2.6.0, CUDA 12.4, FastAPI 0.115+, React 19, Vite 6, Tailwind CSS 4.
- **Version Control & Governance**: Git repository with structured commit tags, automated test validation (`pytest`), and code formatting (`ruff`).
