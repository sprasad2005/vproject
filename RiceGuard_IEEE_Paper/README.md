# RiceGuard: IEEE Research Paper LaTeX Project (Overleaf Ready)

This directory contains the complete, self-contained, publication-ready IEEE LaTeX manuscript for the **RiceGuard** research project.

---

## 🚀 How to Upload to Overleaf

1. Download or locate `RiceGuard_IEEE_Paper.zip` in the root directory.
2. Go to [Overleaf](https://www.overleaf.com) and log in.
3. Click **New Project** $ightarrow$ **Upload Project**.
4. Select `RiceGuard_IEEE_Paper.zip`.
5. In Overleaf project settings, ensure:
   - **Compiler**: `pdfLaTeX` or `LaTeX`
   - **Main document**: `main.tex`
6. Click **Recompile**. The complete double-column IEEE formatted manuscript will compile cleanly.

---

## 📁 Directory Structure

```
RiceGuard_IEEE_Paper/
├── main.tex                    <-- Master compilation document
├── IEEEtran.cls                <-- Official IEEE LaTeX class
├── README.md                   <-- Compilation instructions & guide
├── compile_instructions.md     <-- Step-by-step local & Overleaf compilation guide
├── claim_verification.md       <-- Comprehensive 44-claim scientific audit matrix
│
├── sections/                   <-- Modular paper sections
│   ├── 01_abstract.tex
│   ├── 02_keywords.tex
│   ├── 03_introduction.tex
│   ├── 04_related_work.tex
│   ├── 05_research_gap_contributions.tex
│   ├── 06_methodology.tex
│   ├── 07_experimental_setup.tex
│   ├── 08_results.tex
│   ├── 09_discussion.tex
│   ├── 10_limitations.tex
│   └── 11_conclusion.tex
│
├── tables/                     <-- All 10 IEEE LaTeX table files
│   ├── dataset_summary.tex
│   ├── training_configuration.tex
│   ├── model_architecture.tex
│   ├── main_model_comparison.tex
│   ├── per_class_results.tex
│   ├── localization_results.tex
│   ├── xai_results.tex
│   ├── statistical_results.tex
│   ├── faithfulness_results.tex
│   └── reproducibility.tex
│
├── figures/                    <-- High-resolution publication figures
│   ├── model_evolution_overview.png
│   ├── classification_localization_tradeoff.png
│   ├── per_class_f1_comparison.png
│   ├── confusion_matrix_comparison.png
│   ├── xai_grounding_summary.png
│   └── failure_cases_baseline_vs_multitask.png
│
└── bibliography/
    └── references.bib          <-- Complete 20 IEEE official citations [1]-[20]
```

---

## ⚖️ Scientific Integrity & Reproducibility Guarantee
- **Zero Fabricated Results**: Every numerical result is 100% matched with stored artifacts in `claim_verification.md`.
- **Honest Trade-off Reporting**: Classification baseline superiority in pure Macro F1 (0.8891 vs 0.8751) is transparently reported and analyzed.
- **Strict Data Governance**: RiceSeg5932 is strictly documented as an XAI evaluation dataset, with zero leakage into model training.
