# RiceGuard Official IEEE Conference Research Paper

## Paper Metadata
* **Title**: RiceGuard: Weakly Supervised Multi-Task Rice Leaf Disease Classification, Lesion Localization, and Quantitative Explainability Validation
* **Target Publication Format**: IEEE Conference (Two-Column Standard Template)
* **Maximum Page Budget**: Maximum 7 Pages (Nominal compiled length: ~6.2–6.8 pages)
* **Document Class**: `\documentclass[conference]{IEEEtran}`
* **References**: Exact 20 approved IEEE publication records `[1]`–`[20]`

---

## Directory Structure
```
RiceGuard_IEEE_Paper_Official/
├── main.tex                       # Complete Overleaf-ready LaTeX manuscript
├── references.bib                 # Official BibTeX bibliography
├── README.md                      # Documentation and compilation instructions
├── figures/
│   ├── fig1_pipeline.png          # Architecture and multi-task pipeline
│   ├── fig2_tradeoff.png          # Pareto classification vs. localization trade-off
│   ├── fig3_confusion.png         # Normalized test confusion matrices
│   ├── fig4_per_class_f1.png      # Per-class F1-score comparison
│   ├── fig5_xai_summary.png       # Quantitative XAI grounding & border shortcut analysis
│   └── fig6_deletion_curves.png   # Stepwise pixel deletion faithfulness curves
└── source_notes/
    └── artifact_mapping.md        # Detailed metric-to-repository provenance matrix
```

---

## Overleaf Compilation Instructions
1. Log in to [Overleaf](https://www.overleaf.com/).
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Select `RiceGuard_IEEE_Paper_Official.zip`.
4. Ensure the compiler is set to **pdfLaTeX** and the main file is set to `main.tex`.
5. Click **Recompile**.
