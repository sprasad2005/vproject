# RiceGuard IEEE Paper Publication Package & Research Portfolio

This directory contains the complete, publication-ready research artifacts, 300 DPI high-resolution figures, LaTeX/Markdown tables, model comparisons, selected model specifications, IEEE paper text drafts, 20-paper literature survey (2024–2026), and raw experimental metrics for the **RiceGuard** project.

---

## 📁 Package Directory Organization

### `01_FIGURES_AND_CHARTS/`
- **`high_resolution_300dpi/`**: Contains all 8 core publication figures (`Fig1` to `Fig8`) and 8 supplementary figures (`FigS1` to `FigS8`) rendered in 300 DPI with IEEE formatting:
  - `Fig1_Model_Evolution_Overview.png`: Macro F1 vs. Grounding vs. VRAM efficiency.
  - `Fig2_Classification_Localization_Tradeoff.png`: Pareto frontier of multi-task performance.
  - `Fig3_Per_Class_F1_Comparison.png`: Per-class F1 across 6 pathology categories.
  - `Fig4_Confusion_Matrix_Comparison.png`: 3-way normalized confusion matrix grid.
  - `Fig5_XAI_Grounding_Summary.png`: Attribution IoU, EIM, Pointing Game, and Entropy.
  - `Fig6_Failure_Cases_Comparison.png`: Qualitative failure analysis under severe occlusion/clutter.
  - `Fig7A_XAI_Deletion_Curves.png`: Grad-CAM deletion curves (faithfulness benchmark).
  - `Fig7B_XAI_Insertion_Curves.png`: Grad-CAM insertion curves (faithfulness benchmark).
  - `Fig8_GradCAM_Qualitative_Comparison.png`: Full-resolution qualitative Grad-CAM vs. Ground-truth masks.
  - `FigS1` through `FigS8`: Training dynamics, loss decomposition, dataset distributions, and RiceSeg pairings.
- **`figure_captions_and_descriptions.md`**: Formal IEEE captions and section placement guide.
- **`FIGURE_INDEX.csv`**: Manifest table linking figure ID, filename, resolution, and caption.

### `02_IEEE_TABLES_LATEX_AND_MD/`
All tables provided in `.tex` (LaTeX ready for `IEEEtran.cls`), `.md` (Markdown), and `.csv` (Spreadsheet):
- **Table I**: Architectural Backbone Screening (ResNet-50 vs. ViT-B/16 vs. EfficientNet-B0).
- **Table II**: Master Evolution & Cross-Phase Performance Benchmarks (Phase 2B vs. 3 vs. 3B).
- **Table III**: Per-Class Classification Performance across 6 disease categories.
- **Table IV**: Quantitative Explainability (XAI) Grounding Benchmark on *RiceSeg5932*.
- **Table V**: Statistical Significance Hypothesis Tests ($p$-values, $t$-statistics, Cohen's $d$).
- **Table VI**: Explainability Faithfulness Benchmark (Deletion & Insertion AUC).
- **`all_tables_compiled.tex`**: Single drop-in LaTeX snippet containing all 6 tables.

### `03_MODEL_COMPARISONS_AND_EVALUATION/`
- **`research_gaps_and_novelty_coverage.md`**: Direct mapping of all 12 IEEE Research Gaps (G1–G12) and 10 Novelty points (N1–N10).
- **`model_comparison_master_matrix.csv`**: Side-by-side numerical comparison of all 5 architectures across 11 key metrics.
- **`backbone_comparison_deep_dive.md`**: Rigorous empirical and theoretical justification comparing ResNet-50, ViT-B/16, and EfficientNet-B0.
- **`multitask_ablation_analysis.md`**: Step-by-step ablation analyzing the impact of localization heads, anchor priors, and positive loss weights.

### `04_SELECTED_MODEL_JUSTIFICATION/`
- **`selected_model_architecture_spec.md`**: Complete architectural blueprint of the Phase 3B EfficientNet-B0 Calibrated Multi-Task model (5.35M params, 21.8 MB).
- **`selection_rationale_and_edge_deployment.md`**: Comprehensive defense of why Phase 3B was selected over pure classification baselines and uncalibrated multi-task models.
- **`model_weights_and_reproducibility.json`**: Checkpoint metadata, parameter counts, test metrics, and inference latency benchmarks.

### `05_IEEE_PAPER_TEXT_AND_SECTIONS/`
- **`00_TITLE_ABSTRACT_KEYWORDS.md`**: Publication-ready title, abstract, and IEEE keywords.
- **`01A_DETAILED_IEEE_LITERATURE_SURVEY_2024_2026.md`**: Complete 20-paper IEEE literature survey with verified DOIs and gap synthesis.
- **`01_INTRODUCTION_AND_RELATED_WORK.md`**: Section I: Introduction & Related Work.
- **`02_PROPOSED_METHODOLOGY_AND_LOSS_FORMULATION.md`**: Section II: Proposed Methodology & Loss Formulation.
- **`03_EXPERIMENTAL_SETUP_AND_DATA_GOVERNANCE.md`**: Section III: Experimental Setup & Data Governance.
- **`04_RESULTS_AND_STATISTICAL_DISCUSSION.md`**: Section IV: Results & Statistical Discussion.
- **`05_EXPLAINABILITY_AND_FAITHFULNESS_ANALYSIS.md`**: Section V: Explainability & Faithfulness Analysis.
- **`06_CONCLUSION_AND_FUTURE_WORK.md`**: Section VI: Conclusion & Future Work.
- **`references.bib`**: BibTeX bibliography file with all 20 IEEE papers (2024–2026) formatted for IEEE citation style.

### `06_RAW_METRICS_AND_STATISTICS_JSON_CSV/`
- **`all_metrics_consolidated.json`**: Programmatic JSON feed of all quantitative results.

---

## 🚀 How to Import into LaTeX / Overleaf
1. Upload the `high_resolution_300dpi/` folder to your Overleaf project under `figures/`.
2. Copy the contents of `02_IEEE_TABLES_LATEX_AND_MD/all_tables_compiled.tex` into your LaTeX manuscript.
3. Append `05_IEEE_PAPER_TEXT_AND_SECTIONS/references.bib` to your bibliography.
