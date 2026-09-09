# Research & Exploration Notebooks

This directory is designated for exploratory data analysis (EDA), prototype visualization, and qualitative inspection notebooks.

## Notebook Guidelines
- All notebooks must import from `src` rather than defining ad-hoc pipelines.
- Seeds must be explicitly set using `src.utils.seed.set_seed`.
- Do not commit large inline binary outputs or model weights to git.
- Standalone runnable scripts in `scripts/` are preferred for production-grade experiments.
