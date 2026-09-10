# RiceGuard: FTR-1 Documentation Package

## Purpose
This directory contains the complete, academically structured First Term Review (FTR-1) documentation package for the **RiceGuard** project: *Multi-Task Deep Learning for Rice Leaf Disease Classification and Calibrated Lesion Localization with Explainable AI Grounding*.

All documentation herein is derived strictly from verified repository evidence across experimental phases (Phase 1 through Phase 6), codebases (`src/`, `app/`, `configs/`, `scripts/`, `tests/`), quantitative evaluation reports, statistical audits, and scientific data governance protocols.

---

## Document Structure

| Document File | Title / Topic | Primary Focus | Key Visual / Diagram |
| :--- | :--- | :--- | :--- |
| [`01_problem_statement_title.md`](file:///d:/vproj/FTR1/01_problem_statement_title.md) | Problem Statement & Title | Formal title, background, problem formulation, motivation, research questions, and tiered objectives | Research Alignment Flowchart |
| [`02_abstract.md`](file:///d:/vproj/FTR1/02_abstract.md) | Structured Academic Abstract | 250–350 word executive technical summary with verified metrics and keywords | Metric Summary / Context Schema |
| [`03_introduction.md`](file:///d:/vproj/FTR1/03_introduction.md) | Introduction & Scope | Agricultural context, limitations of classification-only models, spatial localization necessity, explainability, nuanced research gaps, and formal contributions | Multi-Dimensional Gap Map |
| [`04_literature_survey.md`](file:///d:/vproj/FTR1/04_literature_survey.md) | Literature Survey | Critical taxonomy of 20 official IEEE references (`[1]`–`[20]`), detailed comparative landscape matrix, and methodological positioning | Literature Categorization Taxonomy |
| [`05_methodology_algorithms_techniques.md`](file:///d:/vproj/FTR1/05_methodology_algorithms_techniques.md) | Methodology & Algorithms | End-to-end data pipeline, multi-task network bifurcation, Phase 3B positive-weight loss formulation, validation-only threshold and Top-$K$ calibration, quantitative XAI grounding, and pseudocode | Complete Multi-Task Pipeline & Architecture |
| [`06_design_uml_diagrams.md`](file:///d:/vproj/FTR1/06_design_uml_diagrams.md) | Software & System UML Design | Formal UML artifacts: Layered System Architecture, Sequence Interaction, Component Decomposition, Domain Class Structure, and Pipeline Activity Flows | 5 Syntactically Valid UML Diagrams |
| [`07_modules_splitup.md`](file:///d:/vproj/FTR1/07_modules_splitup.md) | Functional Module Split-Up | Comprehensive specification of the 10 functional modules across ingestion, inference, calibration, XAI, evaluation, API, and UI | Inter-Module Dependency Network |
| [`08_proposed_system.md`](file:///d:/vproj/FTR1/08_proposed_system.md) | Proposed System Architecture | Comprehensive delineation between Research System and Application System, input/output contracts, key features, advantages, and scientific limitations | Dual-Track System Architecture |
| [`09_software_tools_technologies.md`](file:///d:/vproj/FTR1/09_software_tools_technologies.md) | Software Stack & Environment | Verified execution stack (PyTorch 2.6.0, CUDA 12.4, FastAPI, React 19, Vite 6), dependencies, reproducibility tooling, and hardware environment | Technology Integration Hierarchy |
| [`10_implementation_25_percent.md`](file:///d:/vproj/FTR1/10_implementation_25_percent.md) | Implementation Status (FTR-1 Milestone) | FTR-1 milestone review vs actual repository completion status across Phase 1–6, codebase evidence audit, and verification artifacts | Phase Progression & Milestone Gantt/Flow |
| [`11_proposed_outcomes.md`](file:///d:/vproj/FTR1/11_proposed_outcomes.md) | Proposed & Validated Outcomes | Strict demarcation of academic outcomes, technical deliverables, validated experimental findings, and future extensions | Outcome Relationship & Evidence Tree |
| [`12_project_plan_2_0.md`](file:///d:/vproj/FTR1/12_project_plan_2_0.md) | Project Plan 2.0 | Comprehensive sequence-based lifecycle timeline, milestone status, dependency mapping, risk matrix, and resource allocation | Relative Project Progression Gantt Chart |
| [`13_srs.md`](file:///d:/vproj/FTR1/13_srs.md) | Software Requirements Specification | IEEE-830 aligned SRS containing functional requirements (`FR-01`–`FR-14`), non-functional parameters, API schemas, and use-case analysis | SRS Use-Case & Interaction Diagram |

---

## Mermaid Diagram Usage
All diagrams throughout this documentation suite are natively encoded as standard GitHub Flavored Markdown `mermaid` code blocks. They render directly in any Mermaid-compliant Markdown viewer, IDE previewer, or documentation renderer without requiring external image assets.

---

## Strict Scientific Truth & Research Governance
In compliance with rigorous academic and scientific standards:
1. **Zero Metric Fabrication**: Every numerical metric (classification Macro F1, localization precision/recall/F1, mean IoU, false-positive reduction, pointing game accuracy, energy-inside-mask, insertion/deletion AUC, inference latency, parameter counts) is directly traced to verified CSV/JSON/log artifacts.
2. **RiceSeg5932 Dataset Governance**: RiceSeg5932 ($N = 4,348$ segmentation masks across 4 disease classes) was strictly reserved for **zero-shot post-hoc XAI ground-truth attribution evaluation**. It was **NEVER** exposed to model training, validation loss, checkpoint selection, or hyperparameter/threshold tuning.
3. **Deployment Status Precision**: Localhost browser-to-backend inference via FastAPI and React 19 is fully implemented and operational. Production deployment infrastructure (Vercel static hosting configuration, containerized Dockerfile, production environment variables) is formally categorized as **Deployment Preparation Completed**, preserving truthful scientific governance.
4. **Scope Integrity**: No source code, checkpoints, datasets, or experimental configurations outside `FTR1/` were altered during the compilation of this review package.
