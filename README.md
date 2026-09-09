# RiceGuard: Lesion-Grounded Explainable and Uncertainty-Aware Deep Learning Framework for Rice Leaf Disease Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Project Status: Phase 0](https://img.shields.io/badge/status-Phase%200%20%7C%20Foundation%20%26%20Dataset%20Registration-orange.svg)](#current-project-status)

---

## 1. Project Title & Overview

**Full Title**: *Lesion-Grounded Explainable and Uncertainty-Aware Deep Learning Framework for Rice Leaf Disease Classification*  
**Codename**: **RiceGuard**

RiceGuard is a research-grade framework engineered to move beyond traditional accuracy-only agricultural classifiers. It couples deep convolutional networks and Vision Transformers with:
* **Lesion-level supervision** using bounding boxes
* **Post-hoc confidence calibration** via Temperature Scaling
* **Epistemic uncertainty quantification** and Out-of-Distribution (OOD) energy detection
* **Quantitative XAI grounding** against expert segmentation masks (IoU, Dice, Precision, Recall)
* **Explanation stability** under realistic image corruptions
* **ACCEPT / ABSTAIN selective prediction** for safe field deployment
* **Strict external cross-domain validation** and distribution shift analysis

---

## 2. Dataset Roles & Access Governance

All existing datasets have been registered and protected by security access policies in `src/data/dataset_registry.py`:

| Dataset | Role | Training Allowed | Description & Permitted Scope |
| :--- | :--- | :---: | :--- |
| **RiceLeafDiseaseBD** | Primary | **Yes** | Primary benchmark for backbone training, validation, internal testing, and bounding-box lesion supervision (6 classes: *Healthy*, *Blast*, *Brown Spot*, *Leaf Smut*, *Tungro*, *Sheath Blight*). |
| **Sethy5932** | External Test | **No** | External benchmark for cross-dataset generalization and domain shift evaluation (4 classes: *Bacterial Blight*, *Blast*, *Brown Spot*, *Tungro*). Never trained or tuned on. |
| **RiceLeafDiseaseBD5** | External Field Test | **No** | Independent field-collected benchmark (5 classes: *Blast*, *Narrow Brown Spot*, *Sheath Blight*, *Tungro*, *Normal*). *Narrow Brown Spot is preserved as a distinct class.* |
| **RiceSeg5932** | XAI Ground Truth | **No** | Expert pixel-level segmentation masks strictly used for quantitative post-hoc explanation validation (IoU, Dice, Precision, Recall). Never used as classification labels or training targets. |

---

## 3. Repository Structure

```text
RiceGuard/
├── README.md                           # Master project documentation
├── requirements.txt                    # Pinned core dependencies
├── .gitignore                          # ML research git exclusion rules
├── pyproject.toml                      # Modern Python project & tool configurations
│
├── configs/                            # Modular YAML configuration system
│   ├── base.yaml                       # Global seed (42), device preferences, logging
│   ├── paths.yaml                      # Relative path references to registered datasets
│   ├── data.yaml                       # Registered datasets, classes, split ratios
│   ├── model.yaml                      # CNN, ResNet50, EfficientNet-B4, ViT, LesionAwareNet
│   ├── training.yaml                   # Optimizer, scheduler, loss hyperparameters
│   ├── calibration.yaml                # Temperature scaling & ECE metrics
│   ├── uncertainty.yaml                # MC Dropout, entropy, selective prediction
│   ├── xai.yaml                        # Grad-CAM++, Integrated Gradients, SHAP, Grounding
│   └── robustness.yaml                 # Perturbation matrix & corruption tests
│
├── data/                               # Dataset directories & processing caches
│   ├── README.md                       # Data policy and ingestion guidelines
│   ├── primary/                        # Placeholder references
│   ├── external/                       # External dataset references
│   ├── xai_ground_truth/               # Segmentation mask references
│   └── processed/                      # Preprocessed manifests & metadata cache
│
├── dataset_reports/                    # Generated dataset inventory & audit reports
│   ├── README.md
│   ├── dataset_inventory.json          # Machine-readable inventory metadata
│   └── dataset_inventory.md            # Markdown summary table of discovered datasets
│
├── splits/                             # Deterministic, leak-free partition manifests
│   └── README.md
│
├── src/                                # Core modular Python package
│   ├── __init__.py
│   ├── data/                           # Registry, Inspector, loaders, label mappings
│   │   ├── dataset_registry.py         # Dataset security rules & access governance
│   │   ├── dataset_inspector.py        # Read-only discovery and audit engine
│   │   ├── dataset.py                  # Dataset interface
│   │   ├── preprocessing.py            # Normalization transforms
│   │   ├── augmentation.py             # Augmentation pipelines
│   │   ├── annotations.py              # BBox and mask parsers
│   │   └── label_mapping.py            # Unified label definitions
│   ├── models/                         # Model backbones & architectures
│   ├── training/                       # Loss functions, optimizers, schedulers, loop
│   ├── evaluation/                     # Macro-F1, confusion matrix, domain shift
│   ├── calibration/                    # Temperature scaling, reliability diagrams
│   ├── uncertainty/                    # Entropy, MC Dropout, selective prediction
│   ├── xai/                            # Saliency maps, lesion grounding, stability
│   ├── robustness/                     # Corruption matrix & perturbation benchmarks
│   └── utils/                          # Device, seed, config, paths, logger, tracker
│
├── scripts/                            # Verification & execution CLI tools
│   ├── inspect_datasets.py             # Automatic dataset discovery & inventory auditor
│   ├── check_environment.py            # Software & hardware dependency validator
│   ├── check_gpu.py                    # CUDA & GPU diagnostic tool
│   └── verify_project.py               # Complete project integrity validator
│
├── checkpoints/                        # Saved model weights & state dicts
├── logs/                               # Formatted experiment logs
├── experiments/                        # Reproducible experiment runs & artifacts
├── results/                            # Exported figures, metrics, and reports
├── tests/                              # Pytest automated test suite
└── app/                                # Interactive demonstration interface
```

---

## 4. Installation & Verification

### Virtual Environment Setup
```powershell
# Navigate to project root
cd d:\vproj

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### Execution & Verification Commands
```powershell
# 1. Run Automated Dataset Discovery & Inventory Audit
python scripts/inspect_datasets.py

# 2. Verify installed dependencies and CUDA availability
python scripts/check_environment.py

# 3. Check GPU diagnostics
python scripts/check_gpu.py

# 4. Verify complete project setup, security policies, and configs
python scripts/verify_project.py

# 5. Run automated test suite
pytest

# 6. Check code quality
ruff check .
```

---

## 5. Current Project Status

> [!IMPORTANT]
> **Status**: **Phase 0 — Research Foundation and Dataset Registration Complete**
> 
> * **Completed**: Project scaffolding, modular architecture, configuration system, reproducible seed control, GPU device auto-detection, dataset discovery, read-only dataset inspection, dataset inventory generation, dataset access governance, and test suite.
> * **Next Phase (Phase 1)**: Dataset manifest generation, image loading, and bounding-box annotation parsing. Model training, preprocessing, splitting, and augmentation are strictly deferred to subsequent phases.
