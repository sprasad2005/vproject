# RiceGuard Data Architecture & Policy

This directory houses the dataset structure for the **RiceGuard** research project.

## Dataset Roles & Policies

### 1. Primary Training Dataset: `RiceLeafDiseaseBD`
* **Path**: `data/primary/RiceLeafDiseaseBD/`
* **Subdirectories**: `raw/`, `images/`, `annotations/`, `metadata/`
* **Role**: Primary dataset used for **training**, **validation**, **internal testing**, and **lesion-aware bounding box supervision**.
* **Target Classes (6)**:
  1. `Healthy`
  2. `Blast`
  3. `Brown Spot`
  4. `Leaf Smut`
  5. `Tungro`
  6. `Sheath Blight`

### 2. External Validation Dataset 1: `Sethy5932`
* **Path**: `data/external/Sethy5932/`
* **Subdirectories**: `raw/`, `images/`, `metadata/`
* **Role**: **Strictly external testing only**. Never used for training or validation. Used for cross-dataset generalization and domain shift evaluation.
* **Classes (4)**:
  1. `Bacterial Blight`
  2. `Blast`
  3. `Brown Spot`
  4. `Tungro`

### 3. External Validation Dataset 2: `RiceLeafDiseaseBD5`
* **Path**: `data/external/RiceLeafDiseaseBD5/`
* **Subdirectories**: `raw/`, `images/`, `metadata/`
* **Role**: **Strictly independent field-domain testing**. Never used for training or validation. Evaluates robustness under natural field variations.
* **Classes (5)**:
  1. `Blast`
  2. `Narrow Brown Spot` *(Note: Maintained as distinct class; not collapsed into Brown Spot)*
  3. `Sheath Blight`
  4. `Tungro`
  5. `Normal`

### 4. XAI Ground Truth Dataset: `RiceSeg5932`
* **Path**: `data/xai_ground_truth/RiceSeg5932/`
* **Subdirectories**: `raw/`, `images/`, `masks/`, `metadata/`
* **Role**: **XAI evaluation only**. Never used for classifier training. Contains pixel-level segmentation masks for quantitative lesion-grounded explanation evaluation (IoU, Dice, Precision, Recall).

### 5. Processed Data: `processed/`
* **Path**: `data/processed/`
* **Role**: Cached manifests, verified image paths, and precomputed metadata indices.
