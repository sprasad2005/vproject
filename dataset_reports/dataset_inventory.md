# RiceGuard Dataset Inventory & Integrity Audit Report

**Project**: RiceGuard  
**Phase**: Phase 0 — Research Foundation, Reproducibility Setup, and Existing Dataset Registration  

---

## 1. Discovered Dataset Summary

| Dataset Key | Dataset Name | Registered Role | Discovered Location | Training Allowed | Total Images / Masks | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `primary` | **RiceLeafDiseaseBD** | `primary_training` | `RiceLeafDiseaseBD A Field-Based Annotated Smartpho` | YES | 9,769 | DISCOVERED |
| `sethy` | **Sethy5932** | `external_test` | `Rice Leaf Disease Image Samples` | **NO** | 5,932 | DISCOVERED |
| `bd5` | **RiceLeafDiseaseBD5** | `external_field_test` | `RiceLeafDisease-BD5 A Field-Collected Five-Class I` | **NO** | 3,150 | DISCOVERED |
| `riceseg` | **RiceSeg5932** | `xai_ground_truth` | `RiceSeg-5932 Complete Pixel-Level Segmentation Mas` | **NO** | 5,932 | DISCOVERED |

---

## 2. Dataset Specific Audits

### A. Primary Benchmark: `RiceLeafDiseaseBD`
* **Discovered Path**: `RiceLeafDiseaseBD A Field-Based Annotated Smartpho`
* **Total Original Images**: 9,769
* **Image Formats**: .jpg
* **Bounding Box Annotations**: Available (8,194 `YOLO (.txt)` files)
* **Class Distribution (Original Images)**:
  * **Blast**: 1,326 images
  * **Brown spot**: 2,178 images
  * **Healthy**: 1,575 images
  * **Leaf smut**: 724 images
  * **Rice Tungro**: 2,244 images
  * **Sheath blight**: 1,722 images

### B. External Benchmark 1: `Sethy5932`
* **Discovered Path**: `Rice Leaf Disease Image Samples`
* **Total Images**: 5,932
* **Image Formats**: .jpg
* **Training Permitted**: `False` *(Strictly External Test Only)*
* **Class Distribution**:
  * **Bacterialblight**: 1,584 images
  * **Blast**: 1,440 images
  * **Brownspot**: 1,600 images
  * **Tungro**: 1,308 images

### C. External Field Benchmark 2: `RiceLeafDiseaseBD5`
* **Discovered Path**: `RiceLeafDisease-BD5 A Field-Collected Five-Class I`
* **Total Field Images**: 3,150
* **Image Formats**: .jpg
* **Training Permitted**: `False` *(Strictly Field Test Only)*
* **Class Distribution**:
  * **Blast**: 720 images
  * **Narrow_Brown_Spot**: 476 images
  * **Normal_Leaf**: 834 images
  * **Sheath_Blight**: 626 images
  * **Tungro**: 494 images

### D. XAI Ground Truth: `RiceSeg5932`
* **Discovered Path**: `RiceSeg-5932 Complete Pixel-Level Segmentation Mas`
* **Total Pixel-Level Masks**: 5,932
* **Mask Formats**: .jpg
* **Mask Class Breakdown**:
  * **Bacterialblight**: 1,584 masks
  * **Blast**: 1,440 masks
  * **Brownspot**: 1,600 masks
  * **Tungro**: 1,308 masks
* **Image-Mask Pairing Verification**:
  * Candidate External Images: 5,932
  * Matched Image-Mask Pairs: **5,932**
  * Unmatched Images: 0
  * Unmatched Masks: 0
  * Pairing Feasible: **True**

---

## 3. Security & Access Policy Enforcement

| Dataset | Permitted Uses | Forbidden Uses |
| :--- | :--- | :--- |
| `RiceLeafDiseaseBD` | Training, Validation, Internal Test, Bounding Box Supervision | Cross-contamination |
| `Sethy5932` | Final External Generalization Test, Domain Shift Analysis | **Training, Validation, Threshold Tuning** |
| `RiceLeafDiseaseBD5` | Final External Field Robustness Test, Domain Shift Analysis | **Training, Validation, Class Merging** |
| `RiceSeg5932` | Post-hoc XAI Explanation Mask IoU/Dice Validation | **Classifier Training, Feature Extraction** |
