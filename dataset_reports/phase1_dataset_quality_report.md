# Phase 1: Comprehensive Dataset Quality & Bounding-Box Audit Report

## 1. Dataset Integrity & Resolution Summary

| Dataset | Total Images | Valid Images | Corrupt Images | Resolution (Median) | File Formats |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RiceLeafDiseaseBD** | 9,769 | 9,769 | 0 | 1024x1024 | JPG |
| **Sethy5932** | 5,932 | 5,932 | 0 | 300x300 | JPG |
| **RiceLeafDiseaseBD5** | 3,150 | 3,150 | 0 | 300x300 | JPG |
| **RiceSeg5932** | 5,932 | 5,932 | 0 | 256x256 | JPG (Masks) |

## 2. Bounding Box & Healthy No-Box Audit

| Class | Images | Images With Boxes | Images Without Boxes | Total Boxes | Mean Boxes/Image |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Blast** | 1,326 | 1,326 | 0 | 4,844 | 3.65 |
| **Brown Spot** | 2,178 | 2,178 | 0 | 9,352 | 4.29 |
| **Healthy** | 1,575 | 0 | 1,575 | 0 | 0.00 |
| **Leaf Smut** | 724 | 724 | 0 | 1,501 | 2.07 |
| **Sheath Blight** | 1,722 | 1,722 | 0 | 2,607 | 1.51 |
| **Tungro** | 2,244 | 2,244 | 0 | 3,156 | 1.41 |

**Healthy No-Box Hypothesis**: **CONFIRMED**  
*Explanation*: All 1,575 Healthy images strictly have 0 bounding boxes because healthy leaves have no disease lesions. All 8,194 diseased images contain valid YOLO bounding boxes (totaling 21,460 lesion boxes).
