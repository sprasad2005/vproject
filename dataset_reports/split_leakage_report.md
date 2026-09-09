# RiceGuard Primary Dataset Split & Leakage Audit Report

**Generated**: 2026-09-08 15:49:33  
**Random Seed**: `42`  

---

## 1. Split Proportions & Sample Counts

| Partition | Sample Count | Percentage | Annotations Available | Healthy (No Boxes) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 6,837 | 70.0% | 5,736 | 1,101 |
| **Validation** | 1,465 | 15.0% | 1,228 | 237 |
| **Internal Test** | 1,467 | 15.0% | 1,230 | 237 |
| **Total** | **9,769** | **100.0%** | **8,194** | **1,575** |

---

## 2. Stratified Class Distributions

| Canonical Class | Total | Train (70%) | Val (15%) | Test (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Blast** | 1,326 | 928 | 199 | 199 |
| **Brown Spot** | 2,178 | 1,524 | 327 | 327 |
| **Healthy** | 1,575 | 1,101 | 237 | 237 |
| **Leaf Smut** | 724 | 507 | 108 | 109 |
| **Sheath Blight** | 1,722 | 1,206 | 258 | 258 |
| **Tungro** | 2,244 | 1,571 | 336 | 337 |

---

## 3. Duplicate Leakage Verification

* **Train ↔ Validation Hash Overlap**: 0
* **Train ↔ Test Hash Overlap**: 0
* **Validation ↔ Test Hash Overlap**: 0
* **Audit Result**: **PASSED (ZERO LEAKAGE)**
