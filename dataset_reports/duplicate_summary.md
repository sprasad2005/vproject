# RiceGuard Duplicate Analysis & Data Leakage Summary

## 1. Exact Duplicate (SHA-256) Findings

* **Total Exact Duplicate Groups Within Same Dataset**: 1137
* **Cross-Dataset Exact Matches (Leakage Risk)**: **0**

## 2. Perceptual Near-Duplicate Findings

* **Total Near-Duplicate Pairs Identified**: 8,002
* **Action Policy**: Read-only flag; no images deleted automatically. Partitions group exact hashes to prevent leakage.
