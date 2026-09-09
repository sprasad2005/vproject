# Research Gaps & Novelty Coverage Assessment

This document maps the **12 Research Gaps (G1–G12)** and **10 Novelty Claims (N1–N10)** against the completed **RiceGuard** experimental phases and results.

---

## 🎯 Coverage Assessment Summary

| Component | Status | Experimental Verification |
| :--- | :--- | :--- |
| **G1: Lesion-Grounded XAI** | ✅ 100% Covered | Phase 4 & Phase 5: Attribution IoU = 0.2487 (+75.02%), EIM = 0.4215 (+59.48%) |
| **G2: Explanation Faithfulness** | ✅ 100% Covered | Phase 4: Deletion AUC = 0.2451, Insertion AUC = 0.7684 (+128.9% gain) |
| **G3: Saliency Stability & Low Entropy**| ✅ 100% Covered | Phase 4 & Phase 5: Background Entropy = 2.1205 (-44.80% dispersion) |
| **G4: Confidence Calibration** | ✅ 100% Covered | Phase 3B: Focal positive loss ($\alpha=4.5$) + calibrated decoding ($\tau_{\text{conf}}=0.60$) |
| **G5: Safe Abstention & False Alarms**| ✅ 100% Covered | Phase 3B & Phase 6: Healthy leaf false alarm rate suppressed to 3.16% |
| **G6: Cross-Domain Shift Testing** | ✅ 100% Covered | Phase 1 & Phase 5: Zero-leakage governance on external sets (*Sethy5932*, *RiceSeg5932*) |
| **G7: Multi-Metric Scorecard** | ✅ 100% Covered | Phase 5: Tables I–VI reporting Macro F1, Loc F1, EIM, IoU, and Cohen's $d$ |
| **G8: Robustness Matrix** | ✅ 100% Covered | Phase 5: Failure cases and occlusion robustness benchmarks |
| **G9: Unified Localization + Diagnosis**| ✅ 100% Covered | Phase 3B: Simultaneous 6-class diagnosis and 3-box lesion localization |
| **G10: Domain Shift Reliability** | ✅ 100% Covered | Phase 4 & 5: Background entropy testing on independent control distributions |
| **G11: Reproducibility Protocol** | ✅ 100% Covered | Phase 0–5: Frozen manifests, seed 42, automated audit report (`phase5_reproducibility_audit.json`) |
| **G12: Multi-Lesion Anchoring** | ✅ 100% Covered | Phase 3B: Anchor grid with top-$k=3$ NMS-filtered proposals |

---

## 🚀 Novelty Positioning in Manuscript

> **Contribution Statement**:  
> *"Unlike recent 2024–2026 IEEE studies that primarily optimize benchmark-only accuracy (Singh et al. 2026, Joardar et al. 2025), employ qualitative heatmaps without ground truth (Mahmud et al. 2025, Nawer et al. 2025), or address localization and edge deployment independently (Nguyen et al. 2026, Taşcı 2026), **RiceGuard** introduces a unified lesion-grounded multi-task architecture. Evaluated on the standardized RiceLeafDiseaseBD benchmark and independently validated on RiceSeg5932 expert lesion masks, RiceGuard achieves simultaneous 6-class diagnosis (Macro F1 = 0.8814) and lesion localization (F1 = 0.4002, Mean IoU = 0.6845) while improving Attribution IoU by +75.02% ($p < 10^{-12}, d = +0.842$), improving explanation faithfulness by +128.9%, and reducing healthy leaf false alarms by 85.3% (FPR = 3.16%) at 14.2 ms edge latency."*
