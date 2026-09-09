"""
Renders clean, publication-ready 300 DPI IEEE figures with:
- 3-panel normalized confusion matrix comparison (Phase 2B, Phase 3, Phase 3B)
- Fixed XAI grounding summary with zero text-legend overlaps and proper y-limits
- Fixed Pareto trade-off and Per-Class figures with clean padding and no number clipping
- Copies all updated figures to both package and paper directories.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE_DIR = r"d:\vproj"
CLASSES = ["Healthy", "Blast", "Brown Spot", "Leaf Smut", "Tungro", "Sheath Blight"]

def load_cm(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["confusion_matrix"])

def render_all():
    figures_out_dirs = [
        os.path.join(BASE_DIR, "results", "figures", "phase5"),
        os.path.join(BASE_DIR, "ieee_paper_publication_package", "01_FIGURES_AND_CHARTS", "high_resolution_300dpi"),
        os.path.join(BASE_DIR, "RiceGuard_IEEE_Paper", "figures"),
    ]
    for d in figures_out_dirs:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------
    # 1. FIGURE 4: 3-PANEL NORMALIZED CONFUSION MATRIX COMPARISON
    # -------------------------------------------------------------
    cm_2b = load_cm(os.path.join(BASE_DIR, "experiments/phase2b_final_baseline/efficientnet_b0/evaluation/efficientnet_b0_evaluation_summary.json"))
    cm_3 = load_cm(os.path.join(BASE_DIR, "experiments/phase3_lesion_aware/efficientnet_b0_multitask/evaluation/efficientnet_b0_multitask_evaluation_summary.json"))
    cm_3b = load_cm(os.path.join(BASE_DIR, "experiments/phase3b_localization_refinement/efficientnet_b0_multitask_refined/evaluation/efficientnet_b0_multitask_refined_evaluation_summary.json"))

    cm_2b_norm = cm_2b.astype("float") / cm_2b.sum(axis=1)[:, np.newaxis]
    cm_3_norm = cm_3.astype("float") / cm_3.sum(axis=1)[:, np.newaxis]
    cm_3b_norm = cm_3b.astype("float") / cm_3b.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)

    panels = [
        (cm_2b_norm, "Phase 2B Baseline\n(Acc = 90.12%, Macro F1 = 0.8891)", "Blues", axes[0]),
        (cm_3_norm, "Phase 3 Multi-Task\n(Acc = 88.75%, Macro F1 = 0.8729)", "Oranges", axes[1]),
        (cm_3b_norm, "Phase 3B Refined Multi-Task\n(Acc = 88.96%, Macro F1 = 0.8751)", "Greens", axes[2]),
    ]

    for cm_norm, title, cmap_name, ax in panels:
        im = ax.imshow(cm_norm, cmap=cmap_name, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                val = cm_norm[i, j]
                text_color = "white" if val > 0.50 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.suptitle("Normalized Test Confusion Matrix Comparison (N = 1,467 Test Images)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])

    for d in figures_out_dirs:
        p1 = os.path.join(d, "confusion_matrix_comparison.png")
        p2 = os.path.join(d, "Fig4_Confusion_Matrix_Comparison.png")
        plt.savefig(p1, dpi=300, bbox_inches="tight")
        if "01_FIGURES_AND_CHARTS" in d:
            plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 4 (3-Panel Confusion Matrix) rendered successfully.")

    # -------------------------------------------------------------
    # 2. FIGURE 5: XAI QUANTITATIVE GROUNDING & SHORTCUT SUMMARY
    # -------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.2), dpi=300)
    models = ["Phase 2B\nBaseline", "Phase 3B\nMulti-Task"]
    colors = ["#3b82f6", "#10b981"]

    # Subplot A: Energy Inside Mask
    energy_in = [9.64, 6.81]
    bars1 = ax1.bar(models, energy_in, color=colors, edgecolor="black", width=0.42, linewidth=1.2)
    ax1.set_ylabel("Energy Inside Mask (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Cross-Dataset Lesion Energy (RiceSeg N=4,348)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 13.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(energy_in):
        ax1.text(i, v + 0.4, f"{v:.2f}%", ha="center", fontsize=9.5, fontweight="bold")

    # Subplot B: Pointing Game Accuracy (Generous Y-Limit to avoid legend overlap!)
    pt_all = [29.58, 21.37]
    pt_corr = [36.79, 38.21]
    x = np.arange(len(models))
    w = 0.32
    ax2.bar(x - w/2, pt_all, w, label="All Samples (N=4,348)", color="#3b82f6", edgecolor="black", linewidth=1.2)
    ax2.bar(x + w/2, pt_corr, w, label="Mutually Correct (N=280)", color="#10b981", edgecolor="black", linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9.5)
    ax2.set_ylabel("Pointing Game Hit Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Peak Attribution Pointing Accuracy", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 52.0)  # Generous headroom for legend and text
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.text(0 - w/2, pt_all[0] + 1.0, f"{pt_all[0]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.text(0 + w/2, pt_corr[0] + 1.0, f"{pt_corr[0]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.text(1 - w/2, pt_all[1] + 1.0, f"{pt_all[1]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.text(1 + w/2, pt_corr[1] + 1.0, f"{pt_corr[1]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # Subplot C: Border Attention Ratio (Healthy)
    border = [18.5, 9.2]
    ax3.bar(models, border, color=["#ef4444", "#10b981"], edgecolor="black", width=0.42, linewidth=1.2)
    ax3.set_ylabel("Border Attention Ratio (%)", fontsize=10, fontweight="bold")
    ax3.set_title("C. Background Shortcut Suppression (Healthy)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 24.0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(border):
        ax3.text(i, v + 0.7, f"{v:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    # Subplot D: Attribution Entropy
    entropy = [0.824, 0.612]
    ax4.bar(models, entropy, color=["#ef4444", "#10b981"], edgecolor="black", width=0.42, linewidth=1.2)
    ax4.set_ylabel("Normalized Spatial Entropy", fontsize=10, fontweight="bold")
    ax4.set_title("D. Saliency Focus & Concentration", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 1.05)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(entropy):
        ax4.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")

    plt.suptitle("XAI Quantitative Grounding & Shortcut Diagnostic Summary", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    for d in figures_out_dirs:
        p1 = os.path.join(d, "xai_grounding_summary.png")
        p2 = os.path.join(d, "Fig5_XAI_Grounding_Summary.png")
        plt.savefig(p1, dpi=300, bbox_inches="tight")
        if "01_FIGURES_AND_CHARTS" in d:
            plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 5 (XAI Grounding Summary) rendered successfully.")

    # -------------------------------------------------------------
    # 3. FIGURE 2: MULTI-DIMENSIONAL TRADE-OFF
    # -------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.2), dpi=300)
    m_names = ["Phase 2B\nBaseline", "Phase 3\nMulti-Task", "Phase 3B\nRefined"]
    m_colors = ["#3b82f6", "#f59e0b", "#10b981"]

    # 1. Classification Macro F1
    f1_scores = [0.8891, 0.8729, 0.8751]
    ax1.bar(m_names, f1_scores, color=m_colors, edgecolor="black", width=0.45, linewidth=1.2)
    ax1.set_ylim(0.84, 0.915)
    ax1.set_ylabel("Macro F1 Score", fontsize=10, fontweight="bold")
    ax1.set_title("A. Classification Performance", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(f1_scores):
        ax1.text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    # 2. Localization F1 & IoU
    loc_f1 = [0.0, 0.0358, 0.0905]
    loc_iou = [0.0, 0.6032, 0.6423]
    x_pos = np.arange(len(m_names))
    w_bar = 0.32
    ax2.bar(x_pos - w_bar/2, loc_f1, w_bar, label="Loc F1", color="#8b5cf6", edgecolor="black", linewidth=1.2)
    ax2.bar(x_pos + w_bar/2, loc_iou, w_bar, label="Mean IoU", color="#059669", edgecolor="black", linewidth=1.2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(m_names, fontsize=9.5)
    ax2.set_ylabel("Metric Score", fontsize=10, fontweight="bold")
    ax2.set_title("B. Localization Performance (N/A for 2B)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for i in range(len(m_names)):
        if i == 0:
            ax2.text(i - w_bar/2, 0.03, "N/A", ha="center", fontsize=8.5, fontweight="bold")
            ax2.text(i + w_bar/2, 0.03, "N/A", ha="center", fontsize=8.5, fontweight="bold")
        else:
            ax2.text(i - w_bar/2, loc_f1[i] + 0.02, f"{loc_f1[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
            ax2.text(i + w_bar/2, loc_iou[i] + 0.02, f"{loc_iou[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # 3. Parameter Count
    params = [4.02, 6.97, 6.97]
    ax3.bar(m_names, params, color=m_colors, edgecolor="black", width=0.45, linewidth=1.2)
    ax3.set_ylabel("Parameters (Millions)", fontsize=10, fontweight="bold")
    ax3.set_title("C. Model Parameter Footprint", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 8.8)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(params):
        ax3.text(i, v + 0.2, f"{v:.2f}M", ha="center", fontsize=9, fontweight="bold")

    # 4. GPU Latency
    latency = [10.36, 13.52, 10.46]
    ax4.bar(m_names, latency, color=m_colors, edgecolor="black", width=0.45, linewidth=1.2)
    ax4.set_ylabel("GPU Latency (ms / image)", fontsize=10, fontweight="bold")
    ax4.set_title("D. Inference Latency (GTX 1650)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 16.5)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(latency):
        ax4.text(i, v + 0.35, f"{v:.2f} ms", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Multi-Dimensional Performance & Efficiency Trade-Offs", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    for d in figures_out_dirs:
        p1 = os.path.join(d, "classification_localization_tradeoff.png")
        p2 = os.path.join(d, "Fig2_Classification_Localization_Tradeoff.png")
        plt.savefig(p1, dpi=300, bbox_inches="tight")
        if "01_FIGURES_AND_CHARTS" in d:
            plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 2 (Trade-off) rendered successfully.")

    # -------------------------------------------------------------
    # 4. FIGURE 3: PER-CLASS F1 SCORE COMPARISON
    # -------------------------------------------------------------
    f1_2b = [0.940, 0.878, 0.871, 0.777, 0.921, 0.949]
    f1_3b = [0.946, 0.862, 0.837, 0.734, 0.914, 0.945]
    x_cls = np.arange(len(CLASSES))
    w_cls = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.bar(x_cls - w_cls/2, f1_2b, w_cls, label="Phase 2B Baseline (Macro F1 = 0.8891)", color="#3b82f6", edgecolor="black", linewidth=1.2)
    ax.bar(x_cls + w_cls/2, f1_3b, w_cls, label="Phase 3B Refined Multi-Task (Macro F1 = 0.8751)", color="#10b981", edgecolor="black", linewidth=1.2)

    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Per-Class Disease Classification F1 Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x_cls)
    ax.set_xticklabels(CLASSES, fontsize=10, fontweight="bold")
    ax.set_ylim(0.60, 1.06)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(CLASSES)):
        ax.text(x_cls[i] - w_cls/2, f1_2b[i] + 0.008, f"{f1_2b[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(x_cls[i] + w_cls/2, f1_3b[i] + 0.008, f"{f1_3b[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()

    for d in figures_out_dirs:
        p1 = os.path.join(d, "per_class_f1_comparison.png")
        p2 = os.path.join(d, "Fig3_Per_Class_F1_Comparison.png")
        plt.savefig(p1, dpi=300, bbox_inches="tight")
        if "01_FIGURES_AND_CHARTS" in d:
            plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 3 (Per-Class F1) rendered successfully.")

if __name__ == "__main__":
    render_all()
