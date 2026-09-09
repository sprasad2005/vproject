"""
Renders a publication-quality failure cases comparison figure with real leaf images,
clean non-overlapping section headers, and precise diagnostic labels.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = r"d:\vproj"
CSV_PATH = os.path.join(BASE_DIR, "results", "reports", "phase5_failure_case_summary.csv")

def build_image_index():
    img_map = {}
    search_roots = [
        os.path.join(BASE_DIR, "RiceLeafDiseaseBD A Field-Based Annotated Smartpho"),
        os.path.join(BASE_DIR, "data"),
    ]
    for s_root in search_roots:
        if os.path.exists(s_root):
            for root, dirs, files in os.walk(s_root):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_map[f.lower()] = os.path.join(root, f)
    return img_map

def render_figure():
    df = pd.read_csv(CSV_PATH)
    img_map = build_image_index()
    
    categories = [
        ("baseline_wins", "Category A: Baseline Wins (Phase 2B Correct, Phase 3B Incorrect)", "#1d4ed8"),
        ("multitask_wins", "Category B: Multi-Task Wins (Phase 3B Correct, Phase 2B Incorrect)", "#059669"),
        ("mutual_errors", "Category C: Mutual Diagnostic Errors (Both Models Incorrect)", "#d97706"),
        ("healthy_fp", "Category D: Healthy False Positives (Phase 3B Over-Detection on Healthy)", "#dc2626"),
        ("lesion_miss", "Category E: Lesion Misses (Phase 3B Missed Small Ground-Truth Lesions)", "#7c3aed"),
    ]
    
    # 5 rows x 3 cols with extra vertical space for titles
    fig, axes = plt.subplots(5, 3, figsize=(13.5, 23), dpi=300)
    fig.suptitle("Comparative Diagnostic Failure Mode Analysis\n(Phase 2B Baseline vs. Phase 3B Multi-Task)", fontsize=15, fontweight='bold', y=0.995)
    
    for row_idx, (cat_key, cat_title, border_color) in enumerate(categories):
        if cat_key in ["healthy_fp", "lesion_miss"]:
            sub_df = df[df[f"is_{cat_key}"]]
        else:
            sub_df = df[df["category"] == cat_key]
            
        if len(sub_df) >= 3:
            indices = np.linspace(0, len(sub_df) - 1, 3, dtype=int)
            samples = sub_df.iloc[indices]
        else:
            samples = sub_df
            
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                row = samples.iloc[col_idx]
                rel_path = str(row["image_path"])
                bname = os.path.basename(rel_path).lower()
                
                img_path = img_map.get(bname)
                if not img_path and os.path.exists(os.path.join(BASE_DIR, rel_path)):
                    img_path = os.path.join(BASE_DIR, rel_path)
                
                if img_path and os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.imshow(np.full((224, 224, 3), 235, dtype=np.uint8))
                
                gt_cls = row['canonical_class']
                p2b_cls = row['phase2b_pred_class']
                p2b_conf = row['phase2b_conf']
                p3b_cls = row['phase3b_pred_class']
                p3b_conf = row['phase3b_conf']
                
                # Single clean title per sample
                title_str = f"GT: {gt_cls} | P2B: {p2b_cls} ({p2b_conf:.2f}) | P3B: {p3b_cls} ({p3b_conf:.2f})"
                ax.set_title(title_str, fontsize=8.5, pad=6, fontweight='medium')
                ax.set_xticks([])
                ax.set_yticks([])
                
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.0)
            else:
                ax.axis('off')
                
        # Category Banner on the Center Column with a clean box
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=border_color, lw=1.5)
        axes[row_idx, 1].text(0.5, 1.18, cat_title, transform=axes[row_idx, 1].transAxes,
                              ha='center', va='bottom', fontsize=10.5, fontweight='bold',
                              color=border_color, bbox=bbox_props)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.985])
    plt.subplots_adjust(hspace=0.45, wspace=0.15)
    
    out_paths = [
        os.path.join(BASE_DIR, "results", "figures", "phase5", "failure_cases_baseline_vs_multitask.png"),
        os.path.join(BASE_DIR, "ieee_paper_publication_package", "01_FIGURES_AND_CHARTS", "high_resolution_300dpi", "Fig6_Failure_Cases_Comparison.png"),
        os.path.join(BASE_DIR, "RiceGuard_IEEE_Paper", "figures", "failure_cases_baseline_vs_multitask.png"),
    ]
    
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {p}")
        
    plt.close()

if __name__ == "__main__":
    render_figure()
