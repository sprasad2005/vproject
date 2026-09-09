"""Faithfulness evaluation via Deletion and Insertion perturbation analysis.

Evaluates how accurately Grad-CAM heatmaps reflect model decision-making
by measuring prediction confidence degradation under progressive feature removal (Deletion)
and confidence recovery under progressive feature restoration (Insertion).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def integrate_auc(curve: List[float], step_fractions: List[float]) -> float:
    """Compute Area Under the Curve using trapezoidal integration."""
    x = np.asarray(step_fractions, dtype=np.float64)
    y = np.asarray(curve, dtype=np.float64)
    # Manual trapezoid integration to be robust across all numpy versions
    auc = 0.5 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))
    return float(auc)


def create_blurred_baseline(image_tensor: torch.Tensor) -> torch.Tensor:
    """Create a heavily blurred baseline for insertion/deletion analysis.

    Args:
        image_tensor: Normalized image tensor of shape [3, H, W] or [1, 3, H, W].

    Returns:
        Blurred tensor of same shape and dtype.
    """
    is_batched = image_tensor.ndim == 4
    if not is_batched:
        img = image_tensor.unsqueeze(0)
    else:
        img = image_tensor

    # Apply Gaussian blur
    blurred = TF.gaussian_blur(img, kernel_size=[21, 21], sigma=[5.0, 5.0])
    return blurred if is_batched else blurred.squeeze(0)


def evaluate_sample_faithfulness(
    model: nn.Module,
    image_tensor: torch.Tensor,
    attribution_map: np.ndarray,
    target_class: int,
    num_steps: int = 20,
    baseline_type: str = "blur",  # "blur" or "zeros"
) -> Dict[str, Any]:
    """Perform Deletion and Insertion perturbation test on a single sample.

    Args:
        model: Evaluated PyTorch model (in eval mode).
        image_tensor: Image tensor of shape [3, H, W] or [1, 3, H, W].
        attribution_map: 2D numpy array [H, W] of normalized saliency scores in [0, 1].
        target_class: Target class index to monitor confidence for.
        num_steps: Number of perturbation intervals (default: 20).
        baseline_type: Type of baseline ("blur" or "zeros").

    Returns:
        Dictionary containing deletion_curve, insertion_curve, deletion_auc, insertion_auc.
    """
    model.eval()
    device = next(model.parameters()).device

    if image_tensor.ndim == 3:
        img = image_tensor.unsqueeze(0).to(device)
    else:
        img = image_tensor.to(device)

    # Prepare baseline
    if baseline_type == "blur":
        baseline = create_blurred_baseline(img)
    else:
        baseline = torch.zeros_like(img)

    # Flatten pixels and rank by attribution score (descending)
    attr_flat = attribution_map.flatten()
    sorted_indices = np.argsort(-attr_flat)  # descending order of importance
    total_pixels = len(attr_flat)

    # Define step thresholds (0% to 100%)
    fractions = [i / float(num_steps) for i in range(num_steps + 1)]
    deletion_probs: List[float] = []
    insertion_probs: List[float] = []

    # Flatten spatial dimensions of image and baseline for efficient masking
    b, c, h, w = img.shape
    img_flat = img.view(b, c, -1)  # [1, 3, H*W]
    baseline_flat = baseline.view(b, c, -1)  # [1, 3, H*W]

    with torch.no_grad():
        for frac in fractions:
            num_perturbed = int(round(frac * total_pixels))
            perturbed_indices = sorted_indices[:num_perturbed]

            # --- DELETION: Replace top pixels with baseline ---
            del_tensor = img_flat.clone()
            if num_perturbed > 0:
                del_tensor[:, :, perturbed_indices] = baseline_flat[:, :, perturbed_indices]
            del_img = del_tensor.view(b, c, h, w)

            del_out = model(del_img)
            del_logits = del_out[0] if isinstance(del_out, tuple) else del_out
            del_prob = float(F.softmax(del_logits, dim=1)[0, target_class].item())
            deletion_probs.append(del_prob)

            # --- INSERTION: Start from baseline, restore top pixels from original ---
            ins_tensor = baseline_flat.clone()
            if num_perturbed > 0:
                ins_tensor[:, :, perturbed_indices] = img_flat[:, :, perturbed_indices]
            ins_img = ins_tensor.view(b, c, h, w)

            ins_out = model(ins_img)
            ins_logits = ins_out[0] if isinstance(ins_out, tuple) else ins_out
            ins_prob = float(F.softmax(ins_logits, dim=1)[0, target_class].item())
            insertion_probs.append(ins_prob)

    del_auc = integrate_auc(deletion_probs, fractions)
    ins_auc = integrate_auc(insertion_probs, fractions)

    return {
        "step_fractions": fractions,
        "deletion_curve": deletion_probs,
        "insertion_curve": insertion_probs,
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
    }


def evaluate_faithfulness_batch(
    model: nn.Module,
    samples: List[Tuple[torch.Tensor, np.ndarray, int]],
    num_steps: int = 20,
) -> Dict[str, Any]:
    """Evaluate faithfulness over a collection of samples and aggregate curves and AUCs."""
    del_aucs: List[float] = []
    ins_aucs: List[float] = []
    del_curves: List[List[float]] = []
    ins_curves: List[List[float]] = []

    for img_tensor, attr_map, target_cls in samples:
        res = evaluate_sample_faithfulness(
            model=model,
            image_tensor=img_tensor,
            attribution_map=attr_map,
            target_class=target_cls,
            num_steps=num_steps,
        )
        del_aucs.append(res["deletion_auc"])
        ins_aucs.append(res["insertion_auc"])
        del_curves.append(res["deletion_curve"])
        ins_curves.append(res["insertion_curve"])

    fractions = [i / float(num_steps) for i in range(num_steps + 1)]
    mean_del_curve = np.mean(del_curves, axis=0).tolist() if del_curves else []
    mean_ins_curve = np.mean(ins_curves, axis=0).tolist() if ins_curves else []

    return {
        "mean_deletion_auc": float(np.mean(del_aucs)) if del_aucs else 0.0,
        "std_deletion_auc": float(np.std(del_aucs, ddof=1)) if len(del_aucs) > 1 else 0.0,
        "mean_insertion_auc": float(np.mean(ins_aucs)) if ins_aucs else 0.0,
        "std_insertion_auc": float(np.std(ins_aucs, ddof=1)) if len(ins_aucs) > 1 else 0.0,
        "step_fractions": fractions,
        "mean_deletion_curve": mean_del_curve,
        "mean_insertion_curve": mean_ins_curve,
        "sample_count": len(samples),
    }
