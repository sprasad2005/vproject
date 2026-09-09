"""Visualization and multi-modal image rendering utilities for RiceGuard Backend."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.backend.schemas import LesionBox, Visualizations
from app.backend.utils.image_processing import pil_to_base64_data_uri


def draw_bounding_boxes(
    orig_img: Image.Image,
    lesions: List[LesionBox],
) -> Image.Image:
    """Draw predicted lesion bounding boxes with confidence labels onto the image."""
    img_copy = orig_img.copy()
    draw = ImageDraw.Draw(img_copy)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(lesions):
        x1 = box.x
        y1 = box.y
        x2 = box.x + box.width
        y2 = box.y + box.height

        # Bounding box color: Amber/Green
        color = "#10B981" if box.confidence >= 0.70 else "#F59E0B"
        border_width = max(2, int(min(orig_img.size) * 0.006))

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=border_width)

        # Label tag
        label = f"Lesion {i+1} ({box.confidence*100:.0f}%)"
        tag_padding = 4
        tag_y = max(0, y1 - 18)
        tag_x = x1

        # Draw label background
        draw.rectangle(
            [tag_x, tag_y, tag_x + len(label) * 7 + tag_padding * 2, tag_y + 16],
            fill=color,
        )
        # Draw label text
        draw.text((tag_x + tag_padding, tag_y + 2), label, fill="#FFFFFF", font=font)

    return img_copy


def generate_gradcam_overlay(
    orig_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Generate Grad-CAM heatmap overlay blended with original image."""
    orig_arr = np.array(orig_img)  # RGB uint8

    # Apply Jet colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    blended = np.clip(
        (1.0 - alpha) * orig_arr.astype(np.float32) + alpha * heatmap_color_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(blended)


def generate_combined_visualization(
    orig_img: Image.Image,
    heatmap: np.ndarray,
    lesions: List[LesionBox],
    alpha: float = 0.40,
) -> Image.Image:
    """Generate combined multi-modal visualization (Heatmap Overlay + Bounding Boxes)."""
    # 1. First create heatmap overlay
    overlay_img = generate_gradcam_overlay(orig_img, heatmap, alpha=alpha)

    # 2. Draw bounding boxes on top of overlay
    combined_img = draw_bounding_boxes(overlay_img, lesions)

    return combined_img


def build_visualizations(
    orig_img: Image.Image,
    heatmap: np.ndarray,
    lesions: List[LesionBox],
) -> Visualizations:
    """Build all 4 visual output representations and encode as base64 Data URIs."""
    # 1. Original
    orig_b64 = pil_to_base64_data_uri(orig_img, format="PNG")

    # 2. Localization
    loc_img = draw_bounding_boxes(orig_img, lesions)
    loc_b64 = pil_to_base64_data_uri(loc_img, format="PNG")

    # 3. Grad-CAM
    gradcam_img = generate_gradcam_overlay(orig_img, heatmap)
    gradcam_b64 = pil_to_base64_data_uri(gradcam_img, format="PNG")

    # 4. Combined
    comb_img = generate_combined_visualization(orig_img, heatmap, lesions)
    comb_b64 = pil_to_base64_data_uri(comb_img, format="PNG")

    return Visualizations(
        original=orig_b64,
        localization=loc_b64,
        gradcam=gradcam_b64,
        combined=comb_b64,
    )
