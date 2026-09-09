"""Image validation, preprocessing, and encoding utilities for RiceGuard Backend."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Standard evaluation transform matching Phase 2B / Phase 3B evaluation
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def validate_and_load_image(image_bytes: bytes) -> Tuple[Image.Image, int, int]:
    """Validate image bytes and return RGB PIL Image and original dimensions (width, height).

    Raises:
        ValueError: If bytes are empty, corrupt, or not a valid image.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    try:
        buf = io.BytesIO(image_bytes)
        img = Image.open(buf)
        img.verify()  # Verify image integrity

        # Reopen after verify
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        width, height = img.size

        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions are invalid.")

        return img, width, height
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {str(e)}") from e


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess PIL Image into a normalized 4D PyTorch tensor [1, 3, 224, 224]."""
    tensor = EVAL_TRANSFORM(pil_img)
    return tensor.unsqueeze(0)


def pil_to_base64_data_uri(pil_img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URI string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def numpy_to_base64_data_uri(arr: np.ndarray, format: str = "PNG") -> str:
    """Convert RGB NumPy uint8 array to base64 data URI string."""
    pil_img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_base64_data_uri(pil_img, format=format)
