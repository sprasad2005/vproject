"""Data augmentation and preprocessing transform pipelines for RiceGuard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torchvision.transforms as T

# Standard ImageNet normalization constants
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    scale: Tuple[float, float] = (0.8, 1.0),
    hflip_prob: float = 0.5,
    rotation_degrees: float = 15.0,
    color_jitter: Optional[Dict[str, float]] = None,
) -> T.Compose:
    """Build moderate agricultural training augmentation pipeline.

    Applies conservative spatial and photometric transformations that preserve
    critical lesion morphology and diagnostic symptoms.
    """
    norm_mean = mean if mean is not None else IMAGENET_MEAN
    norm_std = std if std is not None else IMAGENET_STD

    if isinstance(image_size, int):
        target_size: Tuple[int, int] = (image_size, image_size)
    else:
        target_size = (image_size[0], image_size[1])

    cj = color_jitter or {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}

    transforms_list = [
        T.RandomResizedCrop(target_size, scale=scale),
        T.RandomHorizontalFlip(p=hflip_prob),
        T.RandomRotation(degrees=rotation_degrees),
        T.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.1),
        ),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ]

    return T.Compose(transforms_list)


def get_eval_transforms(
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    resize_scale: float = 256.0 / 224.0,
) -> T.Compose:
    """Build deterministic validation and test preprocessing pipeline."""
    norm_mean = mean if mean is not None else IMAGENET_MEAN
    norm_std = std if std is not None else IMAGENET_STD

    if isinstance(image_size, int):
        target_size: Tuple[int, int] = (image_size, image_size)
        crop_size = image_size
        resize_dim = int(image_size * resize_scale)
    else:
        target_size = (image_size[0], image_size[1])
        crop_size = target_size
        resize_dim = int(image_size[0] * resize_scale)

    transforms_list = [
        T.Resize(resize_dim),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ]

    return T.Compose(transforms_list)


def get_multitask_transforms(
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    is_training: bool = True,
) -> T.Compose:
    """Transform pipeline preserving exact normalized spatial coordinates for localization."""
    norm_mean = mean if mean is not None else IMAGENET_MEAN
    norm_std = std if std is not None else IMAGENET_STD

    if isinstance(image_size, int):
        target_size: Tuple[int, int] = (image_size, image_size)
    else:
        target_size = (image_size[0], image_size[1])

    if is_training:
        return T.Compose(
            [
                T.Resize(target_size),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=norm_mean, std=norm_std),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(target_size),
                T.ToTensor(),
                T.Normalize(mean=norm_mean, std=norm_std),
            ]
        )


def build_transforms(
    is_training: bool = True,
    config: Optional[Dict[str, Any]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
) -> T.Compose:
    """Factory helper to construct transform pipelines from project configuration."""
    cfg = config or {}
    prep_cfg = cfg.get("preprocessing", {})
    aug_cfg = cfg.get("augmentation", {})

    if image_size is not None:
        img_size = image_size
    else:
        raw_img_size = prep_cfg.get("image_size", 224)
        if isinstance(raw_img_size, list):
            img_size = (raw_img_size[0], raw_img_size[1])
        else:
            img_size = int(raw_img_size)

    mean = prep_cfg.get("mean", IMAGENET_MEAN)
    std = prep_cfg.get("std", IMAGENET_STD)

    if is_training:
        train_aug = aug_cfg.get("train", {})
        scale_tuple = tuple(train_aug.get("random_resized_crop", {}).get("scale", [0.8, 1.0]))
        hflip = train_aug.get("random_horizontal_flip", {}).get("p", 0.5)
        rotation = train_aug.get("random_rotation", {}).get("degrees", 15.0)
        cj = train_aug.get("color_jitter", None)
        return get_train_transforms(
            image_size=img_size,
            mean=mean,
            std=std,
            scale=(scale_tuple[0], scale_tuple[1]),
            hflip_prob=hflip,
            rotation_degrees=rotation,
            color_jitter=cj,
        )
    else:
        return get_eval_transforms(image_size=img_size, mean=mean, std=std)
