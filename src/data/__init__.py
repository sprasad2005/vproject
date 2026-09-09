"""Data module for RiceGuard."""

from src.data.annotations import (
    BoundingBox,
    parse_bounding_box_annotation,
    parse_segmentation_mask,
    parse_yolo_bbox_content,
    parse_yolo_bbox_file,
)
from src.data.augmentation import (
    build_transforms,
    get_eval_transforms,
    get_train_transforms,
)
from src.data.dataset import RiceLeafDataset, create_dataloader
from src.data.dataset_inspector import DatasetInspector
from src.data.dataset_registry import (
    DATASETS,
    get_dataset_info,
    is_external,
    is_training_allowed,
    is_validation_allowed,
    is_xai_ground_truth,
    list_registered_datasets,
    validate_dataset_usage,
)
from src.data.label_mapping import (
    BD5_RAW_TO_CANONICAL,
    CANONICAL_CLASS_TO_IDX,
    CANONICAL_IDX_TO_CLASS,
    CANONICAL_PRIMARY_CLASSES,
    PRIMARY_RAW_TO_CANONICAL,
    RICESEG_RAW_TO_CANONICAL,
    SETHY_RAW_TO_CANONICAL,
    YOLO_ID_TO_CANONICAL,
    class_to_index,
    get_canonical_class_list,
    index_to_class,
    normalize_class_name,
)
from src.data.manifest import ManifestGenerator
from src.data.preprocessing import get_default_normalization
from src.data.quality_checks import (
    DuplicateAuditor,
    calculate_sha256,
    compute_dhash,
    hamming_distance,
    validate_image_file,
)
from src.data.splitting import PrimaryDatasetSplitter

# Alias for backwards compatibility
build_augmentation_pipeline = build_transforms
BaseRiceDataset = RiceLeafDataset

__all__ = [
    "RiceLeafDataset",
    "BaseRiceDataset",
    "create_dataloader",
    "DatasetInspector",
    "DATASETS",
    "get_dataset_info",
    "is_training_allowed",
    "is_validation_allowed",
    "is_external",
    "is_xai_ground_truth",
    "validate_dataset_usage",
    "list_registered_datasets",
    "CANONICAL_PRIMARY_CLASSES",
    "CANONICAL_CLASS_TO_IDX",
    "CANONICAL_IDX_TO_CLASS",
    "PRIMARY_RAW_TO_CANONICAL",
    "SETHY_RAW_TO_CANONICAL",
    "BD5_RAW_TO_CANONICAL",
    "RICESEG_RAW_TO_CANONICAL",
    "YOLO_ID_TO_CANONICAL",
    "normalize_class_name",
    "get_canonical_class_list",
    "class_to_index",
    "index_to_class",
    "get_default_normalization",
    "build_transforms",
    "get_train_transforms",
    "get_eval_transforms",
    "build_augmentation_pipeline",
    "BoundingBox",
    "parse_yolo_bbox_content",
    "parse_yolo_bbox_file",
    "parse_bounding_box_annotation",
    "parse_segmentation_mask",
    "validate_image_file",
    "calculate_sha256",
    "compute_dhash",
    "hamming_distance",
    "DuplicateAuditor",
    "ManifestGenerator",
    "PrimaryDatasetSplitter",
]
