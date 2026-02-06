"""
Utility functions for Human Motion Animation Generation Pipeline.

This module re-exports main utilities for easy access.
Specific functionality is organized into submodules:
- dataset.py: Data loading and Text2Motion dataset
- motion_utils.py: Motion feature conversion and skeleton definitions
- visualization.py: Motion visualization
- bvh_utils.py: BVH file handling
- metrics.py: Evaluation metrics
"""

# Data Loading
from .dataset import (
    Text2MotionDataset,
    create_dataloader,
    load_sample,
)

# Motion Processing
from .motion_utils import (
    DATASET_CONFIGS,
    get_dataset_config,
    feature_to_joints,
    joints_to_feature,
    extract_features,
    get_feature_vec_subset,
    recover_from_ric,
)

# Visualization
from .visualization import (
    plot_3d_motion,
    visualize_motion,
    compare_motions,
)

# BVH Utilities
from .bvh_utils import (
    joints_to_bvh,
    save_bvh,
    save_joints,
    validate_bvh,
)

# Metrics
from .metrics import (
    compute_metrics,
)

__all__ = [
    # Data Loading
    "Text2MotionDataset",
    "create_dataloader",
    "load_sample",
    # Motion Processing
    "DATASET_CONFIGS",
    "get_dataset_config",
    "feature_to_joints",
    "joints_to_feature",
    "extract_features",
    "get_feature_vec_subset",
    "recover_from_ric",
    # Visualization
    "plot_3d_motion",
    "visualize_motion",
    "compare_motions",
    # BVH Utilities
    "joints_to_bvh",
    "save_bvh",
    "save_joints",
    "validate_bvh",
    # Metrics
    "compute_metrics",
]
