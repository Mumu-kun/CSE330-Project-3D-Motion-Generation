"""
Utility functions for Human Motion Animation Generation Pipeline.

This module re-exports main utilities for easy access.
Specific functionality is organized into submodules:
- dataset.py: Data loading and Text2Motion dataset
- motion_utils.py: Motion feature conversion and skeleton definitions
- visualization.py: Motion visualization
"""

# Data Loading
from utils.dataset import (
    Text2MotionDataset,
    create_dataloader,
    load_sample,
)

# Motion Processing
from utils.motion_utils import (
    DATASET_CONFIGS,
    get_dataset_config,
    features_to_positions,
    preprocess_sequence,
    IncrementalFeatureExtractor,
)

# Visualization
from utils.visualization import (
    plot_3d_motion,
    visualize_motion,
    compare_motions,
)

__all__ = [
    # Data Loading
    "Text2MotionDataset",
    "create_dataloader",
    "load_sample",
    # Motion Processing
    "DATASET_CONFIGS",
    "get_dataset_config",
    "features_to_positions",
    "preprocess_sequence",
    "IncrementalFeatureExtractor",
    # Visualization
    "plot_3d_motion",
    "visualize_motion",
    "compare_motions",
]
