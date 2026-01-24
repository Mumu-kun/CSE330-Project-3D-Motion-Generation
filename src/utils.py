"""
Utility functions for Human Motion Animation Generation Pipeline.

This module provides utilities compatible with MoMask's input/output format:
- Data loading: HumanML3D dim-263 feature vectors
- Motion processing: Conversion between features and joint positions
- Post-processing: Joint positions (nframe, 22, 3) → BVH files
- Evaluation: Metrics and visualization

Compatible with MoMask format:
- Input: dim-263 feature vectors
- Output: Joint positions (nframe, 22, 3) → BVH files
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json


# ============================================================================
# Data Loading Utilities (MoMask-compatible)
# ============================================================================

def load_humanml3d(
    dataset_path: Path,
    split: str = 'train',
    max_motion_length: int = 196
) -> List[Tuple[np.ndarray, str]]:
    """
    Load HumanML3D dataset with dim-263 feature vectors.
    
    Expected structure:
    dataset_path/
      ├── [split].txt (list of file IDs)
      ├── new_joint_vecs/ (motion features .npy files)
      └── texts/ (text description .txt files)
    
    Args:
        dataset_path: Path to HumanML3D dataset directory
        split: Dataset split ('train', 'val', 'test')
        max_motion_length: Maximum motion length in frames
        
    Returns:
        List of (motion_features, text_description) tuples
    """
    id_list_file = dataset_path / f'{split}.txt'
    if not id_list_file.exists():
        print(f"Warning: Split file {id_list_file} not found. Returning empty list.")
        return []

    with open(id_list_file, 'r') as f:
        file_ids = [line.strip() for line in f.readlines()]

    data = []
    motion_dir = dataset_path / 'new_joint_vecs'
    text_dir = dataset_path / 'texts'

    print(f"Loading {len(file_ids)} samples for {split} split...")
    
    for file_id in file_ids:
        motion_path = motion_dir / f'{file_id}.npy'
        text_path = text_dir / f'{file_id}.txt'
        
        if motion_path.exists() and text_path.exists():
            # Load motion
            motion = np.load(motion_path)
            
            # Load text (HumanML3D text files often have multiple lines/descriptions)
            with open(text_path, 'r') as f:
                descriptions = [line.strip().split('#')[0] for line in f.readlines()]
                # Pick one description or handle multiple if needed
                # For simplicity, we take the first descriptive one
                description = descriptions[0] if descriptions else ""
            
            data.append((motion, description))
        else:
            # Skip missing files silently or log warning
            pass
            
    return data


def preprocess_motion(
    dataset: List[Tuple[np.ndarray, str]],
    config: Any,
    normalize: bool = True
) -> List[Tuple[np.ndarray, str]]:
    """
    Preprocess motion data: padding, truncation, and normalization.
    
    Args:
        dataset: List of (motion_features, text) tuples
        config: Configuration object
        normalize: Whether to normalize motion features
        
    Returns:
        Preprocessed dataset
    """
    processed_data = []
    
    # Load mean and std if normalization is requested
    mean, std = None, None
    if normalize:
        mean_path = config.dataset_path / 'Mean.npy'
        std_path = config.dataset_path / 'Std.npy'
        if mean_path.exists() and std_path.exists():
            mean = np.load(mean_path)
            std = np.load(std_path)
        else:
            print("Warning: Mean.npy or Std.npy not found. Normalization skipped.")
            normalize = False

    for motion, text in dataset:
        # 1. Normalization
        if normalize:
            motion = (motion - mean) / std
            
        # 2. Handle length
        seq_len = motion.shape[0]
        if seq_len > config.max_motion_length:
            motion = motion[:config.max_motion_length]
        elif seq_len < config.max_motion_length:
            # Pad with zeros (or last frame, depending on preference)
            padding = np.zeros((config.max_motion_length - seq_len, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        processed_data.append((motion, text))
    
    return processed_data


def create_dataloader(
    dataset: List[Tuple[np.ndarray, str]],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create PyTorch DataLoader for HumanML3D dataset.
    
    Args:
        dataset: List of (motion_features, text) tuples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    # TODO: Create custom Dataset class if needed
    # For now, use a simple wrapper
    
    class MotionDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            motion, text = self.data[idx]
            return torch.FloatTensor(motion), text
    
    dataset_obj = MotionDataset(dataset)
    
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def load_text_motion_pairs(
    dataset_path: Path,
    split: str = 'train'
) -> List[Tuple[np.ndarray, str]]:
    """
    Load text-motion pairs from HumanML3D dataset.
    
    Args:
        dataset_path: Path to dataset
        split: Dataset split
        
    Returns:
        List of (motion_features, text_description) tuples
    """
    return load_humanml3d(dataset_path, split)


# ============================================================================
# Motion Processing Utilities
# ============================================================================

def feature_to_joints(
    motion_features: np.ndarray,
    skeleton_type: str = 'humanml3d'
) -> np.ndarray:
    """
    Convert dim-263 feature vectors to joint positions (nframe, 22, 3).
    
    Compatible with MoMask's output format.
    
    Args:
        motion_features: Motion features (seq_len, 263)
        skeleton_type: Type of skeleton ('humanml3d')
        
    Returns:
        Joint positions (nframe, 22, 3)
    """
    # TODO: Implement actual conversion
    # HumanML3D dim-263 format contains:
    # - Root position (3D)
    # - Root velocity (3D)
    # - Root rotation (6D representation)
    # - Joint rotations (in various representations)
    # - Joint velocities
    # Need to convert to joint positions in 3D space
    
    seq_len = motion_features.shape[0]
    
    # Placeholder: return random joint positions
    # In actual implementation, use HumanML3D's conversion utilities
    joints = np.random.randn(seq_len, 22, 3)
    
    print(f"TODO: Implement feature_to_joints conversion")
    print(f"Input shape: {motion_features.shape} -> Output shape: {joints.shape}")
    
    return joints


def joints_to_feature(
    joint_positions: np.ndarray,
    skeleton_type: str = 'humanml3d'
) -> np.ndarray:
    """
    Convert joint positions (nframe, 22, 3) back to dim-263 feature format.
    
    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        skeleton_type: Type of skeleton
        
    Returns:
        Motion features (seq_len, 263)
    """
    # TODO: Implement actual conversion
    # Inverse of feature_to_joints
    
    nframe = joint_positions.shape[0]
    features = np.random.randn(nframe, 263)
    
    print(f"TODO: Implement joints_to_feature conversion")
    print(f"Input shape: {joint_positions.shape} -> Output shape: {features.shape}")
    
    return features


# ============================================================================
# Post-processing Utilities (BVH conversion)
# ============================================================================

def joints_to_bvh(
    joint_positions: np.ndarray,
    fps: int = 20,
    skeleton_template: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convert joint positions (nframe, 22, 3) to BVH format.
    
    Compatible with MoMask's BVH output format.
    
    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        fps: Frames per second
        skeleton_template: Optional skeleton template for BVH structure
        
    Returns:
        BVH data dictionary with structure and motion data
    """
    # TODO: Implement BVH conversion
    # Use HumanML3D's skeleton structure (22 joints)
    # Create BVH hierarchy and convert joint positions to rotations
    
    nframe, num_joints, _ = joint_positions.shape
    
    bvh_data = {
        'hierarchy': skeleton_template or _get_default_skeleton_hierarchy(),
        'motion': {
            'frames': nframe,
            'fps': fps,
            'data': joint_positions.tolist()  # Placeholder
        }
    }
    
    print(f"TODO: Implement proper joints_to_bvh conversion")
    print(f"Input: {joint_positions.shape} -> BVH format")
    
    return bvh_data


def _get_default_skeleton_hierarchy() -> Dict:
    """
    Get default skeleton hierarchy for HumanML3D (22 joints).
    
    Returns:
        Skeleton hierarchy dictionary
    """
    # TODO: Define actual HumanML3D skeleton hierarchy
    # 22 joints structure compatible with MoMask
    return {
        'root': {'children': ['pelvis']},
        'pelvis': {'children': ['spine1', 'left_hip', 'right_hip']},
        # ... rest of skeleton structure
    }


def save_bvh(bvh_data: Dict[str, Any], output_path: Path) -> None:
    """
    Save BVH data to file.
    
    Compatible with MoMask's BVH file format.
    
    Args:
        bvh_data: BVH data dictionary
        output_path: Path to save BVH file
    """
    # TODO: Implement BVH file writing
    # Write BVH header (hierarchy) and motion data
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Placeholder: write basic BVH structure
    with open(output_path, 'w') as f:
        f.write("HIERARCHY\n")
        f.write("ROOT root\n")
        f.write("{\n")
        f.write("  OFFSET 0.0 0.0 0.0\n")
        f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        f.write("}\n")
        f.write("MOTION\n")
        f.write(f"Frames: {bvh_data['motion']['frames']}\n")
        f.write(f"Frame Time: {1.0 / bvh_data['motion']['fps']:.6f}\n")
        # TODO: Write actual motion data
    
    print(f"TODO: Implement complete BVH file writing")
    print(f"Saved BVH to {output_path}")


def save_joints(joint_positions: np.ndarray, output_path: Path) -> None:
    """
    Save joint positions as numpy file.
    
    Compatible with MoMask's output format.
    
    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        output_path: Path to save numpy file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, joint_positions)
    print(f"Saved joints to {output_path}")


def validate_bvh(bvh_path: Path) -> bool:
    """
    Validate BVH file structure.
    
    Args:
        bvh_path: Path to BVH file
        
    Returns:
        True if valid, False otherwise
    """
    # TODO: Implement BVH validation
    # Check file structure, hierarchy, motion data format
    
    if not bvh_path.exists():
        return False
    
    # Basic validation: check if file can be read
    try:
        with open(bvh_path, 'r') as f:
            content = f.read()
            if 'HIERARCHY' in content and 'MOTION' in content:
                return True
    except Exception:
        return False
    
    return False


# ============================================================================
# Evaluation Utilities
# ============================================================================

def compute_metrics(
    generated_joints: List[np.ndarray],
    ground_truth_joints: List[np.ndarray],
    generated_texts: List[str],
    gt_texts: List[str]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for generated motions.
    
    Compatible with HumanML3D evaluation metrics.
    
    Metrics:
    - FID (Fréchet Inception Distance): Motion quality
    - Diversity: Motion variety
    - R-Precision: Text-motion alignment
    
    Args:
        generated_joints: List of generated joint positions
        ground_truth_joints: List of ground truth joint positions
        generated_texts: Text descriptions for generated motions
        gt_texts: Ground truth text descriptions
        
    Returns:
        Dictionary of metric names and values
    """
    # TODO: Implement actual metric computation
    # Use HumanML3D evaluation utilities if available
    
    metrics = {
        'fid': 0.0,  # Fréchet Inception Distance
        'diversity': 0.0,  # Motion diversity
        'r_precision': 0.0,  # Text-motion alignment
        'mm_dist': 0.0,  # Multi-modal distance
    }
    
    print("TODO: Implement evaluation metrics computation")
    
    return metrics


def visualize_motion(
    joint_positions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Motion Visualization",
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize motion from joint positions.
    
    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        ground_truth: Optional ground truth for comparison
        title: Plot title
        save_path: Optional path to save visualization
    """
    # TODO: Implement motion visualization
    # Create stick figure animation or 3D plot
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory of root joint (or center of mass)
    root_trajectory = joint_positions[:, 0, :]  # Assuming first joint is root
    ax.plot(root_trajectory[:, 0], root_trajectory[:, 1], root_trajectory[:, 2], 
            label='Generated', linewidth=2)
    
    if ground_truth is not None:
        gt_root = ground_truth[:, 0, :]
        ax.plot(gt_root[:, 0], gt_root[:, 1], gt_root[:, 2], 
                label='Ground Truth', linewidth=2, linestyle='--')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_motions(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    save_path: Optional[Path] = None
) -> None:
    """
    Compare generated motion with ground truth.
    
    Args:
        generated_joints: Generated joint positions (nframe, 22, 3)
        ground_truth_joints: Ground truth joint positions (nframe, 22, 3)
        save_path: Optional path to save comparison
    """
    visualize_motion(
        generated_joints,
        ground_truth=ground_truth_joints,
        title="Generated vs Ground Truth",
        save_path=save_path
    )
