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
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# HumanML3D Skeleton Connections
KINEMATIC_CHAIN = [
    [0, 1, 4, 7, 10],      # Left leg
    [0, 2, 5, 8, 11],      # Right leg
    [0, 3, 6, 9, 12, 15],  # Spine and Head
    [9, 13, 16, 18, 20],   # Left arm
    [9, 14, 17, 19, 21]    # Right arm
]



# ============================================================================
# Data Loading Utilities (MoMask-compatible)
# ============================================================================

def load_humanml3d(
    dataset_path: Path,
    split: str = 'train',
) -> List[Dict[str, Any]]:
    """
    Standard procedure: Load HumanML3D metadata and text descriptions.
    Motion features are loaded lazily in the Dataset class.
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

    print(f"Loading {split} split metadata ({len(file_ids)} samples)...")
    
    for file_id in file_ids:
        motion_path = motion_dir / f'{file_id}.npy'
        text_path = text_dir / f'{file_id}.txt'
        
        if motion_path.exists() and text_path.exists():
            with open(text_path, 'r') as f:
                descriptions = [line.strip().split('#')[0] for line in f.readlines()]
                description = descriptions[0] if descriptions else ""
            
            data.append({
                'motion_path': motion_path,
                'text': description,
                'file_id': file_id
            })
            
    return data


class MotionDataset(Dataset):
    """
    Standard HumanML3D Dataset.
    - Stats (Mean/Std) loaded in __init__.
    - Text pre-loaded in metadata.
    - Motion .npy files loaded lazily in __getitem__.
    """
    def __init__(self, data_list: List[Dict], config: Any, normalize: bool = True):
        self.data_list = data_list
        self.max_len = config.max_motion_length
        self.normalize = normalize
        
        # Standard Procedure: Load stats during initialization
        self.mean = None
        self.std = None
        if normalize:
            mean_path = config.dataset_path / 'Mean.npy'
            std_path = config.dataset_path / 'Std.npy'
            if mean_path.exists() and std_path.exists():
                self.mean = np.load(mean_path)
                self.std = np.load(std_path)
            else:
                print(f"Warning: Normalization files not found. Normalization disabled.")
                self.normalize = False

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load motion features
        motion = np.load(item['motion_path'])
        text = item['text']
        
        # 1. Normalization
        if self.normalize:
            motion = (motion - self.mean) / self.std
            
        # 2. Length handling
        seq_len = motion.shape[0]
        if seq_len > self.max_len:
            motion = motion[:self.max_len]
        elif seq_len < self.max_len:
            padding = np.zeros((self.max_len - seq_len, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
            
        return torch.FloatTensor(motion), text

def preprocess_motion(
    motion: np.ndarray,
    config: Any
):
    pass

def create_dataloader(
    dataset_metadata: List[Dict],
    config: Any,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Standard DataLoader creation."""
    dataset_obj = MotionDataset(dataset_metadata, config)
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
) -> List[Dict[str, Any]]:

    """
    Load text-motion pairs metadata from HumanML3D dataset.
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
    
    HumanML3D dim-263 format (approximate):
    - [0:1]: Root angular velocity
    - [1:3]: Root linear velocity (x, z)
    - [3:4]: Root height (y)
    - [4:67]: Local joint positions (21 joints * 3)
    """
    seq_len = motion_features.shape[0]
    num_joints = 22
    joints = np.zeros((seq_len, num_joints, 3))
    
    # Root Height
    joints[:, 0, 1] = motion_features[:, 3]
    
    # Local Joint Positions (Joints 1 to 21)
    local_pos = motion_features[:, 4:4+63].reshape(seq_len, 21, 3)
    joints[:, 1:, :] = local_pos
    
    return joints



def joints_to_feature(
    joint_positions: np.ndarray,
    skeleton_type: str = 'humanml3d'
) -> np.ndarray:
    """
    Convert joint positions (nframe, 22, 3) back to dim-263 feature format.
    
    This is a partial implementation focusing on returning positions.
    Full implementation requires calculating velocities and rotations.
    """
    nframe = joint_positions.shape[0]
    features = np.zeros((nframe, 263))
    
    # 1. Store Root Height
    features[:, 3] = joint_positions[:, 0, 1]
    
    # 2. Store Local Joint Positions (Joints 1 to 21)
    # Flatten (21, 3) to 63
    features[:, 4:4+63] = joint_positions[:, 1:, :].reshape(nframe, 63)
    
    # TODO: Fill velocities, rotations, and root velocities if needed
    
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
    Based on SMPL hierarchy flattened to 22 joints.
    """
    # Parent indices for 22 joints (HumanML3D/SMPL)
    # -1 means root.
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    
    # Joint names in standard order
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    
    # Construct local hierarchy
    hierarchy = {}
    for i, name in enumerate(joint_names):
        p_idx = parents[i]
        p_name = joint_names[p_idx] if p_idx != -1 else "root"
        
        if p_name not in hierarchy:
            hierarchy[p_name] = {'children': []}
        hierarchy[p_name]['children'].append(name)
        
        if name not in hierarchy:
            hierarchy[name] = {'children': []}
            
    return hierarchy



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


def plot_3d_motion(
    motion: np.ndarray,
    fps: int = 20,
    radius: float = 1.0,
    title: str = "Motion Visualization",
    follow_root: bool = False
) -> FuncAnimation:
    """
    Create a 3D animation of motion joint positions.
    
    Args:
        motion: Joint positions (nframe, 22, 3)
        fps: Frames per second
        radius: Radius of the viewing box around the character
        title: Plot title
        follow_root: Whether the camera should follow the root joint
        
    Returns:
        Matplotlib FuncAnimation object
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    
    colors = ['#2980b9', '#c0392b', '#27ae60', '#f39c12', '#8e44ad']
    lines = [ax.plot([], [], [], color=c, marker='o', ms=2, lw=2)[0] for c in colors]

    ax.set_xlabel('X (Side)')
    ax.set_ylabel('Z (Forward)')
    ax.set_zlabel('Y (Height)')
    ax.set_title(title)

    # Pre-calculate global bounds for static centering if not following root
    pos_min = motion.min(axis=(0, 1))
    pos_max = motion.max(axis=(0, 1))

    def update(frame):
        root = motion[frame, 0, :]
        
        if follow_root:
            ax.set_xlim3d([root[0] - radius, root[0] + radius])
            ax.set_ylim3d([root[2] - radius, root[2] + radius])
            ax.set_zlim3d([pos_min[1], pos_max[1] + radius*0.5])
        else:
            ax.set_xlim3d([pos_min[0] - radius, pos_max[0] + radius])
            ax.set_ylim3d([pos_min[2] - radius, pos_max[2] + radius])
            ax.set_zlim3d([pos_min[1], pos_max[1] + radius*0.5])
        
        for i, c_indices in enumerate(KINEMATIC_CHAIN):
            joints = motion[frame, c_indices, :]
            # Map Data Y to Plot Z (Vertical)
            lines[i].set_data(joints[:, 0], joints[:, 2])
            lines[i].set_3d_properties(joints[:, 1])
        return lines

    ani = FuncAnimation(fig, update, frames=len(motion), interval=1000/fps, blit=False)
    plt.close()
    return ani


def visualize_motion(
    joint_positions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Motion Visualization",
    save_path: Optional[Path] = None,
    fps: int = 20,
    skip_frames: int = 1,
    notebook: bool = True
) -> Any:
    """
    Visualize motion from joint positions.
    
    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        ground_truth: Optional ground truth for comparison
        title: Plot title
        save_path: Optional path to save visualization
        fps: Frames per second
        notebook: Whether to return HTML for notebook display
    """
    fps = fps/skip_frames
    ani = plot_3d_motion(joint_positions[::skip_frames], fps=fps, title=title)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), writer='ffmpeg', fps=fps)
        print(f"Saved animation to {save_path}")
    
    if notebook:
        display_html = HTML(ani.to_html5_video())
        return display_html
    
    return ani


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
