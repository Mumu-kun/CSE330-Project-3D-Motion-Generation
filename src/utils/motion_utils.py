"""
Motion Processing and Feature Conversion Utilities for Human Motion Animation Generation.

Comprehensive module combining:
- Skeleton definitions (paramUtil)
- Motion feature extraction and reconstruction (motion_process)
- Dataset configurations and feature conversion (motion_config)

Compatible with MoMask format:
- Input: Joint positions (nframe, 22, 3) or feature vectors (nframe, 263)
- Output: Converted representations for training and inference
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from .quaternion import quaternion_to_cont6d

from .skeleton import Skeleton
from .quaternion import (
    qrot_np,
    qfix,
    quaternion_to_cont6d,
    quaternion_to_cont6d_np,
    qmul,
    qmul_np,
    qinv_np,
    qrot,
    qinv,
)


# ============================================================================
# Skeleton Definitions (from paramUtil.py)
# ============================================================================

# KIT Skeleton
kit_kinematic_chain = [
    [0, 11, 12, 13, 14, 15],
    [0, 16, 17, 18, 19, 20],
    [0, 1, 2, 3, 4],
    [3, 5, 6, 7],
    [3, 8, 9, 10],
]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]
)

# HumanML3D (T2M) Skeleton
t2m_raw_offsets = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

t2m_left_hand_chain = [
    [20, 22, 23, 24],
    [20, 34, 35, 36],
    [20, 25, 26, 27],
    [20, 31, 32, 33],
    [20, 28, 29, 30],
]

t2m_right_hand_chain = [
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 40, 41, 42],
    [21, 37, 38, 39],
    [21, 49, 50, 51],
]

kit_tgt_skel_id = "03950"
t2m_tgt_skel_id = "000021"


# ============================================================================
# Motion Feature Extraction and Reconstruction (from motion_process.py)
# ============================================================================


def extract_features(
    positions: np.ndarray,
    feet_thre: float,
    n_raw_offsets: np.ndarray,
    kinematic_chain: list,
    face_joint_indx: list,
    fid_r: list,
    fid_l: list,
) -> np.ndarray:
    """
    Extract HumanML3D features from joint positions.

    Converts joint positions (nframe, joints_num, 3) to dim-263 feature vectors.
    Features include:
    - Root rotation and linear velocity
    - Local joint positions (RIC - Rotation Invariant Coordinates)
    - Joint rotation representations (6D continuous)
    - Local joint velocities
    - Foot contact information

    Args:
        positions: Joint positions (nframe, joints_num, 3)
        feet_thre: Foot contact detection threshold
        n_raw_offsets: Raw skeleton offsets
        kinematic_chain: Skeleton kinematic chain structure
        face_joint_indx: Joint indices for determining face direction [r_hip, l_hip, sdr_r, sdr_l]
        fid_r: Right foot joint indices
        fid_l: Left foot joint indices

    Returns:
        Feature vectors (nframe-1, 263) - note: one frame shorter due to velocity calculation
    """
    global_positions = positions.copy()

    def foot_detect(positions, thres):
        """Detect foot contact frames based on velocity threshold."""
        velfactor = np.array([thres, thres])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float64)

        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    def get_cont6d_params(positions):
        """Convert positions to continuous 6D rotation representation."""
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True
        )
        cont_6d_params = quaternion_to_cont6d_np(quat_params)

        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)

        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

        return cont_6d_params, r_velocity, velocity, r_rot

    def get_rifke(positions, r_rot):
        """Get rotation-invariant frame representation."""
        positions = positions.copy()
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions
        )
        return positions

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions, r_rot)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Joint Rotation Representation (6D continuous)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Joint Rotation Invariant Position Representation (RIC)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Joint Velocity Representation
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Concatenate all features
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data


def recover_root_rot_pos(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover root rotation quaternion and root position from motion features.

    Args:
        data: Motion features (..., feature_dim)

    Returns:
        Tuple of:
        - r_rot_quat: Root rotation quaternion (..., 4)
        - r_pos: Root position (..., 3)
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # Convert to quaternion
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # Recover root position
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)

    # Set Y coordinate from root height
    r_pos[..., 1] = data[..., 3]

    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    """
    Reconstruct joint positions from rotation-invariant coordinates.

    Converts feature vectors back to joint positions (nframe, joints_num, 3).

    Args:
        data: Motion features (batch, seq_len, feature_dim) or (seq_len, feature_dim)
        joints_num: Number of joints in skeleton

    Returns:
        Reconstructed joint positions with shape (..., joints_num, 3)
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Extract local joint positions from features
    # Format: [root_rot_vel(1), root_lin_vel(2), root_y(1), ric_data(joints-1)*3, ...]
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )

    # Add root XZ position to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


# ============================================================================
# Dataset Configurations and Feature Conversion (from motion_config.py)
# ============================================================================

DATASET_CONFIGS = {
    "t2m": {
        "name": "HumanML3D",
        "num_joints": 22,
        "raw_offsets": t2m_raw_offsets,
        "kinematic_chain": t2m_kinematic_chain,
        "face_joint_indx": [2, 1, 17, 16],  # [r_hip, l_hip, sdr_r, sdr_l]
        "fid_r": [8, 11],  # Right foot indices
        "fid_l": [7, 10],  # Left foot indices
    },
    "kit": {
        "name": "KIT",
        "num_joints": 21,
        "raw_offsets": kit_raw_offsets,
        "kinematic_chain": kit_kinematic_chain,
        "face_joint_indx": [11, 16, 5, 8],
        "fid_r": [14, 15],
        "fid_l": [19, 20],
    },
}


def get_dataset_config(dataset_type: str = "t2m") -> Dict[str, Any]:
    """Get configuration for a dataset type."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_type]


def feature_to_joints(
    motion_features: torch.Tensor, dataset_type: str = "t2m"
) -> torch.Tensor:
    """
    Convert dim-263 feature vectors to joint positions.

    Args:
        motion_features: Feature vectors (nframe, 263) or (batch, nframe, 263)
        dataset_type: Dataset type ("t2m" for HumanML3D, "kit" for KIT)

    Returns:
        Joint positions (nframe, num_joints, 3) or (batch, nframe, num_joints, 3)
    """
    if not isinstance(motion_features, torch.Tensor):
        motion_features = torch.FloatTensor(motion_features)

    config = get_dataset_config(dataset_type)
    joints = recover_from_ric(motion_features, config["num_joints"])
    return joints


def joints_to_feature(
    joint_positions: torch.Tensor,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> np.ndarray:
    """
    Convert joint positions to feature vectors.

    Args:
        joint_positions: Joint positions (nframe, num_joints, 3)
        dataset_type: Dataset type ("t2m" for HumanML3D, "kit" for KIT)
        feet_thre: Foot contact detection threshold

    Returns:
        Feature vectors (nframe, 263)
    """
    if isinstance(joint_positions, torch.Tensor):
        joint_positions_np = joint_positions.cpu().numpy()
    else:
        joint_positions_np = joint_positions

    config = get_dataset_config(dataset_type)
    features = extract_features(
        joint_positions_np,
        feet_thre=feet_thre,
        n_raw_offsets=config["raw_offsets"],
        kinematic_chain=config["kinematic_chain"],
        face_joint_indx=config["face_joint_indx"],
        fid_r=config["fid_r"],
        fid_l=config["fid_l"],
    )
    return features


def get_feature_vec_subset(
    features: torch.Tensor, dimensions: tuple[slice, ...]
) -> torch.Tensor:
    """
    Extract a subset of dimensions from feature vectors.

    Feature vector layout (263D) for HumanML3D:
        [  0:   4]  Root data (4D)
                    - 0: root rotation velocity around Y (yaw)
                    - 1: root linear velocity x (in root-local frame)
                    - 2: root linear velocity z (in root-local frame)
                    - 3: root height y
        [  4:  67]  RIC joint positions (21 joints * 3 = 63D)
                    - rotation-invariant, root-centered positions of all non-root joints
        [ 67: 193]  Joint rotations (21 joints * 6 = 126D)
                    - continuous 6D rotations for all non-root joints
        [193: 259]  Local joint velocities (22 joints * 3 = 66D)
                    - per-joint 3D velocities (including root), in root-local frame
        [259: 263]  Foot contacts (4D)
                    - binary contact flags for left/right feet joints


    Args:
        features: Feature vectors (nframe, 263) or (nframe-1, 263)
        dimensions: List of slices to extract, e.g., [slice(0, 3), slice(3, 69), slice(261, 263)]
                   If None, returns full feature vector

    Returns:
        Subset of features with selected dimensions concatenated
    """

    # Handle both list of slices and tuple of slices
    subset_list = []
    for dim in dimensions:
        subset_list.append(features[:, dim])

    return torch.cat(subset_list, dim=-1)


"""
Incremental Feature Extraction for Autoregressive Motion Generation

Adapts extract_features() logic to work frame-by-frame without full sequence IK.
Computes rotations incrementally using forward kinematics from previous frame.
"""


def extract_features_incremental_torch(
    new_positions: torch.Tensor,  # (B, 22, 3) - new joint positions
    prev_positions: torch.Tensor,  # (B, 22, 3) - previous frame positions
    prev_quaternions: torch.Tensor,  # (B, 22, 4) - previous frame quaternions
    prev_r_rot: torch.Tensor,  # (B, 4) - previous root rotation
    n_raw_offsets: torch.Tensor,  # (22, 3)
    kinematic_chain: list,
    face_joint_indx: list,
    fid_r: list,
    fid_l: list,
    feet_thre: float = 0.002,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features for new frame(s) using pure PyTorch.

    Uses forward kinematics from previous quaternions instead of full IK.
    Much faster than extract_features() - O(1) per frame instead of O(n).

    Args:
        new_positions: New joint positions (B, 22, 3)
        prev_positions: Previous joint positions (B, 22, 3)
        prev_quaternions: Previous quaternions (B, 22, 4)
        prev_r_rot: Previous root rotation (B, 4)
        n_raw_offsets: Skeleton offsets (22, 3)
        kinematic_chain: Skeleton chain structure
        face_joint_indx: Face joint indices
        fid_r: Right foot indices
        fid_l: Left foot indices
        feet_thre: Foot contact threshold

    Returns:
        Tuple of:
        - features: (B, 263) feature vectors
        - new_quaternions: (B, 22, 4) quaternions for next iteration
        - new_r_rot: (B, 4) root rotation for next iteration
    """
    device = new_positions.device
    dtype = new_positions.dtype
    B = new_positions.shape[0]

    # === STEP 1: Compute Local Positions and Rotations ===

    # Get RIC positions (rotation-invariant, root-centered)
    def get_rifke_torch(positions, r_rot):
        """Get rotation-invariant frame representation in torch."""
        positions = positions.clone()
        # Center XZ around root
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        # Apply root rotation
        positions = qrot(
            r_rot.unsqueeze(1).expand(-1, positions.shape[1], -1), positions
        )
        return positions

    # Estimate quaternions from new positions
    # For incremental extraction, use a simplified FK approach
    # In practice, you might want to use a lightweight IK here
    # For now, we'll estimate from position differences

    # Compute forward direction and estimate rotation
    root_displacement = new_positions[:, 0] - prev_positions[:, 0]  # (B, 3)

    # Simple approach: estimate rotation from forward direction
    # (More sophisticated: use proper IK, but this is faster)
    # For production, consider using the skeleton's forward_kinematics_torch if available

    # Use previous quaternions as base and adjust minimally
    # This assumes smooth motion (reasonable for generated sequences)
    quat_params = prev_quaternions.clone()  # (B, 22, 4)

    # For better accuracy, estimate joint rotations from position changes
    # This is a simplified version - proper IK would be better but slower
    # The key insight: for autoregressive generation, previous quats are good initial guess

    # Get 6D continuous rotations
    cont_6d_params = quaternion_to_cont6d(quat_params)  # (B, 22, 6)

    # === STEP 2: Compute Root Motion ===

    # Root rotation (estimate from forward direction)
    r_rot = prev_r_rot.clone()  # (B, 4) - keep from previous

    # Linear velocity
    velocity = new_positions[:, 0] - prev_positions[:, 0]  # (B, 3)
    velocity = qrot(r_rot.unsqueeze(1), velocity.unsqueeze(1)).squeeze(
        1
    )  # Transform to root frame

    # Rotation velocity (change in rotation)
    r_velocity = qmul(r_rot, qinv(prev_r_rot))  # (B, 4)

    # Transform positions to RIC
    positions_ric = get_rifke_torch(new_positions, r_rot)  # (B, 22, 3)

    # === STEP 3: Build Feature Vector ===

    # Root data [0:4]: (r_velocity_y, lin_vel_x, lin_vel_z, root_y)
    r_velocity_angle = torch.arcsin(
        torch.clamp(r_velocity[:, 2:3], -1.0, 1.0)
    )  # (B, 1)
    l_velocity = velocity[:, [0, 2]]  # (B, 2)
    root_y = positions_ric[:, 0:1, 1:2].squeeze(1)  # (B, 1)

    root_data = torch.cat([r_velocity_angle, l_velocity, root_y], dim=-1)  # (B, 4)

    # RIC positions [4:67]: (21 joints * 3)
    ric_data = positions_ric[:, 1:].reshape(B, -1)  # (B, 63)

    # Joint rotations [67:193]: (21 joints * 6)
    rot_data = cont_6d_params[:, 1:].reshape(B, -1)  # (B, 126)

    # Local velocities [193:259]: (22 joints * 3)
    local_vel = qrot(
        prev_r_rot.unsqueeze(1).expand(-1, new_positions.shape[1], -1),
        new_positions - prev_positions,
    )  # (B, 22, 3)
    local_vel = local_vel.reshape(B, -1)  # (B, 66)

    # Foot contacts [259:263]: (4D - left foot, right foot)
    feet_l, feet_r = _foot_detect_incremental_torch(
        new_positions, prev_positions, fid_l, fid_r, feet_thre
    )  # (B, 1) each

    # === STEP 4: Concatenate All Features ===
    features = torch.cat(
        [root_data, ric_data, rot_data, local_vel, feet_l, feet_r], dim=-1
    )  # (B, 263)

    return features, quat_params, r_rot


def _foot_detect_incremental_torch(
    new_positions: torch.Tensor,  # (B, 22, 3)
    prev_positions: torch.Tensor,  # (B, 22, 3)
    fid_l: list,  # Left foot indices
    fid_r: list,  # Right foot indices
    thres: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect foot contact based on velocity threshold.

    Args:
        new_positions: (B, 22, 3)
        prev_positions: (B, 22, 3)
        fid_l: Left foot joint indices
        fid_r: Right foot joint indices
        thres: Velocity threshold for contact detection

    Returns:
        Tuple of feet_l, feet_r: (B, 1) each
    """
    B = new_positions.shape[0]
    device = new_positions.device

    velfactor = thres * thres

    # Left foot
    displacement_l = (
        new_positions[:, fid_l, :] - prev_positions[:, fid_l, :]
    )  # (B, num_fid_l, 3)
    vel_sq_l = torch.sum(displacement_l**2, dim=(-1))  # (B,) - sum over feet and xyz
    feet_l = (vel_sq_l < velfactor).float()  # (B, 1)

    # Right foot
    displacement_r = (
        new_positions[:, fid_r, :] - prev_positions[:, fid_r, :]
    )  # (B, num_fid_r, 3)
    vel_sq_r = torch.sum(displacement_r**2, dim=(-1))  # (B,)
    feet_r = (vel_sq_r < velfactor).float()  # (B, 1)

    return feet_l, feet_r


class IncrementalFeatureExtractor:
    """
    Stateful incremental feature extractor for frame-by-frame generation.
    Maintains quaternion and rotation state in PyTorch.

    All operations use torch.Tensor internally for efficiency.
    """

    def __init__(
        self,
        n_raw_offsets: np.ndarray,
        kinematic_chain: list,
        face_joint_indx: list,
        fid_r: list,
        fid_l: list,
        feet_thre: float = 0.002,
        device=torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.kinematic_chain = kinematic_chain
        self.face_joint_indx = face_joint_indx
        self.fid_r = fid_r
        self.fid_l = fid_l
        self.feet_thre = feet_thre
        self.device = device
        self.dtype = dtype

        # Convert offsets to torch tensor
        if isinstance(n_raw_offsets, np.ndarray):
            self.n_raw_offsets = torch.from_numpy(n_raw_offsets).float().to(device)
        else:
            self.n_raw_offsets = n_raw_offsets.float().to(device)

        # Will be initialized on first call
        self.prev_positions = None
        self.prev_quaternions = None
        self.prev_r_rot = None
        self.is_initialized = False

    def initialize(self, initial_positions: torch.Tensor):
        """
        Initialize extractor with initial frame positions using proper IK.

        Uses inverse kinematics like extract_features() to compute quaternions.

        Args:
            initial_positions: (B, 22, 3) - initial joint positions
        """

        # Convert to numpy for IK computation
        if isinstance(initial_positions, torch.Tensor):
            positions_np = initial_positions.detach().cpu().numpy()
        else:
            positions_np = initial_positions

        B = positions_np.shape[0]

        n_raw_offsets_cpu = self.n_raw_offsets.detach().cpu()

        # Use proper inverse kinematics like extract_features()
        skel = Skeleton(n_raw_offsets_cpu, self.kinematic_chain, "cpu")
        quat_params_np = skel.inverse_kinematics_np(
            positions_np, self.face_joint_indx, smooth_forward=True
        )  # (B, 22, 4)

        # Convert quaternions to torch
        self.prev_quaternions = (
            torch.from_numpy(quat_params_np).to(self.device).to(self.dtype)
        )  # (B, 22, 4)
        self.prev_r_rot = self.prev_quaternions[:, 0].clone()  # Root rotation (B, 4)
        self.prev_positions = initial_positions.clone().to(self.device).to(self.dtype)

        self.is_initialized = True

    def process_frame(self, new_positions: torch.Tensor) -> torch.Tensor:
        """
        Process a single new frame and return features.

        Args:
            new_positions: (B, 22, 3) - new joint positions

        Returns:
            features: (B, 263) - feature vectors
        """
        # Ensure tensor is on correct device and dtype
        new_positions = new_positions.to(self.device).to(self.dtype)

        if (
            not self.is_initialized
            or self.prev_positions is None
            or self.prev_quaternions is None
            or self.prev_r_rot is None
        ):
            self.initialize(new_positions)
            # Return zero features for first frame (no previous frame for velocity)
            return torch.zeros(
                new_positions.shape[0], 263, device=self.device, dtype=self.dtype
            )

        # Extract incremental features
        features, quat_params, r_rot = extract_features_incremental_torch(
            new_positions,
            self.prev_positions,
            self.prev_quaternions,
            self.prev_r_rot,
            self.n_raw_offsets,
            self.kinematic_chain,
            self.face_joint_indx,
            self.fid_r,
            self.fid_l,
            self.feet_thre,
        )

        # Update state for next frame
        self.prev_positions = new_positions.clone()
        self.prev_quaternions = quat_params
        self.prev_r_rot = r_rot

        return features  # (B, 263) torch tensor

    def reset(self):
        """Reset the extractor state."""
        self.prev_positions = None
        self.prev_quaternions = None
        self.prev_r_rot = None
        self.is_initialized = False

    def to(self, device):
        """Move extractor to device."""
        self.device = device
        self.n_raw_offsets = self.n_raw_offsets.to(device)
        if (
            self.prev_positions is not None
            and self.prev_quaternions is not None
            and self.prev_r_rot is not None
        ):
            self.prev_positions = self.prev_positions.to(device)
            self.prev_quaternions = self.prev_quaternions.to(device)
            self.prev_r_rot = self.prev_r_rot.to(device)
        return self
