"""
Motion Processing and Feature Conversion Utilities for Human Motion Animation Generation.

271D Feature Format (Pure PyTorch) - Updated per normalization plan:
- Root height Y, Root velocity X, Root velocity Z (3D) - velocity form for X,Z
- 22 RIC positions (66D) - local positions relative to root, from actual data
- 22 6D rotations (132D) - auxiliary features from IK
- 22 local velocities (66D) - causal velocities (current - previous)
- 4D foot contacts - binary contact flags

Total: 271D per frame

Note: Root X,Z are stored as velocities for autoregressive stability.

API:
- preprocess_sequence(): Dataset preprocessing (ground truth joints)
- features_to_positions(): Reconstruction (features -> positions)
- extract_features_from_predicted(): Inference (predicted joints)
- IncrementalFeatureExtractor: Frame-by-frame inference
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .quaternion import (
    qrot,
    qinv,
    qmul,
    quaternion_to_cont6d,
    cont6d_to_matrix,
    cont6d_to_quaternion,
)


# ============================================================================
# Skeleton Definitions
# ============================================================================

T2M_RAW_OFFSETS = torch.tensor(
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
    ],
    dtype=torch.float32,
)

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],  # Left leg
    [0, 1, 4, 7, 10],  # Right leg
    [0, 3, 6, 9, 12, 15],  # Spine
    [9, 14, 17, 19, 21],  # Right arm
    [9, 13, 16, 18, 20],  # Left arm
]


# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_CONFIGS = {
    "t2m": {
        "name": "HumanML3D",
        "num_joints": 22,
        "feature_dim": 271,
        "raw_offsets": T2M_RAW_OFFSETS,
        "kinematic_chain": T2M_KINEMATIC_CHAIN,
        "face_joint_indx": [2, 1, 17, 16],  # [r_hip, l_hip, sdr_r, sdr_l]
        "fid_r": [8, 11],  # Right foot indices
        "fid_l": [7, 10],  # Left foot indices
    },
}


def get_dataset_config(dataset_type: str = "t2m") -> Dict[str, Any]:
    """Get configuration for a dataset type."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_type]


# ============================================================================
# Feature Layout Constants
# ============================================================================

FEATURE_SLICES = {
    "root_features": slice(0, 3),  # 3D: root height Y, velocity X, velocity Z
    "ric_positions": slice(3, 69),  # 66D (22 * 3)
    "rotations_6d": slice(69, 201),  # 132D (22 * 6)
    "local_velocities": slice(201, 267),  # 66D (22 * 3)
    "foot_contacts": slice(267, 271),  # 4D
}


# ============================================================================
# Internal Helper Functions
# ============================================================================


def _compute_ik(
    positions: torch.Tensor,
    raw_offsets: torch.Tensor,
    kinematic_chain: List[List[int]],
    face_joint_indx: List[int],
) -> torch.Tensor:
    """
    Compute inverse kinematics using pure PyTorch.

    Args:
        positions: Joint positions (..., 22, 3)
        raw_offsets: Skeleton bone offsets (22, 3)
        kinematic_chain: Skeleton kinematic chain
        face_joint_indx: Joint indices for facing direction [r_hip, l_hip, sdr_r, sdr_l]

    Returns:
        Quaternions (..., 22, 4)
    """
    batch_shape = positions.shape[:-2]
    device = positions.device
    dtype = positions.dtype

    # Flatten batch dimensions
    positions_flat = positions.reshape(-1, 22, 3)
    B = positions_flat.shape[0]

    # Get forward direction
    l_hip, r_hip, sdr_r, sdr_l = face_joint_indx

    across1 = positions_flat[:, r_hip] - positions_flat[:, l_hip]  # (B, 3)
    across2 = positions_flat[:, sdr_r] - positions_flat[:, sdr_l]  # (B, 3)
    across = across1 + across2
    across = across / (torch.norm(across, dim=-1, keepdim=True) + 1e-10)

    # Forward direction (cross with Y-up)
    forward = torch.cross(
        torch.tensor([[0, 1, 0]], device=device, dtype=dtype).expand(B, -1),
        across,
        dim=-1,
    )
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-10)

    # Target forward direction (Z-axis)
    target = torch.tensor([[0, 0, 1]], device=device, dtype=dtype).expand(B, -1)

    # Root rotation (from forward to target)
    root_quat = _qbetween(forward, target)

    # Initialize quaternions
    quaternions = torch.zeros(B, 22, 4, device=device, dtype=dtype)
    quaternions[:, 0] = root_quat

    # IK for each chain
    offsets = raw_offsets.unsqueeze(0).expand(B, -1, -1)  # (B, 22, 3)

    for chain in kinematic_chain:
        R = root_quat
        for i in range(len(chain) - 1):
            parent_idx = chain[i]
            child_idx = chain[i + 1]

            # Get bone direction in T-pose
            u = offsets[:, child_idx]  # (B, 3)

            # Get bone direction in current pose
            v = positions_flat[:, child_idx] - positions_flat[:, parent_idx]
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)

            # Rotation from u to v
            rot_u_v = _qbetween(u, v)

            # Local rotation
            R_loc = qmul(qinv(R), rot_u_v)

            quaternions[:, child_idx] = R_loc
            R = qmul(R, R_loc)

    return quaternions.reshape(batch_shape + (22, 4))


def _qbetween(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion that rotates v0 to v1.

    Args:
        v0: Source vectors (..., 3)
        v1: Target vectors (..., 3)

    Returns:
        Quaternions (..., 4)
    """
    # Normalize
    v0 = v0 / (torch.norm(v0, dim=-1, keepdim=True) + 1e-10)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-10)

    # Compute rotation
    dot = (v0 * v1).sum(dim=-1, keepdim=True)

    # Handle parallel vectors
    cross = torch.cross(v0, v1, dim=-1)
    w = 1.0 + dot

    q = torch.cat([w, cross], dim=-1)
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-10)

    return q


def _forward_kinematics(
    rotations_6d: torch.Tensor,
    root_pos: torch.Tensor,
    offsets: torch.Tensor,
    kinematic_chain: List[List[int]],
) -> torch.Tensor:
    """
    Forward kinematics with 6D rotations (Pure PyTorch).

    Args:
        rotations_6d: 6D rotations (..., 22, 6)
        root_pos: Root position (..., 3)
        offsets: Bone offsets (22, 3)
        kinematic_chain: Skeleton kinematic chain

    Returns:
        Joint positions (..., 22, 3)
    """
    batch_shape = rotations_6d.shape[:-2]
    device = rotations_6d.device
    dtype = rotations_6d.dtype

    # Flatten batch
    rotations_flat = rotations_6d.reshape(-1, 22, 6)
    root_pos_flat = root_pos.reshape(-1, 3)
    B = rotations_flat.shape[0]

    # Initialize positions
    positions = torch.zeros(B, 22, 3, device=device, dtype=dtype)
    positions[:, 0] = root_pos_flat

    # Expand offsets
    offsets_expanded = offsets.unsqueeze(0).expand(B, -1, -1)

    # FK for each chain
    for chain in kinematic_chain:
        # Start with root rotation matrix
        matR = cont6d_to_matrix(rotations_flat[:, 0])  # (B, 3, 3)

        for i in range(1, len(chain)):
            child_idx = chain[i]
            parent_idx = chain[i - 1]

            # Accumulate rotation
            child_rot = cont6d_to_matrix(rotations_flat[:, child_idx])
            matR = torch.bmm(matR, child_rot)

            # Compute position
            offset_vec = offsets_expanded[:, child_idx].unsqueeze(-1)  # (B, 3, 1)
            positions[:, child_idx] = (
                torch.bmm(matR, offset_vec).squeeze(-1) + positions[:, parent_idx]
            )

    return positions.reshape(batch_shape + (22, 3))


# ============================================================================
# Canonical API Functions
# ============================================================================


def preprocess_sequence(
    positions: torch.Tensor,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> torch.Tensor:
    """
    Preprocess a sequence of joint positions to 271D features.

    Used for dataset preprocessing with ground truth joints.
    Uses direct RIC transform for perfect round-trip reconstruction.

    Feature Layout (updated per normalization plan):
        [0:3]   Root height Y, Root velocity X, Root velocity Z
        [3:69]  22 RIC positions (22 * 3)
        [69:201] 22 6D rotations (22 * 6)
        [201:267] 22 local velocities (22 * 3)
        [267:271] Foot contacts (4D)

    Note: Root X,Z are stored as velocities for autoregressive stability.

    Args:
        positions: Joint positions (N, 22, 3) or (B, N, 22, 3)
        dataset_type: Dataset type ("t2m" for HumanML3D)
        feet_thre: Foot contact detection threshold

    Returns:
        Feature vectors (N, 271) or (B, N, 271)
    """
    config = get_dataset_config(dataset_type)
    raw_offsets = config["raw_offsets"]
    kinematic_chain = config["kinematic_chain"]
    face_joint_indx = config["face_joint_indx"]
    fid_r = config["fid_r"]
    fid_l = config["fid_l"]

    device = positions.device
    dtype = positions.dtype

    # Handle batch dimension
    if positions.ndim == 4:
        B, N, J, _ = positions.shape
        features_batch = []

        for b in range(B):
            pos_b = positions[b]
            feat_b = preprocess_sequence(pos_b, dataset_type, feet_thre)
            features_batch.append(feat_b)

        return torch.stack(features_batch, dim=0)

    # Single sequence mode: (N, 22, 3)
    N = positions.shape[0]

    # 1. Root features: height Y (absolute), velocity X, velocity Z
    # Per normalization plan: convert root XZ to velocity form
    root_features = torch.zeros(N, 3, device=device, dtype=dtype)
    root_features[:, 0] = positions[:, 0, 1]  # root height Y (keep absolute) at index 0
    if N > 1:
        root_features[1:, 1] = (
            positions[1:, 0, 0] - positions[:-1, 0, 0]
        )  # vx at index 1
        root_features[1:, 2] = (
            positions[1:, 0, 2] - positions[:-1, 0, 2]
        )  # vz at index 2

    # 2. IK for all frames
    quaternions = _compute_ik(
        positions, raw_offsets, kinematic_chain, face_joint_indx
    )  # (N, 22, 4)

    # 3. Root rotation
    root_quat = quaternions[:, 0].clone()  # (N, 4)

    # 4. RIC positions (center all 3 dimensions on root)
    ric = positions - positions[:, 0:1, :]  # (N, 22, 3)
    ric = qrot(root_quat.unsqueeze(1).expand(-1, 22, -1), ric)  # (N, 22, 3)

    # 5. 6D rotations
    rotations_6d = quaternion_to_cont6d(quaternions)  # (N, 22, 6)

    # 6. Causal velocities (backward differences)
    local_vel = torch.zeros(N, 22, 3, device=device, dtype=dtype)
    if N > 1:
        local_vel[1:] = qrot(
            root_quat[1:].unsqueeze(1).expand(-1, 22, -1),
            positions[1:] - positions[:-1],
        )

    # 7. Foot contacts
    feet_l = torch.zeros(N, 2, device=device, dtype=dtype)
    feet_r = torch.zeros(N, 2, device=device, dtype=dtype)

    if N > 1:
        vel_l = positions[1:, fid_l] - positions[:-1, fid_l]
        vel_r = positions[1:, fid_r] - positions[:-1, fid_r]
        feet_l[1:] = (torch.sum(vel_l**2, dim=-1) < feet_thre).float()
        feet_r[1:] = (torch.sum(vel_r**2, dim=-1) < feet_thre).float()

    # Concatenate all features
    features = torch.cat(
        [
            root_features,  # [0:3] Root height Y, velocity X, velocity Z
            ric.reshape(N, -1),
            rotations_6d.reshape(N, -1),
            local_vel.reshape(N, -1),
            feet_l,
            feet_r,
        ],
        dim=-1,
    )

    return features


def features_to_positions(
    features: torch.Tensor,
    dataset_type: str = "t2m",
) -> torch.Tensor:
    """
    Reconstruct global joint positions from 271D features.

    Uses direct RIC transform for perfect reconstruction.
    This is the canonical reconstruction function.

    Note: Features now use velocity form for root X,Z. Reconstruction
    requires cumulative sum to recover absolute positions.

    Args:
        features: Feature vectors (..., 271)
        dataset_type: Dataset type ("t2m" for HumanML3D)

    Returns:
        Global joint positions (..., 22, 3)
    """
    # Extract components
    root_features = features[..., 0:3]  # root height Y, velocity X, velocity Z
    ric = features[..., 3:69].reshape(features.shape[:-1] + (22, 3))
    rotations_6d = features[..., 69:201].reshape(features.shape[:-1] + (22, 6))

    # Get root quaternion from 6D rotation
    root_quat = cont6d_to_quaternion(rotations_6d[..., 0, :])  # (..., 4)

    # Reconstruct root position from velocity form
    # root_features[..., 0] = root height Y (absolute)
    # root_features[..., 1] = root velocity X
    # root_features[..., 2] = root velocity Z
    root_height_y = root_features[..., 0:1]  # (..., 1)
    root_vel_x = root_features[..., 1:2]  # (..., 1)
    root_vel_z = root_features[..., 2:3]  # (..., 1)

    # Cumulative sum to recover absolute X and Z positions
    root_pos_x = torch.cumsum(root_vel_x, dim=-1)
    root_pos_z = torch.cumsum(root_vel_z, dim=-1)
    global_root_pos = torch.cat([root_pos_x, root_height_y, root_pos_z], dim=-1)

    # Direct transform: RIC -> global
    # global = root_pos + rotate_inverse(RIC, root_rot)
    root_quat_expanded = root_quat.unsqueeze(-2).expand(root_quat.shape[:-1] + (22, -1))
    positions = global_root_pos.unsqueeze(-2) + qrot(qinv(root_quat_expanded), ric)

    return positions


# ============================================================================
# Incremental Feature Extractor for Autoregressive Generation
# ============================================================================


class IncrementalFeatureExtractor:
    """
    Stateful incremental feature extractor for frame-by-frame generation.

    Used during inference/motion generation when processing one frame at a time.
    Uses FK-based extraction for predicted joints to maintain kinematic consistency.

    Feature Layout (271D) - Updated per normalization plan:
        [0:3]   Root height Y, Root velocity X, Root velocity Z
        [3:69]  22 RIC positions
        [69:201] 22 6D rotations
        [201:267] 22 local velocities
        [267:271] Foot contacts

    Note: Root X,Z are stored as velocities for autoregressive stability.
    """

    def __init__(
        self,
        dataset_type: str = "t2m",
        feet_thre: float = 0.002,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the incremental feature extractor.

        Args:
            dataset_type: Dataset type ("t2m" for HumanML3D)
            feet_thre: Foot contact threshold
            device: Torch device
            dtype: Torch dtype
        """
        config = get_dataset_config(dataset_type)
        self.raw_offsets = config["raw_offsets"].to(device).to(dtype)
        self.kinematic_chain = config["kinematic_chain"]
        self.face_joint_indx = config["face_joint_indx"]
        self.fid_r = config["fid_r"]
        self.fid_l = config["fid_l"]
        self.feet_thre = feet_thre
        self.device = device
        self.dtype = dtype

        # State for incremental extraction
        self.prev_positions: Optional[torch.Tensor] = None
        self.is_initialized = False

    def initialize(self, initial_positions: torch.Tensor) -> torch.Tensor:
        """
        Initialize extractor with initial frame positions.

        Args:
            initial_positions: (B, 22, 3) initial joint positions

        Returns:
            Zero features for first frame (B, 271)
        """
        initial_positions = initial_positions.to(self.device).to(self.dtype)
        B = initial_positions.shape[0]

        # Store state
        self.prev_positions = initial_positions.clone()

        # Compute IK for initial frame
        quaternions = _compute_ik(
            initial_positions,
            self.raw_offsets,
            self.kinematic_chain,
            self.face_joint_indx,
        )

        # Compute FK for RIC consistency
        rotations_6d = quaternion_to_cont6d(quaternions)
        root_pos = initial_positions[:, 0]
        fk_positions = _forward_kinematics(
            rotations_6d, root_pos, self.raw_offsets, self.kinematic_chain
        )

        # Store FK positions for next frame's velocity computation
        self.prev_fk_positions = fk_positions

        self.is_initialized = True

        # Return zero features for first frame
        return torch.zeros(B, 271, device=self.device, dtype=self.dtype)

    def process_frame(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame and extract 271D features.

        Args:
            positions: (B, 22, 3) joint positions for current frame

        Returns:
            features: (B, 271) feature vectors
        """
        positions = positions.to(self.device).to(self.dtype)

        if not self.is_initialized:
            return self.initialize(positions)

        B = positions.shape[0]

        # === 1. Root features: height Y (absolute), velocity X, velocity Z ===
        root_height_y = positions[:, 0, 1:2]  # (B, 1)
        root_vel_x = positions[:, 0, 0:1] - self.prev_positions[:, 0, 0:1]  # (B, 1)
        root_vel_z = positions[:, 0, 2:3] - self.prev_positions[:, 0, 2:3]  # (B, 1)
        root_features = torch.cat(
            [root_height_y, root_vel_x, root_vel_z], dim=-1
        )  # (B, 3)

        # === 2. IK for rotations ===
        quaternions = _compute_ik(
            positions, self.raw_offsets, self.kinematic_chain, self.face_joint_indx
        )

        # === 3. Root rotation ===
        root_quat = quaternions[:, 0]  # (B, 4)

        # === 4. RIC from FK positions ===
        rotations_6d = quaternion_to_cont6d(quaternions)

        # Get root position for FK (need absolute position)
        global_root_pos = positions[:, 0]  # (B, 3)

        # Compute FK for kinematic consistency
        fk_positions = _forward_kinematics(
            rotations_6d, global_root_pos, self.raw_offsets, self.kinematic_chain
        )

        # Compute RIC from FK positions
        ric = fk_positions - fk_positions[:, 0:1]
        ric = qrot(root_quat.unsqueeze(1).expand(-1, 22, -1), ric)

        # === 5. Causal velocities ===
        local_vel = qrot(
            root_quat.unsqueeze(1).expand(-1, 22, -1), positions - self.prev_positions
        )

        # === 6. Foot contacts ===
        foot_vel = positions - self.prev_positions
        feet_l = (
            torch.sum(foot_vel[:, self.fid_l] ** 2, dim=-1) < self.feet_thre
        ).float()
        feet_r = (
            torch.sum(foot_vel[:, self.fid_r] ** 2, dim=-1) < self.feet_thre
        ).float()

        # === Update state ===
        self.prev_positions = positions.clone()

        # === Concatenate features ===
        features = torch.cat(
            [
                root_features,  # [0:3] Root height Y, velocity X, velocity Z
                ric.reshape(B, -1),
                rotations_6d.reshape(B, -1),
                local_vel.reshape(B, -1),
                feet_l,
                feet_r,
            ],
            dim=-1,
        )

        return features

    def reset(self):
        """Reset the extractor state."""
        self.prev_positions = None
        self.prev_fk_positions = None
        self.is_initialized = False


# ============================================================================
# Utility Functions
# ============================================================================


def get_feature_subset(
    features: torch.Tensor,
    subset_names: List[str],
) -> torch.Tensor:
    """
    Extract a subset of features by name.

    Args:
        features: Feature vectors (..., 271)
        subset_names: List of feature names to extract

    Returns:
        Concatenated subset of features
    """
    subsets = []
    for name in subset_names:
        if name not in FEATURE_SLICES:
            raise ValueError(
                f"Unknown feature: {name}. Available: {list(FEATURE_SLICES.keys())}"
            )
        subsets.append(features[..., FEATURE_SLICES[name]])

    return torch.cat(subsets, dim=-1)
