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
- sequence_joints_to_features(): Dataset preprocessing (ground truth joints)
- features_to_positions(): Reconstruction (features -> positions)
- extract_features_from_predicted(): Inference (predicted joints)
- IncrementalFeatureExtractor: Frame-by-frame inference
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from utils.quaternion import (
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


def subset_271d_to_72d(x: torch.Tensor) -> torch.Tensor:
    """
    Subset 271D features to 72D features for training.
    Args:
        x: (..., 271) tensor of 271D features
    Returns:
        x_72d: (..., 72) tensor of 72D features
    """
    slices = [
        slice(0, 1),  # root height
        slice(1, 3),  # root velocity
        slice(69, 75),  # root rotation 6D
        slice(6, 69),  # joint RIC positions
    ]

    x_72d_list = []

    for sl in slices:
        x_72d_list.append(x[..., sl])

    x_72d = torch.cat(x_72d_list, dim=-1)
    return x_72d


def _subset_unused(x: torch.Tensor) -> torch.Tensor:
    """
    [DEPRECATED] This function is no longer used.
    Kept for reference purposes only.
    """
    # Root features (9D)
    root_height = x[..., 0:1]  # height_y
    root_vel = x[..., 1:3]  # vel_x, vel_z
    root_rot6d = x[..., 69:75]  # root rotation_6d
    prev_root = torch.cat([root_height, root_vel, root_rot6d], dim=-1)  # (..., 9)

    # Joint features (252D): 21 non-root joints
    joint_ric = x[..., 6:69]  # 21 x 3 = 63D (skip root at [3:6])
    joint_rot6d = x[..., 75:201]  # 21 x 6 = 126D (skip root at [69:75])
    joint_vel = x[..., 204:267]  # 21 x 3 = 63D (skip root at [201:204])
    prev_joints = torch.cat([joint_ric, joint_rot6d, joint_vel], dim=-1)  # (..., 252)

    return torch.cat([prev_root, prev_joints], dim=-1)  # (..., 261)


class FeatureNormalizer:
    def __init__(self, mean=torch.zeros(271), std=torch.ones(271)):
        self.mean = mean
        self.std = std

    @classmethod
    def load_from_files(cls, mean_path, std_path, device=torch.device("cpu")):
        mean_np = np.load(mean_path)
        std_np = np.load(std_path)
        mean = torch.from_numpy(mean_np).float().to(device)  # (271,)
        std = torch.from_numpy(std_np).float().to(device)  # (271,)
        return cls(mean, std)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to zero mean and unit variance.
        Args:
            features: (..., 271) tensor of features (271,) or (B, 271) or (B, N, 271)
        Returns:
            normalized_features: (..., 271) tensor of normalized features
        """
        if features.shape[-1] != 271:
            raise ValueError(
                f"Expected features to have shape (..., 271), got {features.shape}"
            )

        if features.device != self.mean.device:
            self.mean = self.mean.to(features.device)
            self.std = self.std.to(features.device)

        return (features - self.mean) / self.std

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Denormalize features to original scale.
        Args:
            features: (..., 271) tensor of normalized features (271,) or (B, 271) or (B, N, 271)
        Returns:
            denormalized_features: (..., 271) tensor of denormalized features
        """
        if features.shape[-1] != 271:
            raise ValueError(
                f"Expected features to have shape (..., 271), got {features.shape}"
            )

        if features.device != self.mean.device:
            self.mean = self.mean.to(features.device)
            self.std = self.std.to(features.device)

        return features * self.std + self.mean

    def denormalize_flow_output(self, flow_output: torch.Tensor) -> torch.Tensor:
        """
        Denormalize flow output to original scale.
        Flow Output: 72D
            - Root token output (first 9D):
                1D height + 2D velocity + 6D rotation
            - Joint tokens output (next 63D):
                21 joints x 3D RIC positions
        Args:
            flow_output: (..., 72) tensor of flow output (72,) or (B, 72) or (B, N, 72)
        Returns:
            denormalized_flow_output: (..., 72) tensor of denormalized flow output
        """
        if flow_output.shape[-1] != 72:
            raise ValueError(
                f"Expected flow_output to have shape (..., 72), got {flow_output.shape}"
            )

        if flow_output.device != self.mean.device:
            self.mean = self.mean.to(flow_output.device)
            self.std = self.std.to(flow_output.device)

        mean_72d = subset_271d_to_72d(self.mean)
        std_72d = subset_271d_to_72d(self.std)
        return flow_output * std_72d + mean_72d

    def normalize_flow_output(self, flow_output: torch.Tensor) -> torch.Tensor:
        """
        Normalize flow output from raw scale to normalized scale.
        Flow Output: 72D
            - Root token output (first 9D):
                1D height + 2D velocity + 6D rotation
            - Joint tokens output (next 63D):
                21 joints x 3D RIC positions
        Args:
            flow_output: (..., 72) tensor of raw flow output
        Returns:
            normalized_flow_output: (..., 72) tensor of normalized flow output
        """
        if flow_output.shape[-1] != 72:
            raise ValueError(
                f"Expected flow_output to have shape (..., 72), got {flow_output.shape}"
            )

        if flow_output.device != self.mean.device:
            self.mean = self.mean.to(flow_output.device)
            self.std = self.std.to(flow_output.device)

        mean_72d = subset_271d_to_72d(self.mean)
        std_72d = subset_271d_to_72d(self.std)
        return (flow_output - mean_72d) / std_72d


def sequence_joints_to_features(
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
            feat_b = sequence_joints_to_features(pos_b, dataset_type, feet_thre)
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
    # For input shape (N, 271), we need to sum along dim=0 (time dimension)
    # Handle both single sequence (N, 271) and batched (B, N, 271) inputs
    if features.ndim == 2:
        # Single sequence: (N, 271) -> cumsum along dim=0
        root_pos_x = torch.cumsum(root_vel_x, dim=0)
        root_pos_z = torch.cumsum(root_vel_z, dim=0)
    else:
        # Batched: (B, N, 271) -> cumsum along dim=1 (time dimension)
        root_pos_x = torch.cumsum(root_vel_x, dim=-2)
        root_pos_z = torch.cumsum(root_vel_z, dim=-2)
    global_root_pos = torch.cat([root_pos_x, root_height_y, root_pos_z], dim=-1)

    # Direct transform: RIC -> global
    # global = root_pos + rotate_inverse(RIC, root_rot)
    root_quat_expanded = root_quat.unsqueeze(-2).expand(root_quat.shape[:-1] + (22, -1))
    positions = global_root_pos.unsqueeze(-2) + qrot(qinv(root_quat_expanded), ric)

    return positions


def flow_output_to_positions(
    flow_output: torch.Tensor,
    prev_root_pos: torch.Tensor,
    prev_root_rot_6d: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct global joint positions from FlowMatchingPredictor output (72D).

    FlowMatchingPredictor output format (72D):
        [0:9]   Root features: height (1D) + velocity (2D) + rotation_6d (6D)
        [9:72]  Joint RIC positions: 21 non-root joints x 3D = 63D

    This function is designed for autoregressive generation where:
    - Root position is updated using predicted velocity
    - Root rotation comes from the prediction
    - Joint positions are reconstructed from RIC

    Args:
        flow_output: FlowMatchingPredictor output (B, 72)
        prev_root_pos: Previous frame root position (B, 3)
        prev_root_rot_6d: Previous frame root rotation in 6D (B, 6)

    Returns:
        Global joint positions (B, 22, 3)
    """
    B = flow_output.shape[0]
    device = flow_output.device
    dtype = flow_output.dtype

    # Extract root features (9D)
    root_height = flow_output[:, 0:1]  # (B, 1)
    root_vel = flow_output[:, 1:3]  # (B, 2) - velocity X, Z
    root_rot_6d = flow_output[:, 3:9]  # (B, 6)

    # Extract joint RIC positions (63D -> 21 joints x 3D)
    joint_ric = flow_output[:, 9:72].reshape(B, 21, 3)  # (B, 21, 3)

    # Reconstruct root position
    # Height is absolute, X and Z are updated by velocity
    new_root_x = prev_root_pos[:, 0:1] + root_vel[:, 0:1]  # X from velocity
    new_root_y = root_height  # Y is absolute height
    new_root_z = prev_root_pos[:, 2:3] + root_vel[:, 1:2]  # Z from velocity
    new_root_pos = torch.cat([new_root_x, new_root_y, new_root_z], dim=-1)  # (B, 3)

    # Convert root rotation to quaternion
    root_quat = cont6d_to_quaternion(root_rot_6d)  # (B, 4)

    # Reconstruct joint positions from RIC
    # RIC is in root-local coordinates, need to rotate and translate to global
    # global_joint = root_pos + rotate_inverse(RIC, root_rot)
    root_quat_expanded = root_quat.unsqueeze(1).expand(-1, 21, -1)  # (B, 21, 4)
    global_joint_offsets = qrot(qinv(root_quat_expanded), joint_ric)  # (B, 21, 3)

    # Add root position to get global joint positions
    global_joints = new_root_pos.unsqueeze(1) + global_joint_offsets  # (B, 21, 3)

    # Combine root and joints: root at index 0, then 21 joints
    positions = torch.cat(
        [new_root_pos.unsqueeze(1), global_joints], dim=1
    )  # (B, 22, 3)

    return positions


def flow_output_to_displacements(
    flow_output: torch.Tensor,
) -> torch.Tensor:
    """
    Extract joint displacements from FlowMatchingPredictor output (72D).

    This is a simpler interpretation where the output represents
    per-joint displacements that can be added to current positions.

    FlowMatchingPredictor output format (72D):
        [0:9]   Root features: height (1D) + velocity (2D) + rotation_6d (6D)
        [9:72]  Joint RIC positions: 21 non-root joints x 3D = 63D

    Args:
        flow_output: FlowMatchingPredictor output (B, 72)

    Returns:
        Joint displacements (B, 22, 3) - can be added to current positions
    """
    B = flow_output.shape[0]
    device = flow_output.device
    dtype = flow_output.dtype

    # Extract root velocity (interpret as displacement)
    root_disp_x = flow_output[:, 1:2]  # (B, 1)
    root_disp_z = flow_output[:, 2:3]  # (B, 1)
    root_disp_y = torch.zeros_like(root_disp_x)  # No Y displacement from velocity
    root_disp = torch.cat([root_disp_x, root_disp_y, root_disp_z], dim=-1)  # (B, 3)

    # Extract joint RIC as displacements
    joint_disps = flow_output[:, 9:72].reshape(B, 21, 3)  # (B, 21, 3)

    # Combine: root displacement at index 0, then 21 joint displacements
    # Note: For joint 0 (root), we use the root displacement
    # For joints 1-21, we use the RIC values as displacements
    displacements = torch.cat(
        [root_disp.unsqueeze(1), joint_disps], dim=1
    )  # (B, 22, 3)

    return displacements


def flow_output_to_271d(
    flow_output: torch.Tensor,  # (B, 72)
    prev_frame: torch.Tensor,  # (B, 271)
    prev_root_pos: torch.Tensor,  # (B, 3)  <-- tracked externally
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
):
    """
    Incrementally compute next 271D feature frame.

    No cumulative sum.
    No full sequence reconstruction.
    Fully Markov and AR-safe.

    Returns:
        new_frame: (B, 271)
        new_root_pos: (B, 3)  <-- updated absolute root position
    """

    B = flow_output.shape[0]
    device = flow_output.device
    dtype = flow_output.dtype

    # -------------------------------------------------
    # Dataset config (for foot indices and skeleton)
    # -------------------------------------------------
    config = get_dataset_config(dataset_type)
    fid_r = config["fid_r"]
    fid_l = config["fid_l"]
    raw_offsets = config["raw_offsets"].to(device=device, dtype=dtype)
    kinematic_chain = config["kinematic_chain"]
    face_joint_indx = config["face_joint_indx"]

    # -------------------------------------------------
    # 1. Extract flow output components
    # -------------------------------------------------
    root_height = flow_output[:, 0:1]  # (B,1)
    root_vel = flow_output[:, 1:3]  # (B,2)
    root_rot_6d = flow_output[:, 3:9]  # (B,6)
    joint_ric_21 = flow_output[:, 9:72].reshape(B, 21, 3)

    # -------------------------------------------------
    # 2. Update absolute root position (incremental)
    # -------------------------------------------------
    new_root_x = prev_root_pos[:, 0:1] + root_vel[:, 0:1]
    new_root_y = root_height  # absolute
    new_root_z = prev_root_pos[:, 2:3] + root_vel[:, 1:2]

    new_root_pos = torch.cat([new_root_x, new_root_y, new_root_z], dim=-1)  # (B,3)

    # -------------------------------------------------
    # 3. Build RIC block (root = zero)
    # -------------------------------------------------
    root_ric = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    ric = torch.cat([root_ric, joint_ric_21], dim=1)  # (B,22,3)

    # -------------------------------------------------
    # 4. Reconstruct previous positions (incremental)
    # -------------------------------------------------
    prev_root_rot_6d = prev_frame[:, 69:75]
    prev_root_quat = cont6d_to_quaternion(prev_root_rot_6d)

    prev_ric = prev_frame[:, 3:69].reshape(B, 22, 3)

    prev_root_quat_exp = prev_root_quat.unsqueeze(1).expand(-1, 22, -1)
    prev_positions = prev_root_pos.unsqueeze(1) + qrot(
        qinv(prev_root_quat_exp), prev_ric
    )

    # -------------------------------------------------
    # 5. Reconstruct new positions
    # -------------------------------------------------
    root_quat = cont6d_to_quaternion(root_rot_6d)
    root_quat_exp = root_quat.unsqueeze(1).expand(-1, 21, -1)

    global_offsets = qrot(qinv(root_quat_exp), joint_ric_21)  # (B,21,3)

    global_joints = new_root_pos.unsqueeze(1) + global_offsets

    new_positions = torch.cat(
        [new_root_pos.unsqueeze(1), global_joints], dim=1
    )  # (B,22,3)

    # -------------------------------------------------
    # 6. Compute rotations via IK
    # -------------------------------------------------
    quaternions = _compute_ik(
        new_positions, raw_offsets, kinematic_chain, face_joint_indx
    )  # (B, 22, 4)
    rotations_6d = quaternion_to_cont6d(quaternions)  # (B, 22, 6)

    # -------------------------------------------------
    # 7. Local velocities (root-local)
    # -------------------------------------------------
    pos_delta = new_positions - prev_positions

    root_quat_expanded = root_quat.unsqueeze(1).expand(-1, 22, -1)
    local_vel = qrot(root_quat_expanded, pos_delta)  # (B,22,3)

    # -------------------------------------------------
    # 8. Foot contact recomputation
    # -------------------------------------------------
    vel_l = new_positions[:, fid_l] - prev_positions[:, fid_l]
    vel_r = new_positions[:, fid_r] - prev_positions[:, fid_r]

    feet_l = (torch.sum(vel_l**2, dim=-1) < feet_thre).float()
    feet_r = (torch.sum(vel_r**2, dim=-1) < feet_thre).float()

    foot_contacts = torch.cat([feet_l, feet_r], dim=-1)  # (B,4)

    # -------------------------------------------------
    # 9. Root feature block (velocity form)
    # -------------------------------------------------
    root_features = torch.cat([root_height, root_vel], dim=-1)

    # -------------------------------------------------
    # 10. Assemble final 271D frame
    # -------------------------------------------------
    new_frame = torch.cat(
        [
            root_features,  # (B,3)
            ric.reshape(B, -1),  # (B,66)
            rotations_6d.reshape(B, -1),  # (B,132)
            local_vel.reshape(B, -1),  # (B,66)
            foot_contacts,  # (B,4)
        ],
        dim=-1,
    )

    new_frame = new_frame.to(device=device, dtype=dtype)
    new_root_pos = new_root_pos.to(device=device, dtype=dtype)

    return new_frame, new_root_pos


class RootPositionTracker:
    """
    Tracks absolute root position (X, Y, Z) for velocity-form representation.

    Assumes 271D layout:
    [0]   root height Y (absolute)
    [1]   root vel X
    [2]   root vel Z
    """

    def __init__(self, initial_root_pos: torch.Tensor):
        """
        Args:
            initial_root_pos: (B, 3) absolute XYZ
        """

        if initial_root_pos.dim() != 2 or initial_root_pos.size(-1) != 3:
            raise ValueError("initial_root_pos must be shape (B, 3)")

        self.root_pos = initial_root_pos  # preserves device & dtype

    @classmethod
    def from_history(cls, history_271: torch.Tensor):
        """
        Initialize from full history window.

        Args:
            history_271: (B, T, 271)
        """
        if history_271.dim() != 3 or history_271.size(-1) < 3:
            raise ValueError("history_271 must be shape (B, T, 271)")

        device = history_271.device
        dtype = history_271.dtype

        root_height = history_271[..., 0]  # (B,T)
        root_vel_x = history_271[..., 1]
        root_vel_z = history_271[..., 2]

        # Integrate velocities to recover absolute X/Z
        root_pos_x = torch.cumsum(root_vel_x, dim=1)
        root_pos_z = torch.cumsum(root_vel_z, dim=1)

        # Take last frame absolute position
        final_x = root_pos_x[:, -1]
        final_z = root_pos_z[:, -1]
        final_y = root_height[:, -1]

        initial_root_pos = torch.stack([final_x, final_y, final_z], dim=-1).to(
            device=device, dtype=dtype
        )

        return cls(initial_root_pos)

    def get(self):
        return self.root_pos

    def set(self, new_root_pos: torch.Tensor):
        self.root_pos = new_root_pos

    def update(self, new_frame_271: torch.Tensor):
        """
        Update state using newly generated frame.

        Args:
            new_frame_271: (B, 271)
        """
        if new_frame_271.device != self.root_pos.device:
            raise RuntimeError(
                "Device mismatch in RootPositionTracker.update(): "
                f"{new_frame_271.device} vs {self.root_pos.device}"
            )

        root_height = new_frame_271[:, 0:1]
        root_vel = new_frame_271[:, 1:3]

        new_x = self.root_pos[:, 0:1] + root_vel[:, 0:1]
        new_z = self.root_pos[:, 2:3] + root_vel[:, 1:2]
        new_y = root_height

        self.root_pos = torch.cat([new_x, new_y, new_z], dim=-1)


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

    def process_flow_output(
        self,
        flow_output: torch.Tensor,
        prev_frame_271d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process 72D flow output and extract 271D features.

        This is a convenience method that combines flow_output_to_positions()
        with process_frame() for autoregressive generation.

        Args:
            flow_output: (B, 72) FlowMatchingPredictor output
            prev_frame_271d: (B, 271) Previous frame features

        Returns:
            new_frame_271d: (B, 271) New frame features
            new_positions: (B, 22, 3) New global joint positions
        """
        # Extract previous root info from 271D features
        # Root position needs to be reconstructed from velocity form
        prev_root_height = prev_frame_271d[:, 0:1]  # Y height (absolute)
        # For X and Z, we need cumulative position tracking
        # Use prev_positions if available, otherwise start from origin
        if self.prev_positions is not None:
            prev_root_pos = self.prev_positions[:, 0].clone()  # (B, 3)
        else:
            # Initialize from prev_frame_271d
            prev_root_pos = torch.zeros(
                flow_output.shape[0],
                3,
                device=flow_output.device,
                dtype=flow_output.dtype,
            )
            prev_root_pos[:, 1] = prev_root_height.squeeze(-1)  # Y height

        # Extract previous root rotation (6D)
        prev_root_rot_6d = prev_frame_271d[:, 69:75]  # (B, 6)

        # Convert 72D → positions
        new_positions = flow_output_to_positions(
            flow_output, prev_root_pos, prev_root_rot_6d
        )

        # Convert positions → 271D features
        new_frame_271d = self.process_frame(new_positions)

        return new_frame_271d, new_positions
