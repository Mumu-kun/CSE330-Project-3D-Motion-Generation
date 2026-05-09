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


# Precompute once at module level

__PARENT_INDICES, __CHILD_INDICES = zip(
    *[
        (parent, child)
        for chain in T2M_KINEMATIC_CHAIN
        for parent, child in zip(chain[:-1], chain[1:])
    ]
)
_PARENT_INDICES = torch.tensor(__PARENT_INDICES, dtype=torch.long)
_CHILD_INDICES = torch.tensor(__CHILD_INDICES, dtype=torch.long)


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


def root_features_to_root_positions(
    root_features: torch.Tensor,
    prev_root_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Convert velocity-form root features into absolute root positions.

    The velocity-form layout is `[root_height_y, root_vel_x, root_vel_z]`.
    `prev_root_pos` provides the absolute XYZ position immediately before the
    first frame in `root_features`.

    Supported shapes:
    - Single frame: `(..., 3)` with `prev_root_pos` shape `(..., 3)`
    - Sequence: `(..., T, 3)` with `prev_root_pos` shape `(..., 3)`
    """
    if root_features.size(-1) != 3:
        raise ValueError(
            "root_features must end with 3 values: [root_height_y, root_vel_x, root_vel_z]"
        )
    if prev_root_pos.size(-1) != 3:
        raise ValueError("prev_root_pos must end with 3 values: [x, y, z]")

    root_height_y = root_features[..., 0:1]
    root_vel_x = root_features[..., 1:2]
    root_vel_z = root_features[..., 2:3]

    if root_features.ndim == prev_root_pos.ndim:
        root_pos_x = prev_root_pos[..., 0:1] + root_vel_x
        root_pos_z = prev_root_pos[..., 2:3] + root_vel_z
    elif root_features.ndim == prev_root_pos.ndim + 1:
        prev_root_pos = prev_root_pos.unsqueeze(-2)
        root_pos_x = prev_root_pos[..., 0:1] + torch.cumsum(root_vel_x, dim=-2)
        root_pos_z = prev_root_pos[..., 2:3] + torch.cumsum(root_vel_z, dim=-2)
    else:
        raise ValueError(
            "Expected root_features to be either frame-shaped (..., 3) or sequence-shaped (..., T, 3) "
            "relative to prev_root_pos (..., 3)"
        )

    return torch.cat([root_pos_x, root_height_y, root_pos_z], dim=-1)


def root_positions_to_root_features(
    root_positions: torch.Tensor,
    prev_root_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Convert absolute root positions into velocity-form root features.

    The returned layout is `[root_height_y, root_vel_x, root_vel_z]`.
    The first velocity entry is measured against `prev_root_pos`.

    Supported shapes:
    - Single frame: `(..., 3)` with `prev_root_pos` shape `(..., 3)`
    - Sequence: `(..., T, 3)` with `prev_root_pos` shape `(..., 3)`
    """
    if root_positions.size(-1) != 3:
        raise ValueError("root_positions must end with 3 values: [x, y, z]")
    if prev_root_pos.size(-1) != 3:
        raise ValueError("prev_root_pos must end with 3 values: [x, y, z]")

    root_height_y = root_positions[..., 1:2]

    if root_positions.ndim == prev_root_pos.ndim:
        root_vel_x = root_positions[..., 0:1] - prev_root_pos[..., 0:1]
        root_vel_z = root_positions[..., 2:3] - prev_root_pos[..., 2:3]
    elif root_positions.ndim == prev_root_pos.ndim + 1:
        prev_root_pos = prev_root_pos.unsqueeze(-2)
        root_vel_x = torch.cat(
            [
                root_positions[..., :1, 0:1] - prev_root_pos[..., 0:1],
                root_positions[..., 1:, 0:1] - root_positions[..., :-1, 0:1],
            ],
            dim=-2,
        )
        root_vel_z = torch.cat(
            [
                root_positions[..., :1, 2:3] - prev_root_pos[..., 2:3],
                root_positions[..., 1:, 2:3] - root_positions[..., :-1, 2:3],
            ],
            dim=-2,
        )
    else:
        raise ValueError(
            "Expected root_positions to be either frame-shaped (..., 3) or sequence-shaped (..., T, 3) "
            "relative to prev_root_pos (..., 3)"
        )

    return torch.cat([root_height_y, root_vel_x, root_vel_z], dim=-1)


def get_fk_offsets(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute per-bone FK offsets scaled to match average bone lengths in input data.
    Args:
        positions: Joint positions (B, T, 22, 3)
    Returns:
        scaled_offsets: Scaled FK offsets (B, 22, 3)
    """
    B, _, J, _ = positions.shape

    parent_idx = _PARENT_INDICES.to(positions.device)
    child_idx = _CHILD_INDICES.to(positions.device)

    edge_lengths = torch.norm(
        positions[..., child_idx, :] - positions[..., parent_idx, :], dim=-1
    ).mean(
        dim=1
    )  # (B, E)

    mean_lengths = torch.zeros(B, J, dtype=positions.dtype, device=positions.device)
    mean_lengths[:, child_idx] = edge_lengths

    unit_offsets = T2M_RAW_OFFSETS.to(positions.device)
    return (unit_offsets * mean_lengths.unsqueeze(-1)).detach()  # (B, 22, 3)


def _normalize_vector(v: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize vectors along the last dimension with numerical guard."""
    inv_norm = torch.rsqrt((v * v).sum(dim=-1, keepdim=True).clamp(min=eps))
    return v * inv_norm


def _identity_quaternion_like(q: torch.Tensor) -> torch.Tensor:
    """Create an identity quaternion tensor matching q's shape/device/dtype."""
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    return identity


def _ensure_finite_tensor(
    tensor: torch.Tensor,
    stage: str,
    source_shape: torch.Size,
) -> None:
    """Raise a clear error when intermediate generated features become non-finite."""
    if torch.isfinite(tensor).all():
        return
    raise RuntimeError(
        "generated_positions_to_271d produced non-finite values "
        f"at stage '{stage}' for input batch shape {tuple(source_shape)} "
        f"and tensor shape {tuple(tensor.shape)}"
    )


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] in a differentiable way."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def root_rot6d_to_yaw(root_rot_6d: torch.Tensor) -> torch.Tensor:
    """Extract yaw from the yaw-only root 6D rotation convention."""
    if root_rot_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected root_rot_6d with trailing dimension 6, got {root_rot_6d.shape}"
        )
    rotation_matrix = cont6d_to_matrix(root_rot_6d)
    return torch.atan2(-rotation_matrix[..., 2, 0], rotation_matrix[..., 0, 0])


def yaw_to_root_rot6d(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles to the root 6D rotation convention used by this repo."""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    return torch.stack(
        [cos_yaw, zeros, -sin_yaw, zeros, ones, zeros],
        dim=-1,
    )


def yaw_to_sin_cos(yaw: torch.Tensor) -> torch.Tensor:
    """Encode an angle as [sin(angle), cos(angle)]."""
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def sin_cos_to_yaw(sin_cos: torch.Tensor) -> torch.Tensor:
    """Decode [sin(angle), cos(angle)] back to an angle."""
    if sin_cos.shape[-1] != 2:
        raise ValueError(
            f"Expected sin_cos with trailing dimension 2, got {sin_cos.shape}"
        )
    return torch.atan2(sin_cos[..., 0], sin_cos[..., 1])


def root_rot6d_to_yaw_sin_cos(root_rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert root 6D rotation to [sin(yaw), cos(yaw)]."""
    return yaw_to_sin_cos(root_rot6d_to_yaw(root_rot_6d))


def compute_root_delta_yaw_sin_cos(
    root_rot_6d: torch.Tensor,
    prev_root_rot_6d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute [sin(dyaw), cos(dyaw)] between consecutive root rotations."""
    current_yaw = root_rot6d_to_yaw(root_rot_6d)
    if prev_root_rot_6d is None:
        delta_yaw = torch.zeros_like(current_yaw)
    else:
        prev_yaw = root_rot6d_to_yaw(prev_root_rot_6d)
        delta_yaw = wrap_angle(current_yaw - prev_yaw)
    return yaw_to_sin_cos(delta_yaw)


def compute_root_delta_yaw(
    root_rot_6d: torch.Tensor,
    prev_root_rot_6d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scalar yaw velocity between consecutive root rotations."""
    current_yaw = root_rot6d_to_yaw(root_rot_6d)
    if prev_root_rot_6d is None:
        delta_yaw = torch.zeros_like(current_yaw)
    else:
        prev_yaw = root_rot6d_to_yaw(prev_root_rot_6d)
        delta_yaw = wrap_angle(current_yaw - prev_yaw)
    return delta_yaw.unsqueeze(-1)


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
    across = _normalize_vector(across)

    # Forward direction from Y-up cross product: cross([0,1,0], [x,y,z]) -> [z,0,-x]
    forward = positions_flat.new_zeros(B, 3)
    forward[:, 0] = across[:, 2]
    forward[:, 2] = -across[:, 0]
    forward = _normalize_vector(forward)

    # Target forward direction (Z-axis)
    target = positions_flat.new_zeros(B, 3)
    target[:, 2] = 1.0

    # Root rotation (from forward to target)
    root_quat = _qbetween(forward, target, assume_v0_normalized=True)

    # Initialize quaternions
    quaternions = torch.zeros(B, 22, 4, device=device, dtype=dtype)
    quaternions[:, 0] = root_quat

    # IK for each chain
    offsets = raw_offsets.unsqueeze(0).expand(B, -1, -1)  # (B, 22, 3)
    offsets_norm = _normalize_vector(offsets)

    # Conjugation sign for fast inverse of unit quaternions.
    qinv_sign = root_quat.new_tensor([1.0, -1.0, -1.0, -1.0]).view(1, 4)

    for chain in kinematic_chain:
        R = root_quat
        for i in range(len(chain) - 1):
            parent_idx = chain[i]
            child_idx = chain[i + 1]

            # Get bone direction in T-pose
            u = offsets_norm[:, child_idx]  # (B, 3)

            # Get bone direction in current pose
            v = positions_flat[:, child_idx] - positions_flat[:, parent_idx]
            v = _normalize_vector(v)

            # Rotation from u to v
            rot_u_v = _qbetween(
                u,
                v,
                assume_v0_normalized=True,
                assume_v1_normalized=True,
            )

            # Local rotation
            R_loc = qmul(R * qinv_sign, rot_u_v)

            quaternions[:, child_idx] = R_loc
            R = qmul(R, R_loc)

    return quaternions.reshape(batch_shape + (22, 4))


def _qbetween(
    v0: torch.Tensor,
    v1: torch.Tensor,
    assume_v0_normalized: bool = False,
    assume_v1_normalized: bool = False,
) -> torch.Tensor:
    """
    Compute quaternion that rotates v0 to v1.

    Args:
        v0: Source vectors (..., 3)
        v1: Target vectors (..., 3)

    Returns:
        Quaternions (..., 4)
    """
    if not assume_v0_normalized:
        v0 = _normalize_vector(v0)
    if not assume_v1_normalized:
        v1 = _normalize_vector(v1)

    # Compute rotation
    dot = (v0 * v1).sum(dim=-1, keepdim=True)

    # Handle parallel vectors
    cross = torch.cross(v0, v1, dim=-1)
    w = 1.0 + dot

    q = torch.cat([w, cross], dim=-1)
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    identity = _identity_quaternion_like(q)
    normalized_q = q / q_norm.clamp(min=1e-10)
    return torch.where(q_norm >= 1e-10, normalized_q, identity)


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

    if offsets.ndim == 2:
        offsets_expanded = offsets.unsqueeze(0).expand(B, -1, -1)  # (B, 22, 3)
    elif offsets.ndim == 3:
        offsets_expanded = offsets
    else:
        raise ValueError(
            f"Offsets must have shape (22, 3) or (B, 22, 3), got {offsets.shape}"
        )

    rot_matrices = cont6d_to_matrix(rotations_flat)  # (B, 22, 3, 3)

    # FK for each chain
    for chain in kinematic_chain:
        # Start with root rotation matrix
        matR = rot_matrices[:, 0]  # (B, 3, 3)

        for i in range(1, len(chain)):
            child_idx = chain[i]
            parent_idx = chain[i - 1]

            # Accumulate rotation
            child_rot = rot_matrices[:, child_idx]
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


def subset_271d_to_68d(
    x: torch.Tensor,
    prev_frame: Optional[torch.Tensor] = None,
    normalizer: Optional["FeatureNormalizer"] = None,
) -> torch.Tensor:
    """
    Subset 271D features to the reduced 68D predictor state.

    Reduced layout:
      [0:5]   Root: height (1) + velocity (2) + sin(dyaw), cos(dyaw) (2)
      [5:68]  Joint RIC positions: 21 non-root joints x 3D

    Height/velocity/RIC preserve the input scale. If `normalizer` is provided,
    yaw is recovered from denormalized root 6D rotations while the returned
    height/velocity/RIC slices stay in the input feature space.
    """
    if x.shape[-1] != 271:
        raise ValueError(f"Expected x to have trailing dimension 271, got {x.shape}")
    if prev_frame is not None and prev_frame.shape != x.shape:
        raise ValueError(
            f"Expected prev_frame shape {tuple(x.shape)}, got {tuple(prev_frame.shape)}"
        )

    raw_x = normalizer.denormalize(x) if normalizer is not None else x
    raw_prev = (
        normalizer.denormalize(prev_frame)
        if (normalizer is not None and prev_frame is not None)
        else prev_frame
    )

    root_height = x[..., 0:1]
    root_vel = x[..., 1:3]
    root_delta_yaw = compute_root_delta_yaw_sin_cos(
        raw_x[..., 69:75],
        None if raw_prev is None else raw_prev[..., 69:75],
    )
    joint_ric = x[..., 6:69]

    return torch.cat([root_height, root_vel, root_delta_yaw, joint_ric], dim=-1)


def frame_271d_to_263d(
    x: torch.Tensor,
    prev_frame: Optional[torch.Tensor] = None,
    normalizer: Optional["FeatureNormalizer"] = None,
) -> torch.Tensor:
    """Convert a single 271D frame to the legacy 263D evaluator layout."""
    if x.shape[-1] != 271:
        raise ValueError(f"Expected x to have trailing dimension 271, got {x.shape}")
    if prev_frame is not None and prev_frame.shape != x.shape:
        raise ValueError(
            f"Expected prev_frame shape {tuple(x.shape)}, got {tuple(prev_frame.shape)}"
        )

    raw_x = normalizer.denormalize(x) if normalizer is not None else x
    raw_prev = (
        normalizer.denormalize(prev_frame)
        if (normalizer is not None and prev_frame is not None)
        else prev_frame
    )

    root_rot_vel = compute_root_delta_yaw(
        raw_x[..., 69:75], None if raw_prev is None else raw_prev[..., 69:75]
    )
    root_height = x[..., 0:1]
    root_vel = x[..., 1:3]

    root_features = torch.cat([root_rot_vel, root_vel, root_height], dim=-1)
    joint_ric = x[..., 6:69]
    joint_rot6d = x[..., 75:201]
    joint_vel = x[..., 201:267]
    foot_contacts = x[..., 267:271]

    return torch.cat(
        [root_features, joint_ric, joint_rot6d, joint_vel, foot_contacts], dim=-1
    )


def sequence_271d_to_263d(
    x: torch.Tensor,
    normalizer: Optional["FeatureNormalizer"] = None,
) -> torch.Tensor:
    """Convert a 271D motion sequence to the legacy 263D evaluator layout."""
    if x.shape[-1] != 271:
        raise ValueError(f"Expected x to have trailing dimension 271, got {x.shape}")
    if x.ndim not in (2, 3):
        raise ValueError(
            f"Expected x to have shape (T, 271) or (B, T, 271), got {tuple(x.shape)}"
        )

    if x.ndim == 2:
        converted_frames = []
        prev_frame = None
        for frame in x:
            converted_frames.append(
                frame_271d_to_263d(frame, prev_frame=prev_frame, normalizer=normalizer)
            )
            prev_frame = frame
        return torch.stack(converted_frames, dim=0)

    converted_batches = []
    for batch_index in range(x.shape[0]):
        converted_batches.append(
            sequence_271d_to_263d(x[batch_index], normalizer=normalizer)
        )
    return torch.stack(converted_batches, dim=0)


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
        Reduced Flow Output: 68D
            - Root token output (first 5D):
                1D height + 2D velocity + sin(dyaw), cos(dyaw)
            - Joint tokens output (next 63D):
                21 joints x 3D RIC positions
        Args:
            flow_output: (..., 68) tensor of reduced flow output
        Returns:
            denormalized_flow_output: (..., 68) tensor of denormalized flow output
        """
        if flow_output.shape[-1] != 68:
            raise ValueError(
                f"Expected flow_output to have shape (..., 68), got {flow_output.shape}"
            )

        if flow_output.device != self.mean.device:
            self.mean = self.mean.to(flow_output.device)
            self.std = self.std.to(flow_output.device)

        mean_68d = torch.cat(
            [
                self.mean[0:3],
                torch.zeros(2, device=flow_output.device, dtype=flow_output.dtype),
                self.mean[6:69],
            ],
            dim=0,
        )
        std_68d = torch.cat(
            [
                self.std[0:3],
                torch.ones(2, device=flow_output.device, dtype=flow_output.dtype),
                self.std[6:69],
            ],
            dim=0,
        )
        return flow_output * std_68d + mean_68d

    def normalize_flow_output(self, flow_output: torch.Tensor) -> torch.Tensor:
        """
        Normalize flow output from raw scale to normalized scale.
        Reduced Flow Output: 68D
            - Root token output (first 5D):
                1D height + 2D velocity + sin(dyaw), cos(dyaw)
            - Joint tokens output (next 63D):
                21 joints x 3D RIC positions
        Args:
            flow_output: (..., 68) tensor of raw reduced flow output
        Returns:
            normalized_flow_output: (..., 68) tensor of normalized flow output
        """
        if flow_output.shape[-1] != 68:
            raise ValueError(
                f"Expected flow_output to have shape (..., 68), got {flow_output.shape}"
            )

        if flow_output.device != self.mean.device:
            self.mean = self.mean.to(flow_output.device)
            self.std = self.std.to(flow_output.device)

        mean_68d = torch.cat(
            [
                self.mean[0:3],
                torch.zeros(2, device=flow_output.device, dtype=flow_output.dtype),
                self.mean[6:69],
            ],
            dim=0,
        )
        std_68d = torch.cat(
            [
                self.std[0:3],
                torch.ones(2, device=flow_output.device, dtype=flow_output.dtype),
                self.std[6:69],
            ],
            dim=0,
        )
        return (flow_output - mean_68d) / std_68d

    def normalize_current_frame_features(
        self, current_frame_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize the 257D current-frame conditioning vector for the predictor.

        Layout:
            [0:5]     Root: height(1) + vel(2) + sin(yaw), cos(yaw)
            [5:257]   Joints: 21 x 12D (RIC + rot + vel)

        The copied 271D slices reuse their matching dataset statistics, while the
        yaw sin/cos slots stay in their natural [-1, 1] range.
        """
        if current_frame_features.shape[-1] != 257:
            raise ValueError(
                "Expected current_frame_features to have shape (..., 257), got "
                f"{current_frame_features.shape}"
            )

        if current_frame_features.device != self.mean.device:
            self.mean = self.mean.to(current_frame_features.device)
            self.std = self.std.to(current_frame_features.device)

        mean_257d = torch.cat(
            [
                self.mean[0:3],
                torch.zeros(
                    2,
                    device=current_frame_features.device,
                    dtype=current_frame_features.dtype,
                ),
                self.mean[6:69],
                self.mean[75:201],
                self.mean[204:267],
            ],
            dim=0,
        )
        std_257d = torch.cat(
            [
                self.std[0:3],
                torch.ones(
                    2,
                    device=current_frame_features.device,
                    dtype=current_frame_features.dtype,
                ),
                self.std[6:69],
                self.std[75:201],
                self.std[204:267],
            ],
            dim=0,
        )
        return (current_frame_features - mean_257d) / std_257d


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
    raw_offsets = config["raw_offsets"].to(device=positions.device, dtype=positions.dtype)
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
    if features.ndim == 2:
        prev_root_pos = torch.zeros(3, device=features.device, dtype=features.dtype)
    else:
        prev_root_pos = torch.zeros(
            features.shape[:-2] + (3,),
            device=features.device,
            dtype=features.dtype,
        )
    global_root_pos = root_features_to_root_positions(root_features, prev_root_pos)

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
    Reconstruct global joint positions from reduced predictor output (68D).

    Reduced predictor output format (68D):
        [0:5]   Root features: height (1D) + velocity (2D) + sin(dyaw), cos(dyaw)
        [5:68]  Joint RIC positions: 21 non-root joints x 3D = 63D

    This function is designed for autoregressive generation where:
    - Root position is updated using predicted velocity
    - Root rotation is integrated from previous yaw + predicted delta yaw
    - Joint positions are reconstructed from RIC

    Args:
        flow_output: FlowMatchingPredictor output (B, 68)
        prev_root_pos: Previous frame root position (B, 3)
        prev_root_rot_6d: Previous frame root rotation in 6D (B, 6)

    Returns:
        Global joint positions (B, 22, 3)
    """
    B = flow_output.shape[0]
    device = flow_output.device
    dtype = flow_output.dtype

    # Extract root features (5D)
    root_height = flow_output[:, 0:1]  # (B, 1)
    root_vel = flow_output[:, 1:3]  # (B, 2) - velocity X, Z
    root_delta_yaw = flow_output[:, 3:5]  # (B, 2) - sin(dyaw), cos(dyaw)

    # Extract joint RIC positions (63D -> 21 joints x 3D)
    joint_ric = flow_output[:, 5:68].reshape(B, 21, 3)  # (B, 21, 3)

    # Reconstruct root position
    # Height is absolute, X and Z are updated by velocity
    new_root_x = prev_root_pos[:, 0:1] + root_vel[:, 0:1]  # X from velocity
    new_root_y = root_height  # Y is absolute height
    new_root_z = prev_root_pos[:, 2:3] + root_vel[:, 1:2]  # Z from velocity
    new_root_pos = torch.cat([new_root_x, new_root_y, new_root_z], dim=-1)  # (B, 3)

    # Integrate root yaw from the previous frame.
    prev_root_yaw = root_rot6d_to_yaw(prev_root_rot_6d)
    delta_yaw = sin_cos_to_yaw(root_delta_yaw)
    root_yaw = wrap_angle(prev_root_yaw + delta_yaw)
    root_rot_6d = yaw_to_root_rot6d(root_yaw)

    # Convert integrated root rotation to quaternion
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
    Extract joint displacements from reduced predictor output (68D).

    This is a simpler interpretation where the output represents
    per-joint displacements that can be added to current positions.

    Reduced predictor output format (68D):
        [0:5]   Root features: height (1D) + velocity (2D) + sin(dyaw), cos(dyaw)
        [5:68]  Joint RIC positions: 21 non-root joints x 3D = 63D

    Args:
        flow_output: FlowMatchingPredictor output (B, 68)

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
    joint_disps = flow_output[:, 5:68].reshape(B, 21, 3)  # (B, 21, 3)

    # Combine: root displacement at index 0, then 21 joint displacements
    # Note: For joint 0 (root), we use the root displacement
    # For joints 1-21, we use the RIC values as displacements
    displacements = torch.cat(
        [root_disp.unsqueeze(1), joint_disps], dim=1
    )  # (B, 22, 3)

    return displacements


def extract_prev_frame_features(
    frame: torch.Tensor,
    normalizer: Optional["FeatureNormalizer"] = None,
    normalize_output: bool = False,
) -> torch.Tensor:
    """
    Extract the 257D causal conditioning features from a 271D frame.

    257D Output Format:
        [0:5]     Root: height(1) + vel(2) + sin(yaw), cos(yaw)
        [5:257]   Joints: 21 x 12D (RIC + rot + vel) = 252D

    Args:
        frame: (B, 271) single frame. When `normalizer` is provided, this is
            expected to already be in normalized 271D feature space.
        normalizer: Optional canonical feature normalizer. When provided, root yaw
            is recovered from the denormalized root 6D rotation while the other
            slices preserve the requested predictor-input scale.
        normalize_output: If True, normalize the copied predictor-conditioning
            slices before returning them. The derived yaw sin/cos slots remain
            in their natural scale.

    Returns:
        features: (B, 257)
    """
    if frame.ndim != 2 or frame.shape[-1] != 271:
        raise ValueError(f"Expected frame shape (B, 271), got {tuple(frame.shape)}")

    raw_frame = normalizer.denormalize(frame) if normalizer is not None else frame
    source_frame = raw_frame if normalize_output else frame

    # Root features (5D): height + velocity + yaw sin/cos
    root_height = source_frame[:, 0:1]
    root_vel = source_frame[:, 1:3]
    root_yaw = root_rot6d_to_yaw_sin_cos(raw_frame[:, 69:75])
    root_features = torch.cat([root_height, root_vel, root_yaw], dim=-1)  # (B, 5)

    # Joint features: 21 joints x 12D each (RIC + rotation + velocity)
    # RIC: [6:69] = 63D for 21 joints
    # Rotations: [75:201] = 126D for 21 joints
    # Velocities: [204:267] = 63D for 21 joints
    joint_ric = source_frame[:, 6:69]  # (B, 63)
    joint_rot = source_frame[:, 75:201]  # (B, 126)
    joint_vel = source_frame[:, 204:267]  # (B, 63)

    joint_features = torch.cat([joint_ric, joint_rot, joint_vel], dim=-1)  # (B, 252)

    current_frame_features = torch.cat([root_features, joint_features], dim=-1)
    if normalize_output and normalizer is not None:
        current_frame_features = normalizer.normalize_current_frame_features(
            current_frame_features
        )

    return current_frame_features  # (B, 257)


def flow_output_to_271d(
    flow_output: torch.Tensor,  # (B, 68)
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
    root_delta_yaw = flow_output[:, 3:5]  # (B,2)
    joint_ric_21 = flow_output[:, 5:68].reshape(B, 21, 3)

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
    prev_root_yaw = root_rot6d_to_yaw(prev_root_rot_6d)
    delta_yaw = sin_cos_to_yaw(root_delta_yaw)
    root_yaw = wrap_angle(prev_root_yaw + delta_yaw)
    root_rot_6d = yaw_to_root_rot6d(root_yaw)
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


def generated_positions_to_271d(
    new_positions: torch.Tensor,  # (B, 22, 3) absolute global positions
    prev_positions: Optional[
        torch.Tensor
    ] = None,  # (B, 22, 3) previous global positions
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
    fk_offsets: Optional[torch.Tensor] = None,
    normalizer: Optional["FeatureNormalizer"] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute a single 271D feature frame from newly generated global joint positions.

    This is the inverse of frame-wise reconstruction and is intended for autoregressive
    generation loops where one new frame is produced at a time.

    Feature layout:
      [0:3]    root height Y (absolute), root velocity X, root velocity Z
      [3:69]   22 RIC positions
      [69:201] 22 6D rotations
      [201:267]22 local velocities (root-local)
      [267:271]foot contacts

        Previous-frame context is provided via prev_positions.

    If no previous context is given, this behaves like cold-start extraction:
    root velocities, local velocities, and foot contacts are set to zeros.

    Args:
        new_positions: (B, 22, 3) generated global joint positions for current frame.
        prev_positions: (B, 22, 3) previous absolute global positions.
        dataset_type: Dataset key (default "t2m").
        feet_thre: Foot contact threshold.
        fk_offsets: Optional (B, 22, 3) tensor of FK offsets for RIC computation.
        normalizer: Optional FeatureNormalizer applied to the output frame.
        use_fk_for_ric: If True, compute RIC from FK-consistent positions.

    Returns:
        new_frame: (B, 271) extracted feature frame (normalized if normalizer is provided).
        new_root_pos: (B, 3) absolute current root position.
        fk_positions: (B, 22, 3) FK-consistent positions (if fk_offsets provided), else None.
    """
    if new_positions.ndim != 3 or new_positions.shape[-2:] != (22, 3):
        raise ValueError(
            f"Expected new_positions shape (B, 22, 3), got {new_positions.shape}"
        )

    B = new_positions.shape[0]
    device = new_positions.device
    dtype = new_positions.dtype

    cfg = get_dataset_config(dataset_type)
    raw_offsets = cfg["raw_offsets"].to(device=device, dtype=dtype)
    kinematic_chain = cfg["kinematic_chain"]
    face_joint_indx = cfg["face_joint_indx"]
    fid_l = cfg["fid_l"]
    fid_r = cfg["fid_r"]

    # Resolve previous absolute positions.
    if prev_positions is not None:
        if prev_positions.shape != (B, 22, 3):
            raise ValueError(
                f"Expected prev_positions shape {(B, 22, 3)}, got {prev_positions.shape}"
            )
        if prev_positions.device != device or prev_positions.dtype != dtype:
            prev_positions_resolved = prev_positions.to(device=device, dtype=dtype)
        else:
            prev_positions_resolved = prev_positions
    else:
        prev_positions_resolved = None

    candidate_positions = new_positions
    _ensure_finite_tensor(
        candidate_positions, "candidate_positions", new_positions.shape
    )
    candidate_root_pos = candidate_positions[:, 0]  # (B, 3)

    candidate_quaternions = _compute_ik(
        candidate_positions, raw_offsets, kinematic_chain, face_joint_indx
    )
    _ensure_finite_tensor(
        candidate_quaternions, "candidate_quaternions", new_positions.shape
    )
    candidate_rotations_6d = quaternion_to_cont6d(candidate_quaternions)  # (B,22,6)
    _ensure_finite_tensor(
        candidate_rotations_6d, "candidate_rotations_6d", new_positions.shape
    )

    # Resolve the canonical pose first, then derive every returned feature from it.
    fk_positions = None
    if fk_offsets is not None:
        fk_positions = _forward_kinematics(
            candidate_rotations_6d, candidate_root_pos, fk_offsets, kinematic_chain
        )
        canonical_positions = fk_positions
    else:
        canonical_positions = candidate_positions
    _ensure_finite_tensor(
        canonical_positions, "canonical_positions", new_positions.shape
    )
    if fk_positions is not None:
        _ensure_finite_tensor(fk_positions, "fk_positions", new_positions.shape)

    quaternions = _compute_ik(
        canonical_positions, raw_offsets, kinematic_chain, face_joint_indx
    )
    _ensure_finite_tensor(quaternions, "canonical_quaternions", new_positions.shape)
    rotations_6d = quaternion_to_cont6d(quaternions)  # (B,22,6)
    _ensure_finite_tensor(rotations_6d, "canonical_rotations_6d", new_positions.shape)
    root_quat = quaternions[:, 0]  # (B,4)
    root_quat_expanded = root_quat.unsqueeze(1).expand(-1, 22, -1)
    new_root_pos = canonical_positions[:, 0]  # (B, 3)

    # 1) Root features: absolute Y and velocity-form X/Z.
    root_height_y = new_root_pos[:, 1:2]
    if prev_positions_resolved is None:
        root_vel_x = torch.zeros((B, 1), device=device, dtype=dtype)
        root_vel_z = torch.zeros((B, 1), device=device, dtype=dtype)
    else:
        root_vel_x = new_root_pos[:, 0:1] - prev_positions_resolved[:, 0, 0:1]
        root_vel_z = new_root_pos[:, 2:3] - prev_positions_resolved[:, 0, 2:3]
    root_features = torch.cat([root_height_y, root_vel_x, root_vel_z], dim=-1)  # (B,3)

    ric_source = canonical_positions
    ric = ric_source - ric_source[:, 0:1]
    ric = qrot(root_quat_expanded, ric)  # (B,22,3)

    # 4) Local velocities and foot contacts.
    if prev_positions_resolved is None:
        local_vel = torch.zeros((B, 22, 3), device=device, dtype=dtype)
        feet_l = torch.zeros((B, 2), device=device, dtype=dtype)
        feet_r = torch.zeros((B, 2), device=device, dtype=dtype)
    else:
        pos_delta = canonical_positions - prev_positions_resolved
        local_vel = qrot(root_quat_expanded, pos_delta)

        vel_l = canonical_positions[:, fid_l] - prev_positions_resolved[:, fid_l]
        vel_r = canonical_positions[:, fid_r] - prev_positions_resolved[:, fid_r]
        feet_l = (torch.sum(vel_l**2, dim=-1) < feet_thre).float()
        feet_r = (torch.sum(vel_r**2, dim=-1) < feet_thre).float()

    foot_contacts = torch.cat([feet_l, feet_r], dim=-1)  # (B,4)

    # 5) Final 271D assembly.
    new_frame = torch.cat(
        [
            root_features,
            ric.reshape(B, -1),
            rotations_6d.reshape(B, -1),
            local_vel.reshape(B, -1),
            foot_contacts,
        ],
        dim=-1,
    )
    _ensure_finite_tensor(new_frame, "raw_frame", new_positions.shape)

    if normalizer is not None:
        new_frame = normalizer.normalize(new_frame)
        _ensure_finite_tensor(new_frame, "normalized_frame", new_positions.shape)

    return new_frame, new_root_pos, fk_positions


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
