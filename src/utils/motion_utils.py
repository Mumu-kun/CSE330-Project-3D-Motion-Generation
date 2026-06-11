"""
Motion Processing and Feature Conversion Utilities for Human Motion Animation Generation.

271D Feature Format (Pure PyTorch):
- [0:3]   Root height Y, Root velocity X, Root velocity Z
- [3:69]  22 RIC positions (66D)
- [69:201] 22 6D rotations (132D)
- [201:267] 22 local velocities (66D)
- [267:271] 4D foot contacts

Total: 271D per frame

Naming convention:
  x271  = 271-dimensional feature vector (normalized)
  x68   = 68-dimensional reduced predictor state (normalized)
  x263  = 263-dimensional legacy evaluator format

All function inputs/outputs are normalized unless otherwise noted.
All functions accept a FeatureNormalizer and denormalize internally as needed.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.quaternion import (
    cont6d_to_matrix,
    cont6d_to_quaternion,
    qinv,
    qmul,
    qrot,
    quaternion_to_cont6d,
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

# Precompute parent/child index pairs for FK edge-length computation
_PARENT_INDICES, _CHILD_INDICES = zip(
    *[(parent, child) for chain in T2M_KINEMATIC_CHAIN for parent, child in zip(chain[:-1], chain[1:])]
)
_PARENT_INDICES = torch.tensor(_PARENT_INDICES, dtype=torch.long)
_CHILD_INDICES = torch.tensor(_CHILD_INDICES, dtype=torch.long)

DATASET_CONFIGS = {
    "t2m": {
        "name": "HumanML3D",
        "num_joints": 22,
        "feature_dim": 271,
        "raw_offsets": T2M_RAW_OFFSETS,
        "kinematic_chain": T2M_KINEMATIC_CHAIN,
        "face_joint_indx": [2, 1, 17, 16],
        "fid_r": [8, 11],
        "fid_l": [7, 10],
    },
}


def get_dataset_config(dataset_type: str = "t2m") -> Dict[str, Any]:
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_type]


# ============================================================================
# Feature Layout — single source of truth for all slice indices
# ============================================================================


class Features:
    """Feature layout constants for the 271D format."""

    # 271D slices
    ROOT = slice(0, 3)  # height Y, vel X, vel Z
    RIC = slice(3, 69)  # 22 * 3
    ROT6D = slice(69, 201)  # 22 * 6
    VEL = slice(201, 267)  # 22 * 3
    CONTACTS = slice(267, 271)  # 4

    # Sub-slices within the above
    ROOT_Y = slice(0, 1)
    ROOT_VX = slice(1, 2)
    ROOT_VZ = slice(2, 3)
    ROOT_ROT6D = slice(69, 75)  # root joint's 6D rotation
    JOINT_RIC = slice(6, 69)  # 21 non-root joints * 3
    JOINT_ROT6D = slice(75, 201)  # 21 non-root joints * 6
    JOINT_VEL = slice(204, 267)  # 21 non-root joints * 3

    # 68D layout: [root_y(1) + root_vx(1) + root_vz(1) + yaw_sin_cos(2) + joint_ric(63)]
    D68_ROOT = slice(0, 5)
    D68_JOINTS = slice(5, 68)

    @staticmethod
    def joint_mask(collection: list[str | int], sl: slice) -> torch.Tensor:
        """Returns a (271,) mask for the specified collection."""
        start = sl.start
        step = (sl.stop - sl.start) / 22
        mask = torch.zeros(271, dtype=torch.bool)

        def mask_joint(idx: int):
            if idx < 0 or idx >= 22:
                raise ValueError(f"Joint index out of range: {idx}")
            idx = int(start + idx * step)
            mask[idx : idx + int(step)] = True

        for c in collection:
            if isinstance(c, int):
                mask_joint(c)
            else:
                match c:
                    case "leg_l":
                        for idx in T2M_KINEMATIC_CHAIN[0]:
                            mask_joint(idx)
                    case "leg_r":
                        for idx in T2M_KINEMATIC_CHAIN[1]:
                            mask_joint(idx)
                    case "spine":
                        for idx in T2M_KINEMATIC_CHAIN[2]:
                            mask_joint(idx)
                    case "arm_r":
                        for idx in T2M_KINEMATIC_CHAIN[3]:
                            mask_joint(idx)
                    case "arm_l":
                        for idx in T2M_KINEMATIC_CHAIN[4]:
                            mask_joint(idx)
                    case "feet":
                        for idx in DATASET_CONFIGS["t2m"]["fid_l"] + DATASET_CONFIGS["t2m"]["fid_r"]:
                            mask_joint(idx)
                    case _:
                        raise ValueError(f"Unknown collection: {c}")
        return mask


# ============================================================================
# Internal Math Helpers
# ============================================================================


def _normalize_vector(v: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return v * torch.rsqrt((v * v).sum(dim=-1, keepdim=True).clamp(min=eps))


def _identity_quaternion_like(q: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(q)
    out[..., 0] = 1.0
    return out


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] in a differentiable way."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# ============================================================================
# Yaw <-> Rotation Conversions
# ============================================================================


def root_rot6d_to_yaw(root_rot_6d: torch.Tensor) -> torch.Tensor:
    if root_rot_6d.shape[-1] != 6:
        raise ValueError(f"Expected trailing dim 6, got {root_rot_6d.shape}")
    m = cont6d_to_matrix(root_rot_6d)
    return torch.atan2(-m[..., 2, 0], m[..., 0, 0])


def yaw_to_root_rot6d(yaw: torch.Tensor) -> torch.Tensor:
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([cos_y, z, -sin_y, z, o, z], dim=-1)


def yaw_to_sin_cos(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def sin_cos_to_yaw(sin_cos: torch.Tensor) -> torch.Tensor:
    if sin_cos.shape[-1] != 2:
        raise ValueError(f"Expected trailing dim 2, got {sin_cos.shape}")
    return torch.atan2(sin_cos[..., 0], sin_cos[..., 1])


def root_rot6d_to_yaw_sin_cos(root_rot_6d: torch.Tensor) -> torch.Tensor:
    return yaw_to_sin_cos(root_rot6d_to_yaw(root_rot_6d))


def compute_root_delta_yaw_sin_cos(
    root_rot_6d: torch.Tensor,
    prev_root_rot_6d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    current_yaw = root_rot6d_to_yaw(root_rot_6d)
    if prev_root_rot_6d is None:
        delta = torch.zeros_like(current_yaw)
    else:
        delta = wrap_angle(current_yaw - root_rot6d_to_yaw(prev_root_rot_6d))
    return yaw_to_sin_cos(delta)


def compute_root_delta_yaw(
    root_rot_6d: torch.Tensor,
    prev_root_rot_6d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    current_yaw = root_rot6d_to_yaw(root_rot_6d)
    if prev_root_rot_6d is None:
        delta = torch.zeros_like(current_yaw)
    else:
        delta = wrap_angle(current_yaw - root_rot6d_to_yaw(prev_root_rot_6d))
    return delta.unsqueeze(-1)


# ============================================================================
# Root Position <-> Velocity Conversions
# ============================================================================


def root_features_to_root_positions(
    root_features: torch.Tensor,  # [y, vx, vz]
    prev_root_pos: torch.Tensor,  # [x, y, z]
) -> torch.Tensor:
    """Convert velocity-form root features to absolute XYZ positions."""
    if root_features.size(-1) != 3 or prev_root_pos.size(-1) != 3:
        raise ValueError("Both inputs must have trailing dim 3")

    vy, vx, vz = root_features[..., 0:1], root_features[..., 1:2], root_features[..., 2:3]

    if root_features.ndim == prev_root_pos.ndim:
        px = prev_root_pos[..., 0:1] + vx
        pz = prev_root_pos[..., 2:3] + vz
    elif root_features.ndim == prev_root_pos.ndim + 1:
        p = prev_root_pos.unsqueeze(-2)
        px = p[..., 0:1] + torch.cumsum(vx, dim=-2)
        pz = p[..., 2:3] + torch.cumsum(vz, dim=-2)
    else:
        raise ValueError("root_features must be (...,3) or (...,T,3) relative to prev_root_pos (...,3)")

    return torch.cat([px, vy, pz], dim=-1)


def root_positions_to_root_features(
    root_positions: torch.Tensor,  # [x, y, z]
    prev_root_pos: torch.Tensor,  # [x, y, z]
) -> torch.Tensor:
    """Convert absolute XYZ positions to velocity-form [y, vx, vz]."""
    if root_positions.size(-1) != 3 or prev_root_pos.size(-1) != 3:
        raise ValueError("Both inputs must have trailing dim 3")

    ry = root_positions[..., 1:2]

    if root_positions.ndim == prev_root_pos.ndim:
        rvx = root_positions[..., 0:1] - prev_root_pos[..., 0:1]
        rvz = root_positions[..., 2:3] - prev_root_pos[..., 2:3]
    elif root_positions.ndim == prev_root_pos.ndim + 1:
        p = prev_root_pos.unsqueeze(-2)
        rvx = torch.cat(
            [root_positions[..., :1, 0:1] - p[..., 0:1], root_positions[..., 1:, 0:1] - root_positions[..., :-1, 0:1]],
            dim=-2,
        )
        rvz = torch.cat(
            [root_positions[..., :1, 2:3] - p[..., 2:3], root_positions[..., 1:, 2:3] - root_positions[..., :-1, 2:3]],
            dim=-2,
        )
    else:
        raise ValueError("root_positions must be (...,3) or (...,T,3) relative to prev_root_pos (...,3)")

    return torch.cat([ry, rvx, rvz], dim=-1)


# ============================================================================
# IK / FK
# ============================================================================


def _qbetween(
    v0: torch.Tensor,
    v1: torch.Tensor,
    assume_v0_normalized: bool = False,
    assume_v1_normalized: bool = False,
) -> torch.Tensor:
    """Quaternion rotating v0 to v1."""
    if not assume_v0_normalized:
        v0 = _normalize_vector(v0)
    if not assume_v1_normalized:
        v1 = _normalize_vector(v1)

    dot = (v0 * v1).sum(dim=-1, keepdim=True)
    cross = torch.cross(v0, v1, dim=-1)
    w = 1.0 + dot
    q = torch.cat([w, cross], dim=-1)
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    identity = _identity_quaternion_like(q)
    return torch.where(q_norm >= 1e-10, q / q_norm.clamp(min=1e-10), identity)


def _compute_ik(
    positions: torch.Tensor,
    raw_offsets: torch.Tensor,
    kinematic_chain: List[List[int]],
    face_joint_indx: List[int],
) -> torch.Tensor:
    """Pure PyTorch IK. Input: (..., 22, 3). Output: (..., 22, 4) quaternions."""
    batch_shape = positions.shape[:-2]
    device, dtype = positions.device, positions.dtype
    positions_flat = positions.reshape(-1, 22, 3)
    B = positions_flat.shape[0]

    l_hip, r_hip, sdr_r, sdr_l = face_joint_indx
    across = positions_flat[:, r_hip] - positions_flat[:, l_hip]
    across += positions_flat[:, sdr_r] - positions_flat[:, sdr_l]
    across = _normalize_vector(across)

    forward = positions_flat.new_zeros(B, 3)
    forward[:, 0] = across[:, 2]
    forward[:, 2] = -across[:, 0]
    forward = _normalize_vector(forward)

    target = positions_flat.new_zeros(B, 3)
    target[:, 2] = 1.0

    root_quat = _qbetween(forward, target, assume_v0_normalized=True)
    quaternions = torch.zeros(B, 22, 4, device=device, dtype=dtype)
    quaternions[:, 0] = root_quat

    offsets = raw_offsets.unsqueeze(0).expand(B, -1, -1)
    offsets_norm = _normalize_vector(offsets)
    qinv_sign = root_quat.new_tensor([1.0, -1.0, -1.0, -1.0]).view(1, 4)

    for chain in kinematic_chain:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offsets_norm[:, chain[i + 1]]
            v = _normalize_vector(positions_flat[:, chain[i + 1]] - positions_flat[:, chain[i]])
            rot = _qbetween(u, v, assume_v0_normalized=True, assume_v1_normalized=True)
            R_loc = qmul(R * qinv_sign, rot)
            quaternions[:, chain[i + 1]] = R_loc
            R = qmul(R, R_loc)

    return quaternions.reshape(batch_shape + (22, 4))


def _forward_kinematics(
    rotations_6d: torch.Tensor,
    root_pos: torch.Tensor,
    offsets: torch.Tensor,
    kinematic_chain: List[List[int]],
) -> torch.Tensor:
    """Pure PyTorch FK. Input rotations: (..., 22, 6), root: (..., 3). Output: (..., 22, 3)."""
    batch_shape = rotations_6d.shape[:-2]
    device, dtype = rotations_6d.device, rotations_6d.dtype
    rot_flat = rotations_6d.reshape(-1, 22, 6)
    root_flat = root_pos.reshape(-1, 3)
    B = rot_flat.shape[0]

    positions = torch.zeros(B, 22, 3, device=device, dtype=dtype)
    positions[:, 0] = root_flat

    if offsets.ndim == 2:
        off = offsets.unsqueeze(0).expand(B, -1, -1)
    elif offsets.ndim == 3:
        off = offsets
    else:
        raise ValueError(f"Offsets must be (22,3) or (B,22,3), got {offsets.shape}")

    rot_mats = cont6d_to_matrix(rot_flat)

    for chain in kinematic_chain:
        matR = rot_mats[:, 0]
        for i in range(1, len(chain)):
            child, parent = chain[i], chain[i - 1]
            matR = torch.bmm(matR, rot_mats[:, child])
            positions[:, child] = torch.bmm(matR, off[:, child].unsqueeze(-1)).squeeze(-1) + positions[:, parent]

    return positions.reshape(batch_shape + (22, 3))


# ============================================================================
# Shared Feature Extraction Helpers
# ============================================================================


def _compute_foot_contacts(
    new_pos: torch.Tensor,
    prev_pos: torch.Tensor,
    fid_l: List[int],
    fid_r: List[int],
    threshold: float,
) -> torch.Tensor:
    """Compute 4D foot contact flags from consecutive position frames. Input: (B, 22, 3)."""
    vel_l = new_pos[:, fid_l] - prev_pos[:, fid_l]
    vel_r = new_pos[:, fid_r] - prev_pos[:, fid_r]
    feet_l = (vel_l.pow(2).sum(dim=-1) < threshold).float()
    feet_r = (vel_r.pow(2).sum(dim=-1) < threshold).float()
    return torch.cat([feet_l, feet_r], dim=-1)


def _compute_ric(positions: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Root-centered, root-rotated positions. Input: (B, 22, 3) positions, (B, 4) root quat. Output: (B, 22, 3)."""
    ric = positions - positions[:, 0:1]
    return qrot(root_quat.unsqueeze(1).expand(-1, 22, -1), ric)


def _assemble_271d(
    root_features: torch.Tensor,  # (B, 3)
    ric: torch.Tensor,  # (B, 22, 3)
    rotations_6d: torch.Tensor,  # (B, 22, 6)
    local_vel: torch.Tensor,  # (B, 22, 3)
    foot_contacts: torch.Tensor,  # (B, 4)
) -> torch.Tensor:
    """Assemble a (B, 271) feature frame from components."""
    B = root_features.shape[0]
    return torch.cat(
        [
            root_features,
            ric.reshape(B, -1),
            rotations_6d.reshape(B, -1),
            local_vel.reshape(B, -1),
            foot_contacts,
        ],
        dim=-1,
    )


# ============================================================================
# Feature Normalizer
# ============================================================================


class FeatureNormalizer:
    """Normalizes/denormalizes 271D feature vectors and derived sub-spaces."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        # Precompute derived stats for 68D
        self._mean_68d = torch.cat([mean[0:3], torch.zeros(2), mean[6:69]], dim=0)
        self._std_68d = torch.cat([std[0:3], torch.ones(2), std[6:69]], dim=0)
        # Precompute derived stats for 263D
        # 263D layout: [
        #   root_rotvel(1) + root_vel(2) + root_y(1) + joint_ric(63) + joint_rot(126) + joint_vel(63) + contacts(4)
        # ]
        self._mean_263 = torch.cat(
            [
                torch.zeros(1),
                mean[Features.ROOT_VX],
                mean[Features.ROOT_VZ],
                mean[Features.JOINT_RIC],
                mean[Features.JOINT_ROT6D],
                mean[Features.JOINT_VEL],
                torch.zeros(4),
            ],
            dim=0,
        )
        self._std_263 = torch.cat(
            [
                torch.ones(1),
                std[Features.ROOT_VX],
                std[Features.ROOT_VZ],
                std[Features.JOINT_RIC],
                std[Features.JOINT_ROT6D],
                std[Features.JOINT_VEL],
                torch.ones(4),
            ],
            dim=0,
        )

    @classmethod
    def load_from_files(cls, mean_path: str, std_path: str, device=torch.device("cpu")):
        mean = torch.from_numpy(np.load(mean_path)).float().to(device)
        std = torch.from_numpy(np.load(std_path)).float().to(device)
        return cls(mean, std)

    def _sync(self, x: torch.Tensor):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            self._mean_68d = self._mean_68d.to(x.device)
            self._std_68d = self._std_68d.to(x.device)
            self._mean_263 = self._mean_263.to(x.device)
            self._std_263 = self._std_263.to(x.device)

    def normalize(self, x271: torch.Tensor) -> torch.Tensor:
        """Normalize a 271D feature vector."""
        assert x271.shape[-1] == 271
        self._sync(x271)
        return (x271 - self.mean) / self.std

    def denormalize(self, x271: torch.Tensor) -> torch.Tensor:
        """Denormalize a 271D feature vector to original scale."""
        assert x271.shape[-1] == 271
        self._sync(x271)
        return x271 * self.std + self.mean

    def normalize_x68(self, x68: torch.Tensor) -> torch.Tensor:
        """Normalize a 68D predictor state."""
        assert x68.shape[-1] == 68
        self._sync(x68)
        return (x68 - self._mean_68d) / self._std_68d

    def denormalize_x68(self, x68: torch.Tensor) -> torch.Tensor:
        """Denormalize a 68D predictor state to original scale."""
        assert x68.shape[-1] == 68
        self._sync(x68)
        return x68 * self._std_68d + self._mean_68d

    def normalize_x263(self, x263: torch.Tensor) -> torch.Tensor:
        """Normalize a 263D evaluator feature vector."""
        assert x263.shape[-1] == 263
        self._sync(x263)
        return (x263 - self._mean_263) / self._std_263

    def denormalize_x263(self, x263: torch.Tensor) -> torch.Tensor:
        """Denormalize a 263D evaluator feature vector to original scale."""
        assert x263.shape[-1] == 263
        self._sync(x263)
        return x263 * self._std_263 + self._mean_263


# ============================================================================
# Core Conversion: Positions <-> x271
# ============================================================================


def positions_to_x271(
    new_positions: torch.Tensor,  # (B, 22, 3) absolute global positions
    prev_positions: torch.Tensor,  # (B, 22, 3) previous frame (None for cold start)
    normalizer: FeatureNormalizer,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert global joint positions to a normalized 271D feature frame.

    Returns:
        x271: (B, 271) normalized
        root_pos: (B, 3) absolute current root position
    """
    B = new_positions.shape[0]
    device, dtype = new_positions.device, new_positions.dtype
    cfg = get_dataset_config(dataset_type)
    raw_offsets = cfg["raw_offsets"].to(device=device, dtype=dtype)

    quaternions = _compute_ik(new_positions, raw_offsets, cfg["kinematic_chain"], cfg["face_joint_indx"])
    rotations_6d = quaternion_to_cont6d(quaternions)
    root_quat = quaternions[:, 0]

    root_pos = new_positions[:, 0]
    if prev_positions is None:
        root_vel_x = torch.zeros(B, 1, device=device, dtype=dtype)
        root_vel_z = torch.zeros(B, 1, device=device, dtype=dtype)
    else:
        root_vel_x = root_pos[:, 0:1] - prev_positions[:, 0, 0:1]
        root_vel_z = root_pos[:, 2:3] - prev_positions[:, 0, 2:3]
    root_features = torch.cat([root_pos[:, 1:2], root_vel_x, root_vel_z], dim=-1)

    ric = _compute_ric(new_positions, root_quat)

    if prev_positions is None:
        local_vel = torch.zeros(B, 22, 3, device=device, dtype=dtype)
        foot_contacts = torch.zeros(B, 4, device=device, dtype=dtype)
    else:
        pos_delta = new_positions - prev_positions
        local_vel = qrot(root_quat.unsqueeze(1).expand(-1, 22, -1), pos_delta)
        foot_contacts = _compute_foot_contacts(new_positions, prev_positions, cfg["fid_l"], cfg["fid_r"], feet_thre)

    frame = _assemble_271d(root_features, ric, rotations_6d, local_vel, foot_contacts)
    return normalizer.normalize(frame), root_pos


def x271_to_positions(
    x271: torch.Tensor,  # (B, 271) or (B, T, 271) normalized
    normalizer: FeatureNormalizer,
    prev_positions: Optional[torch.Tensor] = None,  # (B, 22, 3),
    dataset_type: str = "t2m",
) -> torch.Tensor:
    """
    Reconstruct global joint positions from normalized 271D features.
    For sequences, provide prev_root_pos for velocity integration.
    For single frames without prev_root_pos, root X/Z are assumed zero.
    """
    single_frame = x271.ndim == 2
    if single_frame:
        x271 = x271.unsqueeze(1)

    B, T = x271.shape[0], x271.shape[1]
    device, dtype = x271.device, x271.dtype

    prev_root_pos = prev_positions[:, 0] if prev_positions is not None else None

    raw = normalizer.denormalize(x271)

    root_features = raw[..., Features.ROOT]
    ric = raw[..., Features.RIC].reshape(B, T, 22, 3)
    rotations_6d = raw[..., Features.ROT6D].reshape(B, T, 22, 6)

    root_quat = cont6d_to_quaternion(rotations_6d[:, :, 0])

    if prev_root_pos is None:
        prev_root_pos = torch.zeros(B, 3, device=device, dtype=dtype)
    global_root = root_features_to_root_positions(root_features, prev_root_pos)

    root_q_exp = root_quat.unsqueeze(-2).expand(B, T, 22, -1)
    positions = global_root.unsqueeze(-2) + qrot(qinv(root_q_exp), ric)

    return positions.squeeze(1) if single_frame else positions


# ============================================================================
# x271 <-> x68 Conversion
# ============================================================================


def x271_to_x68(
    x271: torch.Tensor,  # (B, 271) normalized
    normalizer: FeatureNormalizer,
    prev_x271: Optional[torch.Tensor] = None,  # (B, 271) normalized
) -> torch.Tensor:
    """
    Convert normalized 271D -> reduced 68D predictor state.

    68D layout: [root_y(1) + root_vx(1) + root_vz(1) + yaw_sin_cos(2) + joint_ric_21(63)]
    Output is in the same normalized space as the predictor expects.
    """
    assert x271.shape[-1] == 271
    raw = normalizer.denormalize(x271)
    raw_prev = normalizer.denormalize(prev_x271) if prev_x271 is not None else None

    root_y = raw[..., 0:1]
    root_v = raw[..., 1:3]
    delta_yaw = compute_root_delta_yaw_sin_cos(
        raw[..., Features.ROOT_ROT6D],
        None if raw_prev is None else raw_prev[..., Features.ROOT_ROT6D],
    )
    joint_ric = raw[..., Features.JOINT_RIC]

    x68 = torch.cat([root_y, root_v, delta_yaw, joint_ric], dim=-1)
    x68 = normalizer.normalize_x68(x68)

    return x68


def x68_to_positions(
    x68: torch.Tensor,  # (B, 68) normalized predictor output
    prev_positions: torch.Tensor,  # (B, 22, 3) previous frame's absolute joint positions
    prev_x271: torch.Tensor,  # (B, 271) normalized — previous frame for delta-yaw computation
    normalizer: FeatureNormalizer,
) -> torch.Tensor:
    """
    Reconstruct global joint positions from denormalized 68D predictor output.

    68D layout: [root_y(1) + root_vx(1) + root_vz(1) + yaw_sin_cos(2) + joint_ric_21(63)]
    """
    B = x68.shape[0]

    x68 = normalizer.denormalize_x68(x68)
    prev_x271 = normalizer.denormalize(prev_x271)

    prev_root_pos = prev_positions[:, 0]  # (B, 3)
    prev_root_rot_6d = prev_x271[:, Features.ROOT_ROT6D]  # (B, 6)

    root_y = x68[:, 0:1]
    root_vx = x68[:, 1:2]
    root_vz = x68[:, 2:3]
    delta_yaw = sin_cos_to_yaw(x68[:, 3:5])
    joint_ric_21 = x68[:, 5:68].reshape(B, 21, 3)

    new_root_x = prev_root_pos[:, 0:1] + root_vx
    new_root_z = prev_root_pos[:, 2:3] + root_vz
    new_root_pos = torch.cat([new_root_x, root_y, new_root_z], dim=-1)

    prev_yaw = root_rot6d_to_yaw(prev_root_rot_6d)
    root_quat = cont6d_to_quaternion(yaw_to_root_rot6d(wrap_angle(prev_yaw + delta_yaw)))

    global_offsets = qrot(qinv(root_quat.unsqueeze(1).expand(-1, 21, -1)), joint_ric_21)
    new_joint_pos = new_root_pos.unsqueeze(1) + global_offsets
    return torch.cat([new_root_pos.unsqueeze(1), new_joint_pos], dim=1)


# ============================================================================
# x271 <-> x263 Conversion (legacy evaluator format)
# ============================================================================


def x271_to_x263(
    x271: torch.Tensor,  # (B, 271) normalized
    normalizer: FeatureNormalizer,
    prev_x271: Optional[torch.Tensor] = None,  # (B, 271) normalized — previous frame for delta-yaw computation
) -> torch.Tensor:
    """
    Convert normalized 271D -> legacy 263D evaluator layout.

    263D layout: [
        root_rotvel(1) + root_vel(2) + root_y(1) + joint_ric(63) + joint_rot(126) + joint_vel(63) + contacts(4)
    ]
    """
    assert x271.shape[-1] == 271
    raw = normalizer.denormalize(x271)
    raw_prev = normalizer.denormalize(prev_x271) if prev_x271 is not None else None

    root_rot_vel = compute_root_delta_yaw(
        raw[..., Features.ROOT_ROT6D],
        None if raw_prev is None else raw_prev[..., Features.ROOT_ROT6D],
    )
    root_block = torch.cat([root_rot_vel, raw[..., 1:3], raw[..., 0:1]], dim=-1)

    x263 = torch.cat(
        [
            root_block,  # 4D
            raw[..., Features.JOINT_RIC],  # 63D
            raw[..., Features.JOINT_ROT6D],  # 126D
            raw[..., Features.JOINT_VEL],  # 63D
            raw[..., Features.CONTACTS],  # 4D
        ],
        dim=-1,
    )

    return normalizer.normalize_x263(x263)


def x271_seq_to_x263(
    x271_seq: torch.Tensor,  # (T, 271) or (B, T, 271) normalized
    normalizer: FeatureNormalizer,
) -> torch.Tensor:
    """Convert a normalized 271D motion sequence to the legacy 263D evaluator layout."""
    if x271_seq.ndim == 2:
        frames, prev = [], None
        for frame in x271_seq:
            frames.append(x271_to_x263(frame.unsqueeze(0), normalizer, prev))
            prev = frame.unsqueeze(0)
        return torch.cat(frames, dim=0)

    return torch.stack([x271_seq_to_x263(x271_seq[i], normalizer) for i in range(x271_seq.shape[0])])


# ============================================================================
# x68 <-> x271 and x68 <-> x263 Cross-Conversions
# ============================================================================


def x68_to_x271(
    x68: torch.Tensor,  # (B, 68) normalized predictor output
    normalizer: FeatureNormalizer,
    prev_positions: torch.Tensor,  # (B, 22, 3) previous frame's absolute joint positions
    prev_x271: torch.Tensor,  # (B, 271) normalized — previous frame for delta-yaw computation
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> torch.Tensor:
    """
    Convert denormalized 68D predictor output -> normalized 271D feature frame.

    Simple path: x68 -> positions (via x68_to_positions), then positions + prev_positions -> x271
    (via positions_to_x271). No expensive round-trip through x271_to_positions.

    68D layout: [root_y(1) + root_vx(1) + root_vz(1) + yaw_sin_cos(2) + joint_ric_21(63)]
    """

    positions = x68_to_positions(x68, prev_positions, prev_x271, normalizer)
    x271, _ = positions_to_x271(positions, prev_positions, normalizer, dataset_type, feet_thre)
    return x271


def x68_to_x263(
    x68: torch.Tensor,  # (B, 68) normalized predictor output
    prev_positions: torch.Tensor,  # (B, 22, 3) previous frame's absolute joint positions
    prev_x271: torch.Tensor,  # (B, 271) normalized — previous frame for delta-yaw computation
    normalizer: FeatureNormalizer,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> torch.Tensor:
    """
    Convert denormalized 68D predictor output -> 263D evaluator layout.

    Route: x68 -> positions -> x271 -> x263
    """
    x271 = x68_to_x271(x68, normalizer, prev_positions, prev_x271, dataset_type, feet_thre)
    return x271_to_x263(x271, normalizer, prev_x271)
