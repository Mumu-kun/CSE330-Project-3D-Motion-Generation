"""
Motion Processing and Feature Conversion Utilities for Human Motion Animation Generation.

271D Feature Format (Pure PyTorch):
- [0:3]   Root height Y, Root velocity X, Root velocity Z
- [3:69]  22 RIC positions (66D)
- [69:201] 22 6D rotations (132D)
- [201:267] 22 local velocities (66D)
- [267:271] 4D foot contacts

Total: 271D per frame

68D Feature Format (Predictor state) - Delta/Velocity representation:
- [0:1]   root_y (absolute height)
- [1:2]   root_vx (velocity X)
- [2:3]   root_vz (velocity Z)
- [3:4]   delta_yaw_sin
- [4:5]   delta_yaw_cos
- [5:68]  21 joint velocities * 3 (63)

Total: 68D

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

    ROOT = slice(0, 3)
    RIC = slice(3, 69)
    ROT6D = slice(69, 201)
    VEL = slice(201, 267)
    CONTACTS = slice(267, 271)

    ROOT_Y = slice(0, 1)
    ROOT_VX = slice(1, 2)
    ROOT_VZ = slice(2, 3)
    ROOT_ROT6D = slice(69, 75)
    JOINT_RIC = slice(6, 69)
    JOINT_ROT6D = slice(75, 201)
    JOINT_VEL = slice(204, 267)

    D68_ROOT_Y = slice(0, 1)
    D68_ROOT_VX = slice(1, 2)
    D68_ROOT_VZ = slice(2, 3)
    D68_YAW_SIN = slice(3, 4)
    D68_YAW_COS = slice(4, 5)
    D68_JOINTS_VEL = slice(5, 68)
    D68_YAW_SINCOS = slice(3, 5)

    D72_ROOT_Y = slice(0, 1)
    D72_ROOT_VX = slice(1, 2)
    D72_ROOT_VZ = slice(2, 3)
    D72_YAW_SIN = slice(3, 4)
    D72_YAW_COS = slice(4, 5)
    D72_JOINTS_RIC = slice(5, 68)
    D72_CONTACTS = slice(68, 72)
    D72_YAW_SINCOS = slice(3, 5)

    D75_ROOT_Y = D72_ROOT_Y
    D75_ROOT_VX = D72_ROOT_VX
    D75_ROOT_VZ = D72_ROOT_VZ
    D75_YAW_SIN = D72_YAW_SIN
    D75_YAW_COS = D72_YAW_COS
    D75_JOINTS_RIC = D72_JOINTS_RIC
    D75_CONTACTS = D72_CONTACTS
    D75_YAW_SINCOS = D72_YAW_SINCOS

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


class Features263:
    """Extended feature layout for the 263D legacy evaluator format."""

    # 263D slices
    ROOT_ROTVEL = slice(0, 1)  # root angular velocity around Y (for legacy compatibility)
    ROOT_VEL = slice(1, 3)  # root velocity XZ (for legacy compatibility)
    RIC = slice(4, 67)  # 21 non-root joints * 3
    ROT6D = slice(67, 193)  # 21 non-root joints * 6
    VEL = slice(193, 259)  # 22 non-root joints * 3
    CONTACTS = slice(259, 263)  # 4 foot contacts

    JOINT_VEL = slice(196, 259)  # 21 non-root joints * 3

    @staticmethod
    def calc_mean_std(
        data: torch.Tensor,  # (B, N, 263)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean and std with optional masking of specified joint collections."""
        root_rotvel = data[:, :, Features263.ROOT_ROTVEL]
        root_vel = data[:, :, Features263.ROOT_VEL]
        ric = data[:, :, Features263.RIC]
        rot6d = data[:, :, Features263.ROT6D]
        vel = data[:, :, Features263.VEL]

        mean = torch.cat(
            [
                root_rotvel.mean(dim=(0, 1)),
                root_vel.mean(dim=(0, 1)),
                ric.mean(dim=(0, 1)),
                rot6d.mean(dim=(0, 1)),
                vel.mean(dim=(0, 1)),
                torch.zeros(4),  # contacts are binary, so we set mean to 0 for stability
            ],
            dim=0,
        )

        std = torch.cat(
            [
                root_rotvel.std(dim=(0, 1)),
                root_vel.std(dim=(0, 1)),
                ric.std(dim=(0, 1)),
                rot6d.std(dim=(0, 1)),
                vel.std(dim=(0, 1)),
                torch.ones(4),  # contacts are binary, so we set std to 1 for stability
            ],
            dim=0,
        )

        return mean, std


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
        self._mean_68d = torch.cat(
            [
                mean[Features.ROOT_Y],
                mean[Features.ROOT_VX],
                mean[Features.ROOT_VZ],
                torch.zeros(2),
                mean[Features.JOINT_VEL],
            ],
            dim=0,
        )
        self._std_68d = torch.cat(
            [
                std[Features.ROOT_Y],
                std[Features.ROOT_VX],
                std[Features.ROOT_VZ],
                torch.ones(2),
                std[Features.JOINT_VEL],
            ],
            dim=0,
        )
        self._mean_72d = torch.cat(
            [
                mean[Features.ROOT_Y],
                mean[Features.ROOT_VX],
                mean[Features.ROOT_VZ],
                torch.zeros(2),
                mean[Features.JOINT_RIC],
                torch.zeros(4),
            ],
            dim=0,
        )
        self._std_72d = torch.cat(
            [
                std[Features.ROOT_Y],
                std[Features.ROOT_VX],
                std[Features.ROOT_VZ],
                torch.ones(2),
                std[Features.JOINT_RIC],
                torch.ones(4),
            ],
            dim=0,
        )
        # Precompute derived stats for 263D
        # 263D layout: [
        #   root_rotvel(1) + root_vel(2) + root_y(1) + joint_ric(63) + joint_rot(126) + joint_vel(66) + contacts(4)
        # ]
        self._mean_263 = torch.cat(
            [
                torch.zeros(1),
                mean[Features.ROOT_VX],
                mean[Features.ROOT_VZ],
                mean[Features.ROOT_Y],
                mean[Features.JOINT_RIC],
                mean[Features.JOINT_ROT6D],
                mean[Features.VEL],
                torch.zeros(4),
            ],
            dim=0,
        )
        self._std_263 = torch.cat(
            [
                torch.ones(1),
                std[Features.ROOT_VX],
                std[Features.ROOT_VZ],
                std[Features.ROOT_Y],
                std[Features.JOINT_RIC],
                std[Features.JOINT_ROT6D],
                std[Features.VEL],
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
            self._mean_72d = self._mean_72d.to(x.device)
            self._std_72d = self._std_72d.to(x.device)
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

    def normalize_x72(self, x72: torch.Tensor) -> torch.Tensor:
        """Normalize a 72D (~75D) predictor/decoder state."""
        assert x72.shape[-1] in (72, 75, 68)
        self._sync(x72)
        mean = self._mean_72d if x72.shape[-1] == 72 else (self._mean_68d if x72.shape[-1] == 68 else self._mean_72d)
        std = self._std_72d if x72.shape[-1] == 72 else (self._std_68d if x72.shape[-1] == 68 else self._std_72d)
        return (x72 - mean) / std

    def denormalize_x72(self, x72: torch.Tensor) -> torch.Tensor:
        """Denormalize a 72D (~75D) predictor/decoder state to original scale."""
        assert x72.shape[-1] in (72, 75, 68)
        self._sync(x72)
        mean = self._mean_72d if x72.shape[-1] == 72 else (self._mean_68d if x72.shape[-1] == 68 else self._mean_72d)
        std = self._std_72d if x72.shape[-1] == 72 else (self._std_68d if x72.shape[-1] == 68 else self._std_72d)
        return x72 * std + mean

    normalize_x75 = normalize_x72
    denormalize_x75 = denormalize_x72

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
# x271 <-> x72 / x75 / x68 Conversion
# ============================================================================


def x271_to_x72(
    x271: torch.Tensor,
    normalizer: FeatureNormalizer,
    prev_positions: Optional[torch.Tensor] = None,
    prev_x271: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert normalized 271D -> reduced 72D (~75D) predictor/decoder state.

    72D layout: [root_y(1) + root_vx(1) + root_vz(1) + delta_yaw_sin(1) + delta_yaw_cos(1) + joint_ric(63) + contacts(4)]
    Output is in the same normalized space as the predictor/decoder expects.
    """
    assert x271.shape[-1] == 271
    raw = normalizer.denormalize(x271)
    raw_prev = normalizer.denormalize(prev_x271) if prev_x271 is not None else None

    root_y = raw[..., Features.ROOT_Y]
    root_vx = raw[..., Features.ROOT_VX]
    root_vz = raw[..., Features.ROOT_VZ]

    delta_yaw_sin_cos = compute_root_delta_yaw_sin_cos(
        raw[..., Features.ROOT_ROT6D],
        None if raw_prev is None else raw_prev[..., Features.ROOT_ROT6D],
    )
    joint_ric = raw[..., Features.JOINT_RIC]
    contacts = raw[..., Features.CONTACTS]

    x72 = torch.cat([root_y, root_vx, root_vz, delta_yaw_sin_cos, joint_ric, contacts], dim=-1)
    x72 = normalizer.normalize_x72(x72)

    return x72


x271_to_x75 = x271_to_x72
x271_to_x68 = x271_to_x72


def x72_to_positions(
    x72: torch.Tensor,
    normalizer: FeatureNormalizer,
    prev_x271: torch.Tensor,
    prev_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reconstruct global joint positions from 72D (~75D) decoder output.

    72D layout: [root_y(1) + root_vx(1) + root_vz(1) + delta_yaw_sin(1) + delta_yaw_cos(1) + joint_ric(63) + contacts(4)]
    prev_x271: normalized 271D feature vector for previous frame
    """
    B = x72.shape[0]

    x72_denorm = normalizer.denormalize_x72(x72)
    prev_x271_raw = normalizer.denormalize(prev_x271)

    root_y = x72_denorm[:, Features.D72_ROOT_Y]
    root_vx = x72_denorm[:, Features.D72_ROOT_VX]
    root_vz = x72_denorm[:, Features.D72_ROOT_VZ]

    prev_root_pos = prev_positions[:, 0] if prev_positions is not None else torch.zeros(B, 3, device=x72.device, dtype=x72.dtype)
    root_x = prev_root_pos[:, 0:1] + root_vx
    root_z = prev_root_pos[:, 2:3] + root_vz
    root_pos = torch.cat([root_x, root_y, root_z], dim=-1)

    delta_yaw_sin_cos = x72_denorm[:, Features.D72_YAW_SINCOS]
    prev_root_rot_6d = prev_x271_raw[:, Features.ROOT_ROT6D]
    prev_yaw = root_rot6d_to_yaw(prev_root_rot_6d)
    delta_yaw = sin_cos_to_yaw(delta_yaw_sin_cos)
    yaw = wrap_angle(prev_yaw + delta_yaw)
    root_quat = yaw_to_root_rot6d(yaw)
    root_quat = cont6d_to_quaternion(root_quat)

    # Direct RIC positions for 21 non-root joints (NO frame-to-frame velocity accumulation!)
    joint_ric = x72_denorm[:, Features.D72_JOINTS_RIC].reshape(B, 21, 3)

    global_joints = qrot(qinv(root_quat.unsqueeze(1).expand(-1, 21, -1)), joint_ric)
    new_joint_pos = root_pos.unsqueeze(1) + global_joints
    return torch.cat([root_pos.unsqueeze(1), new_joint_pos], dim=1)


x75_to_positions = x72_to_positions
x68_to_positions = x72_to_positions


def x72_to_x271(
    x72: torch.Tensor,
    normalizer: FeatureNormalizer,
    prev_x271: torch.Tensor,
    prev_positions: Optional[torch.Tensor] = None,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> torch.Tensor:
    """
    Convert denormalized 72D predictor output -> normalized 271D feature frame.
    """
    positions = x72_to_positions(x72, normalizer, prev_x271, prev_positions)
    x271, _ = positions_to_x271(positions, prev_positions, normalizer, dataset_type, feet_thre)
    return x271


x75_to_x271 = x72_to_x271
x68_to_x271 = x72_to_x271


def x72_to_x263(
    x72: torch.Tensor,
    normalizer: FeatureNormalizer,
    prev_x271: torch.Tensor,
    prev_positions: Optional[torch.Tensor] = None,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> torch.Tensor:
    """
    Convert 72D predictor output -> 263D evaluator layout.
    """
    x271 = x72_to_x271(x72, normalizer, prev_x271, prev_positions, dataset_type, feet_thre)
    prev_x271_raw = normalizer.denormalize(prev_x271)
    return x271_to_x263(x271, normalizer, prev_x271_raw[:, Features.ROOT_ROT6D])


x75_to_x263 = x72_to_x263
x68_to_x263 = x72_to_x263
