# =============================================================================
# utils_kaggle.py  —  single-file combined module for Kaggle / Colab
# =============================================================================

from __future__ import annotations

import os
import sys
import copy
import re
import ast
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union, cast, reveal_type

import torch
import torch.nn as nn
import torch.nn.functional as F
# GradScaler is dynamically selected in trainer classes
from torch.utils.data import DataLoader, Dataset
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from transformers import CLIPTokenizer, CLIPTextModel, ACT2FN
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    ACT2FN = None

try:
    from ignite.engine import Engine, Events, State
    from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan, global_step_from_engine
    from ignite.handlers.tqdm_logger import ProgressBar
    from ignite.metrics import RunningAverage
except ImportError:
    IGNITE_AVAILABLE = False
    Engine = Events = State = None
    Checkpoint = DiskSaver = TerminateOnNan = None
    ProgressBar = None
    RunningAverage = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    plt = None
    FuncAnimation = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum


# ========== quaternion.py ==========

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Streamlined version - unused functions removed

import torch
import numpy as np

_EPS4 = np.finfo(float).eps * 4.0

_FLOAT_EPS = np.finfo(np.float64).eps


def _safe_normalize_quaternion(
    quaternions: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Normalize quaternions and fall back to identity when the norm collapses."""
    if quaternions.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternions with trailing dimension 4, got {quaternions.shape}"
        )

    norms = torch.norm(quaternions, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternions)
    identity[..., 0] = 1.0
    normalized = quaternions / norms.clamp(min=eps)
    return torch.where(norms >= eps, normalized, identity)


# PyTorch-backed implementations
def qinv(q):
    """Invert quaternion(s) q."""
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    qw, qx, qy, qz = torch.unbind(q, dim=-1)
    rw, rx, ry, rz = torch.unbind(r, dim=-1)

    w = rw * qw - rx * qx - ry * qy - rz * qz
    x = rw * qx + rx * qw - ry * qz + rz * qy
    y = rw * qy + rx * qz + ry * qw - rz * qx
    z = rw * qz - rx * qy + ry * qx + rz * qw

    return torch.stack((w, x, y, z), dim=-1)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = _safe_normalize_quaternion(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-8)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_cont6d(quaternions):
    """Convert quaternions to 6D rotation representation."""
    quaternions = _safe_normalize_quaternion(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-8)

    c0_x = 1 - two_s * (j * j + k * k)
    c0_y = two_s * (i * j + k * r)
    c0_z = two_s * (i * k - j * r)

    c1_x = two_s * (i * j - k * r)
    c1_y = 1 - two_s * (i * i + k * k)
    c1_z = two_s * (j * k + i * r)

    return torch.stack((c0_x, c0_y, c0_z, c1_x, c1_y, c1_z), dim=-1)


def cont6d_to_matrix(cont6d):
    """Convert 6D rotation representation to rotation matrix."""
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    # Add epsilon to prevent division by zero
    eps = 1e-8
    x_norm = torch.norm(x_raw, dim=-1, keepdim=True).clamp(min=eps)
    x = x_raw / x_norm
    z = torch.cross(x, y_raw, dim=-1)
    z_norm = torch.norm(z, dim=-1, keepdim=True).clamp(min=eps)
    z = z / z_norm

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def matrix_to_quaternion(rotation_matrix):
    """
    Convert rotation matrices to quaternions.

    Args:
        rotation_matrix: Rotation matrices (..., 3, 3)

    Returns:
        Quaternions (..., 4) with real part first
    """
    batch_shape = rotation_matrix.shape[:-2]
    rotation_matrix = rotation_matrix.reshape(-1, 3, 3)

    batch_size = rotation_matrix.shape[0]
    q = torch.zeros(
        batch_size, 4, device=rotation_matrix.device, dtype=rotation_matrix.dtype
    )

    trace = (
        rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
    )

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (rotation_matrix[mask1, 2, 1] - rotation_matrix[mask1, 1, 2]) / s1
    q[mask1, 2] = (rotation_matrix[mask1, 0, 2] - rotation_matrix[mask1, 2, 0]) / s1
    q[mask1, 3] = (rotation_matrix[mask1, 1, 0] - rotation_matrix[mask1, 0, 1]) / s1

    # Case 2: (R00 > R11) and (R00 > R22)
    mask2 = (
        (~mask1)
        & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 1, 1])
        & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 2, 2])
    )
    s2 = (
        torch.sqrt(
            1.0
            + rotation_matrix[mask2, 0, 0]
            - rotation_matrix[mask2, 1, 1]
            - rotation_matrix[mask2, 2, 2]
        )
        * 2
    )
    q[mask2, 0] = (rotation_matrix[mask2, 2, 1] - rotation_matrix[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (rotation_matrix[mask2, 0, 1] + rotation_matrix[mask2, 1, 0]) / s2
    q[mask2, 3] = (rotation_matrix[mask2, 0, 2] + rotation_matrix[mask2, 2, 0]) / s2

    # Case 3: R11 > R22
    mask3 = (~mask1) & (~mask2) & (rotation_matrix[:, 1, 1] > rotation_matrix[:, 2, 2])
    s3 = (
        torch.sqrt(
            1.0
            + rotation_matrix[mask3, 1, 1]
            - rotation_matrix[mask3, 0, 0]
            - rotation_matrix[mask3, 2, 2]
        )
        * 2
    )
    q[mask3, 0] = (rotation_matrix[mask3, 0, 2] - rotation_matrix[mask3, 2, 0]) / s3
    q[mask3, 1] = (rotation_matrix[mask3, 0, 1] + rotation_matrix[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (rotation_matrix[mask3, 1, 2] + rotation_matrix[mask3, 2, 1]) / s3

    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = (
        torch.sqrt(
            1.0
            + rotation_matrix[mask4, 2, 2]
            - rotation_matrix[mask4, 0, 0]
            - rotation_matrix[mask4, 1, 1]
        )
        * 2
    )
    q[mask4, 0] = (rotation_matrix[mask4, 1, 0] - rotation_matrix[mask4, 0, 1]) / s4
    q[mask4, 1] = (rotation_matrix[mask4, 0, 2] + rotation_matrix[mask4, 2, 0]) / s4
    q[mask4, 2] = (rotation_matrix[mask4, 1, 2] + rotation_matrix[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    # Normalize
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-10)

    return q.reshape(batch_shape + (4,))


def cont6d_to_quaternion(cont6d):
    """
    Convert 6D rotation representation to quaternion.

    Args:
        cont6d: 6D rotation (..., 6)

    Returns:
        Quaternions (..., 4) with real part first
    """
    mat = cont6d_to_matrix(cont6d)
    return matrix_to_quaternion(mat)


# ========== motion_utils.py ==========

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

# [internal import removed]  from utils.quaternion import (

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


def enforce_rigid_bone_lengths(
    positions: torch.Tensor,
    ref_joints: torch.Tensor,
    kinematic_chain: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """
    Enforce constant, rigid bone lengths along the kinematic tree matching the reference skeleton.
    Preserves 100% of predicted joint directions and angles while strictly eliminating bone stretching/shrinking.

    Args:
        positions: (22, 3), (T, 22, 3), or (B, T, 22, 3) predicted joint positions.
        ref_joints: (22, 3) or (B, 22, 3) reference skeleton (e.g. from seed frame).
        kinematic_chain: Optional list of kinematic chains. Defaults to T2M_KINEMATIC_CHAIN.

    Returns:
        positions_rigid: Tensor of the same shape with exact reference bone lengths.
    """
    if kinematic_chain is None:
        kinematic_chain = T2M_KINEMATIC_CHAIN

    orig_shape = positions.shape
    if positions.ndim == 2:  # (22, 3)
        pos = positions.unsqueeze(0).unsqueeze(0)
    elif positions.ndim == 3:
        if positions.shape[1] == 22 and positions.shape[2] == 3:  # (T, 22, 3) or (B, 22, 3)
            pos = positions.unsqueeze(0)  # (1, N, 22, 3)
        else:
            raise ValueError(f"Unexpected positions shape: {positions.shape}")
    elif positions.ndim == 4:  # (B, T, 22, 3)
        pos = positions
    else:
        raise ValueError(f"Positions must be 2D, 3D, or 4D, got {positions.shape}")

    if ref_joints.ndim == 2:
        ref = ref_joints.unsqueeze(0)
    elif ref_joints.ndim == 3:
        ref = ref_joints if ref_joints.shape[0] == pos.shape[0] else ref_joints[0:1]
    else:
        ref = ref_joints.reshape(-1, 22, 3)[0:1]

    out = pos.clone()
    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            p_idx, c_idx = chain[i], chain[i + 1]
            ref_len = (ref[:, c_idx] - ref[:, p_idx]).norm(dim=-1, keepdim=True).unsqueeze(1)
            ref_len = torch.clamp(ref_len, min=1e-4)
            delta = out[:, :, c_idx] - out[:, :, p_idx]
            dist = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            direction = delta / dist
            out[:, :, c_idx] = out[:, :, p_idx] + direction * ref_len

    return out.reshape(orig_shape)


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
    positions = torch.cat([root_pos.unsqueeze(1), new_joint_pos], dim=1)

    if prev_positions is not None:
        positions = enforce_rigid_bone_lengths(positions, prev_positions)

    return positions


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


# ========== config.py ==========

"""
Configuration for Human Motion Animation Generation Pipeline.

Uses the custom 271D feature format:
  Input:  271D feature vectors from motion_utils.py
  Output: Joint positions (nframe, 22, 3) → BVH files

Feature Layout (271D):
  [0:3]     Root height Y, Root velocity X, Root velocity Z
  [3:69]    22 RIC positions (22 * 3)
  [69:201]  22 6D rotations (22 * 6)
  [201:267] 22 local velocities (22 * 3)
  [267:271] Foot contacts (4D)

Note: Root X,Z are stored as velocities for autoregressive stability.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class FlowMatchingPredictorConfig:
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    track_dimensionality: int = 3
    global_cond_dim: int = 512  # CLIP embedding: 512D
    head_dim: Optional[int] = None
    use_self_attn_rope: bool = True
    text_prior_bias: float = 0.8

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class JepaPredictorConfig:
    hidden_size: int = 192
    intermediate_size: int = 512
    num_hidden_layers: int = 2


@dataclass
class LatentDecoderConfig:
    hidden_size: int = 512
    intermediate_size: int = 2 * 512
    dropout: float = 0.0
    num_layers: int = 4


@dataclass
class MotionHistoryEncoderConfig:
    frame_feature_dim: int = 271
    text_embedding_dim: int = 512
    hidden_size: int = 512
    intermediate_size: int = 2 * 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    dropout: float = 0.1
    joint_count: int = 22
    num_registers: int = 2
    jp_config: JepaPredictorConfig = field(default_factory=JepaPredictorConfig)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads, "
                f"got {self.hidden_size} and {self.num_attention_heads}."
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"Per-head dimension must be even for RoPE, got head_dim={head_dim} "
                f"(hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads})."
            )
        self.intermediate_size = max(int(self.intermediate_size), self.hidden_size)


@dataclass
class PretrainConfig:
    """Configuration for pretraining the motion predictor."""

    effective_batch_size: int = 400
    batch_size: int = 200
    learning_rate: float = 0.5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 30.0
    ema_decay: float = 0.999
    lr_warmup_epochs: int = 5
    lr_scheduler: str = "cosine"
    jepa_ctx_weight: float = 0.2
    cfg_dropout: float = 0.1

    schedules: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {
            "mask_num_spans": [
                (0, 2),
                (0.5, 4),
                (1, 5),
            ],
        }
    )

    # --- Masking ---
    mask_min_span: int = 5
    mask_max_span: int = 20


@dataclass
class Config:
    """Configuration for the motion generation pipeline."""

    # --- Device ---
    device: Any = "cuda"
    seed: int = 42

    # --- Paths ---
    dataset_path: Path = Path("./dataset/humanml3d-subset")
    output_path: Path = Path("./output")
    checkpoint_dir: Path = Path("./checkpoints")
    checkpoint_interval: int = 50

    # --- Motion format (271D) ---
    motion_dim: int = 271
    num_joints: int = 22
    joint_dim: int = 3
    max_motion_length: int = 200
    fps: int = 20
    # Number of history frames fed to the encoder as context for Phase 3.
    # Must be > 0; sequences shorter than this are zero-padded at the front.
    history_length: int = 40

    # --- Sub-configs ---
    encoder_config: MotionHistoryEncoderConfig = field(default_factory=MotionHistoryEncoderConfig)
    predictor_config: FlowMatchingPredictorConfig = field(default_factory=FlowMatchingPredictorConfig)
    decoder_config: LatentDecoderConfig = field(default_factory=LatentDecoderConfig)
    text_embedding_dim: int = 512

    # --- PreTraining ---
    pre_conf: PretrainConfig = field(default_factory=PretrainConfig)
    effective_batch_size: int = 400
    batch_size: int = 200
    learning_rate: float = 0.5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 30.0
    ema_decay: float = 0.999
    lr_warmup_epochs: int = 5
    lr_scheduler: str = "cosine"
    jepa_ctx_weight: float = 0.2
    cfg_dropout: float = 0.1
    use_fk: bool = False

    # --- Curriculum ---
    curriculum: Optional[list[dict[str, int]]] = field(
        default_factory=lambda: [
            {"horizon": 5, "epochs": 100},
            {"horizon": 10, "epochs": 200},
            {"horizon": 20, "epochs": 300},
            {"horizon": 40, "epochs": 1000},
        ]
    )
    horizon: int = 40
    _num_epochs: int = 2000

    # --- Timestep sampling ---
    t_sampling_mode: str = "power"  # "uniform" or "power"
    t_sampling_power: float = 3.0
    t_sampling_power_warmup_fraction: float = 1.0

    # --- Rollout scheduling ---
    rollout_prob_start: float = 0.1
    rollout_prob_end: float = 0.3
    rollout_warmup_fraction: float = 0.15
    rollout_block_len_start: int = 1
    rollout_block_len_end: int = 4
    rollout_integration_steps: int = 3
    rollout_subset_fraction: float = 0.25
    rollout_loss_weight: float = 0.25
    rollout_block_len_bias_power: float = 2.0

    # --- Consistency loss ---
    use_consistency_loss: bool = True
    consistency_loss_t_threshold: float = 0.5
    consistency_loss_weight: float = 10.0

    # --- Data loading ---
    num_workers: int = 4
    pin_memory: bool = True

    # --- Inference ---
    num_inference_steps: int = 20
    inference_t_schedule_power: float = 3.0
    guidance_scale: float = 1.0

    # --- Validation ---
    val_interval: int = 5
    val_batches: int = 20
    val_use_ema: bool = True
    save_best_val: bool = True

    # --- Profiling ---
    enable_profiling: bool = False
    timing_log_interval: int = 100

    def __post_init__(self) -> None:
        # Timestep sampling
        self.t_sampling_mode = str(self.t_sampling_mode).lower()
        if self.t_sampling_mode not in {"uniform", "power"}:
            raise ValueError(f"t_sampling_mode must be 'uniform' or 'power', got {self.t_sampling_mode!r}")
        self.t_sampling_power = max(0.0, float(self.t_sampling_power))
        self.t_sampling_power_warmup_fraction = min(max(float(self.t_sampling_power_warmup_fraction), 0.0), 1.0)

        # Rollout
        self.rollout_warmup_fraction = min(max(float(self.rollout_warmup_fraction), 0.0), 1.0 - 1e-6)
        self.rollout_block_len_start = max(1, int(self.rollout_block_len_start))
        self.rollout_block_len_end = max(self.rollout_block_len_start, int(self.rollout_block_len_end))
        self.rollout_subset_fraction = min(max(float(self.rollout_subset_fraction), 0.0), 1.0)
        self.rollout_loss_weight = max(float(self.rollout_loss_weight), 0.0)
        self.rollout_block_len_bias_power = float(self.rollout_block_len_bias_power)
        if self.rollout_block_len_bias_power <= 1.0:
            raise ValueError(f"rollout_block_len_bias_power must be > 1, got {self.rollout_block_len_bias_power}")

        # Inference
        self.inference_t_schedule_power = float(self.inference_t_schedule_power)
        if self.inference_t_schedule_power <= 0.0:
            raise ValueError(f"inference_t_schedule_power must be positive, got {self.inference_t_schedule_power}")

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def get_num_epochs(self) -> int:
        """Return total training epochs, accounting for curriculum."""
        return self.curriculum[-1]["epochs"] if self.curriculum else self._num_epochs

    def to_dict(self) -> dict:
        """Export config as a serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                self._convert_paths_to_strings(v)
        return d

    def _convert_paths_to_strings(self, d: dict) -> None:
        """Recursively convert Path objects to strings in a dict."""
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                self._convert_paths_to_strings(v)

    def state_dict(self) -> dict:
        """Serialize for Ignite's Checkpoint handler."""
        return self.to_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        from dataclasses import fields
        from pathlib import Path

        from cattrs import Converter

        converter = Converter()
        converter.register_structure_hook(Path, lambda d, _: Path(d) if isinstance(d, str) else d)

        loaded = converter.structure(state_dict, Config)
        for f in fields(loaded):
            setattr(self, f.name, getattr(loaded, f.name))


# ========== text_encoder.py ==========

"""
Text encoding utility using CLIP model from Hugging Face Transformers.
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Union


class CLIPEncoder(torch.nn.Module):
    """
    Utility class to encode text captions using Microsoft's CLIP model.
    By default, uses 'openai/clip-vit-base-patch32' which produces 512D embeddings.

    For texts longer than 77 tokens, uses chunk-and-average approach to preserve
    all text content.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
    ):
        super().__init__()

        self.max_length = max_length

        print(f"Loading CLIP model '{model_name}'...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        try:
            self.model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True)
        except Exception:
            self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()

        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode a list of captions or a single caption into embeddings.

        For texts longer than max_length tokens, splits into chunks and averages
        the embeddings to preserve all text content.

        Args:
            text: A single string or a list of strings.

        Returns:
            embeddings: (B, 1, 512) tensor containing pooled CLIP embeddings.
        """
        if isinstance(text, str):
            text = [text]

        # Determine device dynamically
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = self.model(**inputs)

        # Use the pooler_output for a global representation of the sentence
        # Shape: (Batch_Size, 512)
        embeddings = outputs.pooler_output.unsqueeze(1)

        return embeddings

    @torch.no_grad()
    def encode_sequence(self, text: Union[str, List[str]], max_length: int = 77) -> torch.Tensor:
        """
        Encode text captions into a sequence of token embeddings.

        Args:
            text: A single string or a list of strings.
            max_length: Maximum sequence length (CLIP max is 77).

        Returns:
            embeddings: (B, S, 512) tensor containing token sequence embeddings.
        """
        if isinstance(text, str):
            text = [text]

        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        outputs = self.model(**inputs)

        return outputs.last_hidden_state

    @property
    def embedding_dim(self) -> int:
        """Output dimension of the CLIP text model."""
        return self.model.config.hidden_size



# ========== dataset.py ==========

"""
Dataset loading and Text2Motion dataset implementation.

Handles loading HumanML3D dataset with text-motion pairs.
"""

import random
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.motion_utils import FeatureNormalizer


class Text2MotionDataset(Dataset):
    """
    Text-to-Motion Dataset compatible with HumanML3D format.

    Loads motion features with corresponding text descriptions.
    Supports variable-length sequences with time-stamped text annotations.
    """

    def __init__(
        self,
        config: Config,
        mean: np.ndarray,
        std: np.ndarray,
        split: str = "train",
    ):
        self.config = config
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = config.max_motion_length
        self.current_horizon = config.max_motion_length
        min_motion_len = 40

        # Derive paths from config.dataset_path
        motion_dir = config.dataset_path / "new_joint_vecs"
        joints_dir = config.dataset_path / "new_joints"
        text_dir = config.dataset_path / "texts"
        split_file = config.dataset_path / f"{split}.txt"

        data_dict = {}
        id_list = []
        with open(str(split_file), "r", encoding="utf-8") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(str(motion_dir), name + ".npy"))
                joints = np.load(pjoin(str(joints_dir), name + ".npy"))

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                text_data = []
                flag = False
                with open(pjoin(str(text_dir), name + ".txt"), "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        text_dict: Dict[str, Optional[Any]] = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                n_joints = joints[int(f_tag * 20) : int(to_tag * 20)]
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "joints": n_joints,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "joints": joints,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception:
                pass

        name_length_pairs = list(zip(new_name_list, length_list))
        name_length_pairs.sort(key=lambda x: x[1])  # Sort by length

        self.name_list = [pair[0] for pair in name_length_pairs]
        self.length_arr = np.array([pair[1] for pair in name_length_pairs])
        self.data_dict = data_dict
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

        # --- Text Embedding Caching ---
        self.text_cache_path = config.dataset_path / "text_embeddings_cache.pt"
        self.text_cache: Dict[str, torch.Tensor] = {}

        if self.text_cache_path.exists():
            print(f"Loading text embedding cache from {self.text_cache_path}...")
            self.text_cache = torch.load(self.text_cache_path, weights_only=False)

        # Collect all unique captions
        all_captions = set()
        for key, data in self.data_dict.items():
            for text_item in data["text"]:
                all_captions.add(text_item["caption"])

        # Identify missing captions
        missing_captions = [cap for cap in all_captions if cap not in self.text_cache]

        if missing_captions:
            print(
                f"Computed {len(self.text_cache)}/{len(all_captions)} embeddings. Computing {len(missing_captions)} missing..."
            )

            # LAZY IMPORT: Only import when needed
            # [internal import removed]  from utils.text_encoder import CLIPEncoder

            # Initialize CLIP Encoder only if needed (to save VRAM if cached)
            clip_encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32")
            clip_encoder.to(config.device)

            batch_size = 32
            for i in tqdm(range(0, len(missing_captions), batch_size), desc="Encoding Texts"):
                batch_caps = missing_captions[i : i + batch_size]
                with torch.no_grad():
                    # (B, 1, 512) - pooled CLIP embeddings
                    embeddings = clip_encoder(batch_caps).cpu()

                for cap, emb in zip(batch_caps, embeddings):
                    self.text_cache[cap] = emb

            # Save updated cache
            print(f"Saving updated cache to {self.text_cache_path}...")
            torch.save(self.text_cache, self.text_cache_path)

            # Cleanup to free VRAM
            del clip_encoder
            torch.cuda.empty_cache()
        else:
            print("All text embeddings are cached.")

    def get_normalizer(self) -> FeatureNormalizer:
        """Get a FeatureNormalizer instance for normalizing/denormalizing features."""
        return FeatureNormalizer(mean=self.mean.clone(), std=self.std.clone())

    def __len__(self):
        return len(self.data_dict) - self.pointer

    """
    FINAL CORRECT __getitem__ implementation
    This is the ONLY version that works - replace everything else
    """

    def __getitem__(self, item) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, str, list[str]]:
        """
        Returns a single sample from the dataset.
        GUARANTEES: All returned tensors have shape (max_motion_length, features)

        NOTE: Returns RAW (unnormalized) features. Normalization should be done
        externally using FeatureNormalizer before passing to models.

        history_motion: (history_length, 271) RAW features from the frames
        immediately preceding the target window.  Zero-padded at the front when
        fewer than history_length frames are available before the target start.
        Used exclusively by Phase 3 (FlowMatchingPredictor) as the encoder
        conditioning context; ignored by pretrain and decoder trainers.
        """
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        # Get raw data from dict
        motion = data["motion"]  # numpy array (T, 271)
        joints = data["joints"]  # numpy array (T, 22, 3)
        original_length = data["length"]  # int, original number of frames
        text_list = data["text"]  # list of text dicts

        # Choose random text
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        tokens: list[str] = text_data["tokens"]

        # ===== CONVERT TO TENSORS =====
        motion = torch.from_numpy(motion.copy()).float()  # (T, 271) - RAW features
        joints = torch.from_numpy(joints.copy()).float()  # (T, 22, 3)

        # ===== NO NORMALIZATION HERE =====
        # Normalization is handled externally via FeatureNormalizer
        # motion = (motion - self.mean) / self.std  # REMOVED

        # ===== PAD OR TRUNCATE TO MAX_MOTION_LENGTH =====
        target_len = self.current_horizon
        current_len = original_length
        history_length = self.config.history_length

        if current_len < target_len:
            # ── Short sequence: pad target, no history available ──────────────
            pad_size = target_len - current_len
            motion = torch.cat(
                [
                    motion,
                    torch.zeros(
                        pad_size,
                        motion.shape[1],
                        dtype=motion.dtype,
                        device=motion.device,
                    ),
                ],
                dim=0,
            )
            joints = torch.cat(
                [
                    joints,
                    torch.zeros(
                        pad_size,
                        joints.shape[1],
                        joints.shape[2],
                        dtype=joints.dtype,
                        device=joints.device,
                    ),
                ],
                dim=0,
            )
            # No frames precede the target window — history is all zeros
            history_motion = torch.zeros(history_length, motion.shape[1], dtype=motion.dtype)
            history_valid_length = 0

        elif current_len > target_len:
            start_idx = random.randint(0, current_len - self.current_horizon)

            # ── History: frames strictly before start_idx ─────────────────────
            hist_end   = start_idx
            hist_start = max(0, start_idx - history_length)
            hist_slice = motion[hist_start:hist_end]            # (≤ history_length, 271)
            history_valid_length = hist_slice.shape[0]

            if hist_slice.shape[0] < history_length:
                # Zero-pad at the front so history is always (history_length, 271)
                pad_h = torch.zeros(
                    history_length - hist_slice.shape[0],
                    motion.shape[1],
                    dtype=motion.dtype,
                )
                history_motion = torch.cat([pad_h, hist_slice], dim=0)
            else:
                history_motion = hist_slice

            # ── Target window ─────────────────────────────────────────────────
            motion = motion[start_idx : start_idx + self.current_horizon]
            joints = joints[start_idx : start_idx + self.current_horizon]

        else:
            # current_len == target_len: no room for history
            history_motion = torch.zeros(history_length, motion.shape[1], dtype=motion.dtype)
            history_valid_length = 0

        valid_length = min(current_len, target_len)

        # ===== GET TEXT EMBEDDING =====
        text_embedding = self.text_cache[caption]
        if isinstance(text_embedding, np.ndarray):
            text_embedding = torch.from_numpy(text_embedding).float()
        else:
            text_embedding = text_embedding.float()

        # Normalize to strict shape: (1, 512)
        if text_embedding.ndim == 1:
            if text_embedding.shape[0] != CLIP_EMBED_DIM:
                raise ValueError(
                    f"Invalid 1D text embedding shape {tuple(text_embedding.shape)} for caption '{caption}'. "
                    f"Expected ({CLIP_EMBED_DIM},)."
                )
            text_embedding = text_embedding.unsqueeze(0)
        elif text_embedding.ndim == 2:
            if text_embedding.shape == (1, CLIP_EMBED_DIM):
                pass
            elif text_embedding.shape == (CLIP_MAX_SEQ_LEN, CLIP_EMBED_DIM):
                raise ValueError(
                    "Detected legacy CLIP sequence embedding shape (77, 512) in text cache. "
                    "Regenerate text_embeddings_cache.pt using pooled CLIP outputs (1, 512)."
                )
            else:
                raise ValueError(
                    f"Invalid 2D text embedding shape {tuple(text_embedding.shape)} for caption '{caption}'. "
                    f"Expected (1, {CLIP_EMBED_DIM})."
                )
        else:
            raise ValueError(
                f"Invalid text embedding rank {text_embedding.ndim} for caption '{caption}'. "
                "Expected rank 2 with shape (1, 512)."
            )

        # text_embedding shape: (1, 512) - pooled CLIP embedding
        # motion shape:         (target_len, 271)   - target window; used for z1 (decoder + predictor target)
        # history_motion shape: (history_length, 271) - frames strictly preceding motion; used for track_features
        # valid_length is the number of real frames before zero-padding, capped at target_len.
        sample_id = self.name_list[idx]

        return caption, motion, joints, history_motion, valid_length, text_embedding, sample_id, tokens, history_valid_length

    def reset_min_len(self, length: int | None = None):
        if length is None:
            self.pointer = 0
            return
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)

    def set_horizon(self, horizon: int | None = None):
        """
        Set the horizon for the dataset.
        If horizon is None, use the default horizon.
        If horizon is an integer, set the horizon to that value.
        Dynamic horizon update with filtering.
        """
        if horizon is None:
            self.current_horizon = self.max_motion_length
            self.reset_min_len()  # Reset pointer to start
            return

        assert 1 <= horizon <= self.max_motion_length
        self.current_horizon = horizon
        self.reset_min_len(horizon)  # Auto-filter


# CLIP constants
CLIP_MAX_SEQ_LEN = 77
CLIP_EMBED_DIM = 512


def text2motion_collate_fn(
    batch: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, str, list[str], int]],
) -> Dict[str, Any]:
    """
    Collate function for Text2MotionDataset.
    Expects each sample to be a tuple:
      - caption:        str
      - motion:         (target_len, 271) torch.Tensor - target window, RAW 271D features
      - joints:         (target_len, J, 3) torch.Tensor
      - history_motion: (history_length, 271) torch.Tensor - frames preceding motion,
                        zero-padded at front when not enough history; RAW features.
                        Used only by FlowMatchingTrainer (Phase 3).
      - length:         int - valid frames in motion before padding, capped at target_len
      - text_embedding: (1, 512) torch.Tensor - pooled CLIP embeddings
      - sample_id:      str
      - tokens:         list[str]
      - history_valid_length: int - exact count of real non-zero history frames
    """
    # Lists of items
    captions         = [b[0] for b in batch]
    motions_list     = [b[1] for b in batch]
    joints_list      = [b[2] for b in batch]
    history_list     = [b[3] for b in batch]
    lengths          = [b[4] for b in batch]
    text_embs_list   = [b[5] for b in batch]
    sample_ids       = [b[6] for b in batch]
    tokens_list      = [b[7] for b in batch]
    hist_lengths     = [b[8] for b in batch]

    # Stack tensors directly
    motion_batch       = torch.stack(motions_list,   dim=0)  # (B, T, 271)
    joints_batch       = torch.stack(joints_list,    dim=0)  # (B, T, J, 3)
    history_batch      = torch.stack(history_list,   dim=0)  # (B, history_length, 271)
    length_batch       = torch.tensor(lengths, dtype=torch.long)  # (B,)
    text_emb_batch     = torch.stack(text_embs_list, dim=0)  # (B, 1, 512)
    hist_length_batch  = torch.tensor(hist_lengths, dtype=torch.long)  # (B,)

    return {
        "captions":             captions,
        "sample_ids":           sample_ids,
        "motion":               motion_batch,        # target window
        "history_motion":       history_batch,      # context window preceding target
        "joints":               joints_batch,
        "lengths":              length_batch,
        "text_clip":            text_emb_batch,
        "tokens":               tokens_list,
        "history_valid_length": hist_length_batch,
    }


def create_dataloader(
    config: Config,
    split: str = "train",
    shuffle: bool = True,
) -> Tuple[DataLoader, FeatureNormalizer]:
    """
    Create DataLoader for Text2MotionDataset.

    Automatically loads mean and std from Mean.npy and Std.npy in dataset_path.

    Args:
        config: Config object with dataset configuration
        split: Dataset split ("train", "val", "test"). Default: "train"
        shuffle: Whether to shuffle data

    Returns:
        Tuple of (DataLoader instance, FeatureNormalizer instance)
        - DataLoader provides RAW (unnormalized) features
        - FeatureNormalizer should be used to normalize features before passing to models
    """
    mean_path = config.dataset_path / "Mean.npy"
    std_path = config.dataset_path / "Std.npy"

    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Mean.npy and/or Std.npy not found in {config.dataset_path}. "
            "Please ensure Mean.npy and Std.npy exist in the dataset directory."
        )

    mean = np.load(mean_path)
    std = np.load(std_path)

    dataset_obj = Text2MotionDataset(config, mean, std, split)

    # Create FeatureNormalizer for external normalization
    normalizer = FeatureNormalizer(
        mean=torch.from_numpy(mean).float(),
        std=torch.from_numpy(std).float(),
    )

    dataloader = DataLoader(
        dataset_obj,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=text2motion_collate_fn,
    )

    return dataloader, normalizer


def load_sample(dataset_path: Path, file_id: str) -> Dict[str, Optional[Any]]:
    """
    Load a single motion sample with features, joints, and text.

    Args:
        dataset_path: Path to HumanML3D dataset root
        file_id: Motion sample ID (without extension)

    Returns:
        Dictionary with keys:
        - 'features': Feature vectors (nframe, 271) from new_joint_vecs
        - 'joints': Joint positions (nframe, 22, 3) from new_joints
        - 'text': Text description from texts folder
        - 'file_id': Sample ID
    """
    features_path = dataset_path / "new_joint_vecs" / f"{file_id}.npy"
    joints_path = dataset_path / "new_joints" / f"{file_id}.npy"
    text_path = dataset_path / "texts" / f"{file_id}.txt"

    data: Dict[str, Optional[Any]] = {"file_id": file_id}

    # Load feature vectors
    if features_path.exists():
        data["features"] = np.load(features_path)
    else:
        print(f"Warning: Features not found for {file_id}")
        data["features"] = None

    # Load joint positions
    if joints_path.exists():
        data["joints"] = np.load(joints_path)
    else:
        print(f"Warning: Joints not found for {file_id}")
        data["joints"] = None

    # Load text description
    if text_path.exists():
        with open(text_path, "r") as f:
            descriptions = [line.strip().split("#")[0] for line in f.readlines()]
            data["text"] = descriptions[0] if descriptions else ""
    else:
        data["text"] = ""

    return data


# ========== visualization.py ==========

"""
Motion visualization utilities.

Provides 3D animation and comparison visualization for motion sequences.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# [internal import removed]  from utils.motion_utils import T2M_KINEMATIC_CHAIN


def compute_forward_direction(joints: np.ndarray) -> np.ndarray:
    """Compute forward direction vector from joint positions using face joints."""
    l_hip, r_hip, sdr_r, sdr_l = 2, 1, 17, 16
    across = joints[r_hip] - joints[l_hip] + joints[sdr_r] - joints[sdr_l]
    norm = np.linalg.norm(across)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    across = across / norm
    return np.array([across[2], 0.0, -across[0]])


def probe_camera_state(ax) -> dict:
    """
    Print and return matplotlib 3D camera + scene state.

    Call this interactively after plot_3d_motion returns to capture the
    current view angle and grid sizing.

    Returns:
        dict with elev, azim, xlim, ylim, zlim, and x/y/z ranges
    """
    elev = ax.elev
    azim = ax.azim
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]

    state = dict(
        elev=elev,
        azim=azim,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        x_range=xr,
        y_range=yr,
        z_range=zr,
    )

    print("=" * 50)
    print(f"  elev       : {elev:.2f} deg")
    print(f"  azim       : {azim:.2f} deg")
    print(f"  xlim       : [{xlim[0]:.3f}, {xlim[1]:.3f}]  range={xr:.3f}")
    print(f"  ylim       : [{ylim[0]:.3f}, {ylim[1]:.3f}]  range={yr:.3f}")
    print(f"  zlim       : [{zlim[0]:.3f}, {zlim[1]:.3f}]  range={zr:.3f}")
    print("=" * 50)
    return state


def plot_3d_motion(
    motion: np.ndarray,
    fps: float = 20,
    radius: float = 1.0,
    title: str = "Motion Visualization",
    follow_root: bool = False,
    probe: bool = False,
    save_path: Optional[Path] = None,
    show_forward_vector: bool = False,
):
    import base64
    import io

    import imageio
    from IPython.display import HTML

    colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12", "#8e44ad"]
    pos_min = motion.min(axis=(0, 1))
    pos_max = motion.max(axis=(0, 1))

    x_range = [pos_min[0] - radius, pos_max[0] + radius]
    y_range = [pos_min[2] - radius, pos_max[2] + radius]
    z_range = [pos_min[1], pos_max[1] + 0.5]

    # create figure ONCE and reuse
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(False)
    ax.view_init(elev=15, azim=65)
    ax.set_xlim3d(x_range)
    ax.set_ylim3d(y_range)
    ax.set_zlim3d(z_range)
    ax.set_xlabel("X (Side)")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Height)")
    ax.set_title(title)

    if probe:
        print(f"\n[probe] Matplotlib camera + scene state for '{title}':")
        probe_camera_state(ax)

    # create line artists ONCE
    lines = [
        ax.plot([], [], [], color=colors[i % len(colors)], marker="o", ms=2, lw=2)[0]
        for i in range(len(T2M_KINEMATIC_CHAIN))
    ]

    forward_quiver = None
    if show_forward_vector:
        forward_quiver = ax.quiver(0, 0, 0, 0, 0, 1, color="red", alpha=0.8, normalize=True)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(save_path) if save_path else io.BytesIO()

    writer_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "output_params": ["-preset", "ultrafast", "-crf", "28"],
    }
    if save_path:
        writer = imageio.get_writer(target, format="FFMPEG", **writer_kwargs)
    else:
        writer = imageio.get_writer(target, format="mp4", **writer_kwargs)

    for frame_idx in range(len(motion)):
        if follow_root:
            root = motion[frame_idx, 0, :]
            ax.set_xlim3d([root[0] - radius, root[0] + radius])
            ax.set_ylim3d([root[2] - radius, root[2] + radius])

        # update artist data only — no new objects created
        for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
            joints = motion[frame_idx, c_indices, :]
            lines[i].set_data(joints[:, 0], joints[:, 2])
            lines[i].set_3d_properties(joints[:, 1])

        if show_forward_vector:
            root = motion[frame_idx, 0, :]
            forward = compute_forward_direction(motion[frame_idx])
            if forward_quiver is not None:
                forward_quiver.remove()
            forward_quiver = ax.quiver(
                root[0], root[2], root[1],
                forward[0], forward[2], forward[1],
                length=radius * 0.5, normalize=True, color="red", alpha=0.8
            )

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)

    writer.close()
    plt.close(fig)

    if save_path:
        print(f"Saved animation to {save_path}")
        return save_path

    target.seek(0)
    b64 = base64.b64encode(target.read()).decode()
    return HTML(f'<video controls width="600"><source src="data:video/mp4;base64,{b64}"></video>')


def visualize_motion(
    joint_positions: np.ndarray,
    title: str = "Motion Visualization",
    save_path: Optional[Path] = None,
    fps: float = 20,
    skip_frames: int = 1,
    radius: float = 1,
    notebook: bool = True,
    probe: bool = False,
    backend: str = "matplotlib",
    show_forward_vector: bool = False,
    follow_root: bool = False,
) -> Any:
    """
    Visualize motion from joint positions.

    Args:
        joint_positions: Joint positions (nframe, 22, 3)
        title: Plot title
        save_path: Optional path to save visualization
        fps: Frames per second
        skip_frames: Skip every N frames (reduces rendering time)
        radius: Radius of the viewing box
        notebook: Whether to return visualization for notebook display
        probe: If True, print camera + scene state
        backend: Visualization backend - only "matplotlib" is supported
        show_forward_vector: If True, draw forward direction vector from root joint
        follow_root: If True, center camera view on root joint; if False, camera stays fixed so motion traverses the room
    """
    if backend != "matplotlib":
        print(f"Backend '{backend}' is not supported. Using matplotlib.")
    fps = fps / skip_frames
    motion_subsampled = joint_positions[::skip_frames]
    html = plot_3d_motion(
        motion_subsampled,
        radius=radius,
        fps=fps,
        title=title,
        probe=probe,
        show_forward_vector=show_forward_vector,
        follow_root=follow_root,
    )
    return html


def plot_3d_motion_comparison(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    fps: float = 20,
    radius: float = 1.0,
    title: str = "Generated vs Ground Truth",
    probe: bool = False,
    save_path: Optional[Path] = None,
    show_forward_vector: bool = False,
):
    import base64
    import io

    import imageio
    from IPython.display import HTML

    gen_colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12", "#8e44ad"]
    gt_color = matplotlib.colors.to_rgba("#aaaaaa", alpha=0.4)

    n_frames = min(len(generated_joints), len(ground_truth_joints))

    all_joints = np.concatenate([generated_joints[:n_frames], ground_truth_joints[:n_frames]], axis=1)
    pos_min = all_joints.min(axis=(0, 1))
    pos_max = all_joints.max(axis=(0, 1))

    x_range = [pos_min[0] - radius, pos_max[0] + radius]
    y_range = [pos_min[2] - radius, pos_max[2] + radius]
    z_range = [pos_min[1], pos_max[1] + 0.5]

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(False)
    ax.view_init(elev=15, azim=65)
    ax.set_xlim3d(x_range)
    ax.set_ylim3d(y_range)
    ax.set_zlim3d(z_range)
    ax.set_xlabel("X (Side)")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Height)")
    ax.set_title(title)

    if probe:
        print(f"\n[probe] Matplotlib camera + scene state for '{title}':")
        probe_camera_state(ax)

    gt_lines = [
        ax.plot([], [], [], color=gt_color, marker="o", ms=2, lw=2)[0]
        for _ in range(len(T2M_KINEMATIC_CHAIN))
    ]
    gen_lines = [
        ax.plot([], [], [], color=gen_colors[i % len(gen_colors)], marker="o", ms=2, lw=2)[0]
        for i in range(len(T2M_KINEMATIC_CHAIN))
    ]

    gt_root_traj_color = matplotlib.colors.to_rgba("#aaaaaa", alpha=0.25)
    gen_root_traj_color = matplotlib.colors.to_rgba("#2980b9", alpha=0.35)
    gt_root_line = ax.plot([], [], [], color=gt_root_traj_color, lw=1.5, linestyle="--")[0]
    gen_root_line = ax.plot([], [], [], color=gen_root_traj_color, lw=1.5, linestyle="--")[0]

    gt_forward_quiver = None
    gen_forward_quiver = None
    if show_forward_vector:
        gt_forward_quiver = ax.quiver(0, 0, 0, 0, 0, 1, color="red", alpha=0.8, normalize=True)
        gen_forward_quiver = ax.quiver(0, 0, 0, 0, 0, 1, color="red", alpha=0.8, normalize=True)

    gt_roots_x = ground_truth_joints[:n_frames, 0, 0]
    gt_roots_z = ground_truth_joints[:n_frames, 0, 2]
    gt_roots_y = ground_truth_joints[:n_frames, 0, 1]
    gen_roots_x = generated_joints[:n_frames, 0, 0]
    gen_roots_z = generated_joints[:n_frames, 0, 2]
    gen_roots_y = generated_joints[:n_frames, 0, 1]

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(save_path) if save_path else io.BytesIO()

    writer_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "output_params": ["-preset", "ultrafast", "-crf", "28"],
    }
    if save_path:
        writer = imageio.get_writer(target, format="FFMPEG", **writer_kwargs)
    else:
        writer = imageio.get_writer(target, format="mp4", **writer_kwargs)

    for frame_idx in range(n_frames):
        for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
            gt_joints = ground_truth_joints[frame_idx, c_indices, :]
            gt_lines[i].set_data(gt_joints[:, 0], gt_joints[:, 2])
            gt_lines[i].set_3d_properties(gt_joints[:, 1])

            gen_joints = generated_joints[frame_idx, c_indices, :]
            gen_lines[i].set_data(gen_joints[:, 0], gen_joints[:, 2])
            gen_lines[i].set_3d_properties(gen_joints[:, 1])

        gt_root_line.set_data(gt_roots_x[:frame_idx + 1], gt_roots_z[:frame_idx + 1])
        gt_root_line.set_3d_properties(gt_roots_y[:frame_idx + 1])
        gen_root_line.set_data(gen_roots_x[:frame_idx + 1], gen_roots_z[:frame_idx + 1])
        gen_root_line.set_3d_properties(gen_roots_y[:frame_idx + 1])

        if show_forward_vector:
            gt_root = ground_truth_joints[frame_idx, 0, :]
            gt_forward = compute_forward_direction(ground_truth_joints[frame_idx])
            if gt_forward_quiver is not None:
                gt_forward_quiver.remove()
            gt_forward_quiver = ax.quiver(
                gt_root[0], gt_root[2], gt_root[1],
                gt_forward[0], gt_forward[2], gt_forward[1],
                length=radius * 0.5, normalize=True, color="red", alpha=0.8
            )

            gen_root = generated_joints[frame_idx, 0, :]
            gen_forward = compute_forward_direction(generated_joints[frame_idx])
            if gen_forward_quiver is not None:
                gen_forward_quiver.remove()
            gen_forward_quiver = ax.quiver(
                gen_root[0], gen_root[2], gen_root[1],
                gen_forward[0], gen_forward[2], gen_forward[1],
                length=radius * 0.5, normalize=True, color="red", alpha=0.8
            )

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)

    writer.close()
    plt.close(fig)

    if save_path:
        print(f"Saved animation to {save_path}")
        return save_path

    target.seek(0)
    b64 = base64.b64encode(target.read()).decode()
    return HTML(f'<video controls width="600"><source src="data:video/mp4;base64,{b64}"></video>')


def compare_motions(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    save_path: Optional[Path] = None,
    fps: float = 20,
    radius: float = 1.0,
    backend: str = "matplotlib",
    probe: bool = False,
    show_forward_vector: bool = False,
) -> Any:
    """
    Compare generated motion with ground truth on the same 3D axes.

    GT is rendered in light gray (#aaaaaa) with alpha=0.4.
    Generated is rendered in original kinematic chain colors.
    """
    return plot_3d_motion_comparison(
        generated_joints,
        ground_truth_joints,
        fps=fps,
        radius=radius,
        title="Generated vs Ground Truth",
        probe=probe,
        save_path=save_path,
        show_forward_vector=show_forward_vector,
    )


# ========== pipeline.py ==========

"""
Inference pipeline for Text-to-3D Motion generation using FlowMatchingPredictor and MotionHistoryEncoder.
"""

from typing import Any, Optional, Union

import numpy as np
import torch

# [internal import removed]  from utils.motion_utils import FeatureNormalizer, enforce_rigid_bone_lengths, positions_to_x271, x68_to_positions


def temporal_gaussian_smooth(positions: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """Smooth joint positions along the temporal dimension using a 1D Gaussian filter.

    Args:
        positions: (horizon, 22, 3) tensor of joint positions.
        sigma: Standard deviation of Gaussian filter in frames.

    Returns:
        smoothed_positions: (horizon, 22, 3) tensor of smoothed joint positions.
    """
    if sigma <= 0.0 or positions.shape[0] < 5:
        return positions
    radius = int(round(2.0 * sigma))
    kernel_size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=positions.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    T, J, D = positions.shape
    # Reshape to (1, J*D, T) for 1D depthwise convolution
    feat = positions.permute(1, 2, 0).reshape(1, J * D, T)
    weight = kernel.view(1, 1, kernel_size).expand(J * D, 1, -1)
    pad_feat = torch.nn.functional.pad(feat, (radius, radius), mode="replicate")
    smoothed = torch.nn.functional.conv1d(pad_feat, weight, groups=J * D)
    return smoothed.view(J, D, T).permute(2, 0, 1)


@torch.no_grad()
def generate_motion_from_prompt(
    text_prompt: str,
    history_motion: torch.Tensor,     # (1, history_length, 271) raw
    initial_joints: torch.Tensor,     # (1, 22, 3) raw - seed positions
    initial_frame: torch.Tensor,      # (1, 271) raw - seed frame
    predictor_trainer: Any,
    decoder_trainer: Any,
    normalizer: FeatureNormalizer,
    clip_encoder: Any,
    device: torch.device | str = "cuda",
    num_inference_steps: int = 20,
    time_schedule_power: float = 1.0,
    guidance_scale: float = 2.5,
    horizon: int = 80,
    history_valid_length: Optional[Union[int, torch.Tensor]] = None,
    solver: str = "midpoint",
    velocity_scale: float = 1.0,
    smooth_output: bool = True,
    smooth_sigma: float = 1.2,
    enforce_rigid_bones: bool = True,
) -> np.ndarray:
    """Generate joint positions by integrating flow matching ODE in latent space.

    Args:
        text_prompt: Text description of the motion to generate.
        history_motion: Historical motion sequence (1, history_length, 271) RAW features.
        initial_joints: Seed joint positions (1, 22, 3) RAW features for initial frame.
        initial_frame: Seed frame features (1, 271) RAW features for frame 0.
        predictor_trainer: FlowMatchingTrainer instance.
        decoder_trainer: LatentDecoder trainer instance (or object with .ema_decoder).
        normalizer: FeatureNormalizer instance for raw feature normalization.
        clip_encoder: CLIP text sequence encoder.
        device: Device to execute generation on.
        num_inference_steps: Number of integration steps.
        time_schedule_power: Power parameter for ODE time discretization schedule (1.0 = Uniform/Linear).
        guidance_scale: Classifier-Free Guidance (CFG) scale (2.5 recommended).
        horizon: Target horizon sequence length (default 80 frames).
        history_valid_length: Exact count of non-padded history frames. Derived if None.
        solver: ODE solver to use ('midpoint' 2nd-order RK2 or 'euler' 1st-order).
        velocity_scale: Scaling factor for velocity vector to restore ground truth kinematic energy (default 1.15).
        smooth_output: Whether to apply temporal Gaussian smoothing to joint positions.
        smooth_sigma: Gaussian smoothing standard deviation in frames (default 1.2).

    Returns:
        generated_positions: (horizon, 22, 3) numpy array of generated joint positions.
    """
    target_encoder = decoder_trainer.ema_encoder.model
    target_encoder.eval()
    predictor = predictor_trainer.ema_predictor.model
    predictor.eval()
    decoder = decoder_trainer.ema_decoder.model
    decoder.eval()

    device = torch.device(device) if isinstance(device, str) else device

    # Step 1: Clear KV cache and reference state in predictor
    predictor.clear_cache()

    # Step 2: Encode text prompt to sequence embedding
    text_seq = clip_encoder.encode_sequence([text_prompt]).to(device)  # (1, S, 512)
    text_pooled_ref = text_seq.mean(dim=1)  # for zeros sizing

    # Step 3: Compute history conditioning context
    zeros_hist = torch.zeros(1, text_pooled_ref.shape[1], device=device, dtype=history_motion.dtype)
    history_norm = normalizer.normalize(history_motion.to(device))
    track_features = target_encoder(
        history_norm, zeros_hist, mask=None, return_layer_outputs=False
    ).detach()  # (1, T_hist, H_enc)

    T_hist = track_features.shape[1]
    S_text = text_seq.shape[1]
    S_cond = S_text + T_hist

    # Determine valid history length and construct boolean key-padding mask
    if history_valid_length is None:
        # Check if history_motion is all zeros (e.g. generation from scratch)
        is_all_zero = (history_motion == 0).all().item()
        val_len = 0 if is_all_zero else T_hist
    elif isinstance(history_valid_length, torch.Tensor):
        val_len = int(history_valid_length.item())
    else:
        val_len = int(history_valid_length)

    hist_idx = torch.arange(T_hist, device=device).unsqueeze(0)  # (1, T_hist)
    valid_start = T_hist - val_len
    hist_mask = hist_idx >= valid_start  # (1, T_hist) bool
    text_mask = torch.ones((1, S_text), dtype=torch.bool, device=device)  # (1, S_text) bool
    cond_mask = torch.cat([text_mask, hist_mask], dim=1)  # (1, S_cond) bool
    key_padding_mask = cond_mask.unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S_cond) bool

    # Step 4: Sample noise for target sequence latents (T_target = horizon - 1 frames)
    T_target = horizon - 1
    z_t = torch.randn(1, T_target, target_encoder.config.hidden_size, device=device)

    # Step 5: Integrate flow ODE from t=0 to t=1
    s = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    tau = s if time_schedule_power == 1.0 else 1.0 - (1.0 - s).pow(time_schedule_power)

    # Prepare conditioned & unconditioned embeddings
    combined_cond = torch.cat([text_seq, track_features], dim=1)  # (1, S + T_hist, 512)
    text_pooled = text_seq.mean(dim=1)  # (1, 512)

    null_emb = predictor_trainer.null_text_embedding(1).to(dtype=text_seq.dtype)  # (1, 512)
    null_seq = null_emb.unsqueeze(1).expand(-1, text_seq.shape[1], -1)  # (1, S, 512)
    combined_uncond = torch.cat([null_seq, track_features], dim=1)  # (1, S + T_hist, 512)
    null_pooled = null_seq.mean(dim=1)  # (1, 512)

    def eval_velocity(z_curr: torch.Tensor, t_curr: torch.Tensor) -> torch.Tensor:
        v_c, _, _ = predictor(
            noisy_states=z_curr,
            timesteps=t_curr,
            track_features=combined_cond,
            text_embedding=text_pooled,
            key_padding_mask=key_padding_mask,
            history_states=track_features,
        )
        if guidance_scale > 1.0:
            v_u, _, _ = predictor(
                noisy_states=z_curr,
                timesteps=t_curr,
                track_features=combined_uncond,
                text_embedding=null_pooled,
                key_padding_mask=key_padding_mask,
                history_states=track_features,
            )
            v_eff = v_u + guidance_scale * (v_c - v_u)
        else:
            v_eff = v_c
        return v_eff * velocity_scale

    for step in range(num_inference_steps):
        t_start = tau[step].expand(1)
        dt = (tau[step + 1] - tau[step]).item()

        if solver == "midpoint":
            v1 = eval_velocity(z_t, t_start)
            t_mid = (tau[step] + 0.5 * dt).expand(1)
            z_mid = z_t + 0.5 * dt * v1
            v_mid = eval_velocity(z_mid, t_mid)
            z_t = z_t + v_mid * dt
        else:
            v1 = eval_velocity(z_t, t_start)
            z_t = z_t + v1 * dt

    # Final integrated latent in normalized space
    z1_pred_norm = z_t  # (1, T_target, H_enc)

    # Step 6: Denormalize latent back to encoder's native latent scale before passing to decoder
    z1_pred = predictor_trainer.denormalize_latent(z1_pred_norm)
    decoded_68d = decoder(z1_pred)  # (1, T_target, 68)

    # Step 7: Step-by-step autoregressive reconstruction of joint positions
    current_pos = initial_joints.to(device)  # (1, 22, 3)
    current_frame = normalizer.normalize(initial_frame.to(device))  # (1, 271)

    generated_positions = [current_pos[0].cpu()]

    for frame_idx in range(T_target):
        pred_68d = decoded_68d[:, frame_idx]  # (1, 68)

        new_pos = x68_to_positions(
            pred_68d,
            normalizer,
            current_frame,
            current_pos,
        )  # (1, 22, 3)

        new_frame, _ = positions_to_x271(new_pos, current_pos, normalizer)

        current_pos = new_pos
        current_frame = new_frame
        generated_positions.append(new_pos[0].cpu())

    positions_tensor = torch.stack(generated_positions, dim=0)  # (horizon, 22, 3)
    if enforce_rigid_bones:
        positions_tensor = enforce_rigid_bone_lengths(positions_tensor, initial_joints[0].cpu())
    if smooth_output:
        positions_tensor = temporal_gaussian_smooth(positions_tensor, sigma=smooth_sigma)

    return positions_tensor.cpu().numpy()


# ========== wandb_logger.py ==========

"""
W&B Logger for external training monitoring.

Works seamlessly on both local machine and Kaggle notebooks.
Auto-detects Kaggle environment and uses secrets for authentication.

Usage:
    # [internal import removed]  from utils.wandb_logger import WandbLogger

    logger = WandbLogger(
        project="motion-generation",
        config={"lr": 1e-4, "epochs": 100}
    )
    # Run name auto-generated as "run-YYYY-MM-DD_HH-MM-SS"

    logger.log({"loss": loss, "lr": lr}, step=global_step)
    logger.log_model("checkpoints/best.pt", "best-model")
    logger.finish()
"""

import os
import sys
from typing import Any, Dict, Optional

# Check if wandb is available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


def is_kaggle_environment() -> bool:
    """Detect if running in Kaggle notebook."""
    return os.path.exists("/kaggle") or "kaggle" in sys.executable.lower()


def get_kaggle_secret(secret_name: str) -> Optional[str]:
    """Get secret from Kaggle secrets if available."""
    if not is_kaggle_environment():
        return None

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(secret_name)
    except Exception:
        return None


class WandbLogger:
    """
    W&B logger with Kaggle support and graceful fallback.

    Features:
    - Auto-detects Kaggle environment and authenticates via secrets
    - Graceful fallback when wandb is not installed
    - Simple API for logging metrics and model checkpoints

    Args:
        project: W&B project name
        name: Run name (optional, auto-generated if not provided)
        config: Dictionary of hyperparameters to log
        kaggle_secret_name: Name of the Kaggle secret containing W&B API key
        enabled: Set to False to disable logging (useful for testing)

    Example:
        >>> logger = WandbLogger(
        ...     project="motion-generation",
        ...     name="flow-predictor-v1",
        ...     config={"lr": 1e-4, "epochs": 100, "batch_size": 64}
        ... )
        >>> logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume_id: Optional[str] = None,
        kaggle_secret_name: str = "WANDB_API_KEY",
        enabled: bool = True,
    ):
        self.project = project
        self.config = config or {}
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.resume_id = resume_id

        # Auto-generate run name from datetime if not provided
        if name is None:
            self.name = "motion-generation-buet"
        else:
            self.name = name

        if wandb is None:
            print("[WandbLogger] wandb not available. Cannot log model.")
            return

        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("[WandbLogger] wandb not installed. Logging disabled.")
            elif not enabled:
                print("[WandbLogger] Logging disabled by user.")
            return

        # Try to authenticate
        self._authenticate(kaggle_secret_name)

        # Initialize run
        try:
            self.run = wandb.init(
                project=project,
                entity="motion-generation-buet",
                name=self.name,
                config=config,
                id=self.resume_id,
                resume=True if self.resume_id else None,
                reinit=True,
            )
            print(f"[WandbLogger] Initialized run: {self.run.name}")
            print(f"[WandbLogger] View at: {self.run.url}")
        except Exception as e:
            print(f"[WandbLogger] Failed to initialize: {e}")
            self.enabled = False

    def _authenticate(self, secret_name: str) -> None:
        """Authenticate with W&B, using Kaggle secret if available."""
        # Check for API key in environment or Kaggle secrets
        api_key = os.environ.get("WANDB_API_KEY")

        if api_key is None:
            api_key = get_kaggle_secret(secret_name)

        if wandb is None:
            print("[WandbLogger] wandb not available. Cannot log model.")
            return

        if api_key:
            try:
                wandb.login(key=api_key)
                print("[WandbLogger] Authenticated successfully.")
            except Exception as e:
                print(f"[WandbLogger] Authentication failed: {e}")
        else:
            print("[WandbLogger] No API key found. Using existing login or anonymous mode.")

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step (optional, auto-incremented if not provided)
        """
        if not self.enabled or self.run is None:
            return

        if wandb is None:
            print("[WandbLogger] wandb not available. Cannot log metrics.")
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[WandbLogger] Failed to log metrics: {e}")

    def log_model(
        self,
        path: str,
        name: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Log a model checkpoint as a W&B artifact.

        Args:
            path: Path to the checkpoint file
            name: Name for the artifact
            description: Optional description
        """
        if not self.enabled or self.run is None:
            return

        if wandb is None:
            print("[WandbLogger] wandb not available. Cannot log model.")
            return

        try:
            artifact = wandb.Artifact(name, type="model", description=description)
            artifact.add_file(path)
            self.run.log_artifact(artifact)
            print(f"[WandbLogger] Logged model artifact: {name}")
        except Exception as e:
            print(f"[WandbLogger] Failed to log model: {e}")

    def log_image(
        self,
        key: str,
        path: str,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log an image file to W&B so it appears in the run media.

        Args:
            key: Metric/media key shown in W&B
            path: Local path to the image file
            step: Optional global step for the log entry
            caption: Optional display caption
        """
        if not self.enabled or self.run is None:
            return

        if wandb is None:
            print("[WandbLogger] wandb not available. Cannot log model.")
            return

        try:
            wandb.log({key: wandb.Image(path, caption=caption)}, step=step)
        except Exception as e:
            print(f"[WandbLogger] Failed to log image: {e}")

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Log final summary metrics (shown in W&B run summary).

        Args:
            metrics: Dictionary of final metric values
        """
        if not self.enabled or self.run is None:
            return

        try:
            for key, value in metrics.items():
                wandb.run.summary[key] = value  # type: ignore
        except Exception as e:
            print(f"[WandbLogger] Failed to log summary: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self.run is None:
            return

        if wandb is None:
            print("[WandbLogger] wandb not available. Cannot log model.")
            return

        try:
            wandb.finish()
            print("[WandbLogger] Run finished.")
        except Exception as e:
            print(f"[WandbLogger] Failed to finish run: {e}")

    def __enter__(self) -> "WandbLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically finish run."""
        self.finish()


# ========== models/__init__.py ==========

"""
Model architectures for Human Motion Animation Generation.

This module contains:
- AutoregressiveContextEncoder: Encodes motion context sequentially
- FlowMatchingNetwork: Generates motion sequences using flow matching

Compatible with 271D custom feature format from motion_utils.py:
- [0:3]   Root height Y, Root velocity X, Root velocity Z (velocity form)
- [3:69]  22 RIC positions (22 * 3)
- [69:201] 22 6D rotations (22 * 6)
- [201:267] 22 local velocities (22 * 3)
- [267:271] Foot contacts (4D)

Note: Root X,Z are stored as velocities for autoregressive stability.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

# [internal import removed]  from utils.config import FlowMatchingPredictorConfig, MotionHistoryEncoderConfig


class KinematicChainEncoder(nn.Module):
    """
    Encodes the kinematic hierarchy of the skeleton.
    Each joint is mapped to a unique (chain_id, depth) pair based on the T2M skeleton.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        # t2m_kinematic_chain:
        # 0: [0, 2, 5, 8, 11] (Root -> R-Leg)
        # 1: [0, 1, 4, 7, 10] (Root -> L-Leg)
        # 2: [0, 3, 6, 9, 12, 15] (Root -> Spine -> Head)
        # 3: [9, 14, 17, 19, 21] (Neck -> R-Arm)
        # 4: [9, 13, 16, 18, 20] (Neck -> L-Arm)

        joint_to_chain = [0] * 22
        joint_to_depth = [0] * 22

        # Trace and assign:
        # Chain 0: Root + Right Leg
        for d, j in enumerate([0, 2, 5, 8, 11]):
            joint_to_chain[j], joint_to_depth[j] = 0, d
        # Chain 1: Left Leg
        for d, j in enumerate([1, 4, 7, 10], 1):
            joint_to_chain[j], joint_to_depth[j] = 1, d
        # Chain 2: Spine + Head
        for d, j in enumerate([3, 6, 9, 12, 15], 1):
            joint_to_chain[j], joint_to_depth[j] = 2, d
        # Chain 3: Right Arm (starts from joint 9, depth 3)
        for d, j in enumerate([14, 17, 19, 21], 4):
            joint_to_chain[j], joint_to_depth[j] = 3, d
        # Chain 4: Left Arm (starts from joint 9, depth 3)
        for d, j in enumerate([13, 16, 18, 20], 4):
            joint_to_chain[j], joint_to_depth[j] = 4, d

        self.register_buffer("joint_to_chain", torch.tensor(joint_to_chain))
        self.register_buffer("joint_to_depth", torch.tensor(joint_to_depth))

        # Type annotations for Pylance (converts Module buffers to indexed tensors)
        self.joint_to_chain: torch.Tensor
        self.joint_to_depth: torch.Tensor

        self.chain_emb = nn.Embedding(5, model_dim // 2)
        self.depth_emb = nn.Embedding(8, model_dim // 2)

    def forward(self, joint_ids: torch.Tensor) -> torch.Tensor:
        # joint_ids: (n_joints,)
        chains = self.joint_to_chain[joint_ids]
        depths = self.joint_to_depth[joint_ids]
        return torch.cat([self.chain_emb(chains), self.depth_emb(depths)], dim=-1)  # (n_joints, model_dim)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Applies conditioning BEFORE an operation (attention or FFN)
    by modulating the normalized input with learned scale and shift.

    Following DiT (Peebles & Xie, 2023) AdaLN-Zero formulation:
      - scale, shift, gate are all predicted from the condition
      - all three are zero-initialized → identity at step 0
      - gate is applied AFTER the operation as a residual scale
    """

    def __init__(self, d_model: int, d_cond: int, init_gate_bias: float = 0.0):
        super().__init__()

        # No learnable affine params — AdaLN supplies them externally
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Predicts scale (γ), shift (β), gate (α) for one sub-layer
        # Zero-init → at step 0: scale=0, shift=0, gate=0
        #   effective scale = 1 + 0 = 1 (identity norm)
        #   effective gate  = 0         (zero residual contribution)
        self.proj = nn.Linear(d_cond, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if init_gate_bias != 0.0:
            nn.init.constant_(self.proj.bias[2 * d_model:], init_gate_bias)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, d_model]
        cond: torch.Tensor,  # [B, d_cond]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_modulated: AdaLN(x, cond) — feed this into attention/FFN
            gate:        [B, 1, d_model] — multiply with op output
                         before adding residual
        """
        # Project condition to scale, shift, gate
        # SiLU activation on condition before projection — standard
        params = self.proj(torch.nn.functional.silu(cond))  # [B, 3*d]
        scale, shift, gate = params.chunk(3, dim=-1)  # each [B, d]

        # Unsqueeze over T for broadcasting
        scale = scale.unsqueeze(1)  # [B, 1, d_model]
        shift = shift.unsqueeze(1)  # [B, 1, d_model]
        gate = gate.unsqueeze(1)  # [B, 1, d_model]

        # Modulated norm — applied before attention/FFN
        x_modulated = self.norm(x) * (1 + scale) + shift

        # Gate returned separately — applied after attention/FFN
        return x_modulated, gate


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: Optional[int] = None,
        *,
        bias: bool,
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        if output_size is None:
            output_size = hidden_size

        self.down_proj = nn.Linear(intermediate_size, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = ACT2FN[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TemporalRoPEAttention(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig | FlowMatchingPredictorConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.config = config
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"TemporalRoPEAttention requires an even per-head dimension, got head_dim={self.head_dim}."
            )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = config.attention_dropout

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _build_rope(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = torch.exp(
            torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / self.head_dim)
        )
        freqs = positions[:, None] * inv_freq[None, :]
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)
        return cos.view(1, 1, seq_len, self.head_dim), sin.view(1, 1, seq_len, self.head_dim)

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)


@dataclass
class TemporalLayerCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


@dataclass
class TemporalCacheState:
    layers: List[TemporalLayerCache]


def init_weights(module: nn.Module, linear_init: str = "xavier_normal", linear_std: float = 0.02) -> None:
    """Apply standard weight initialization to a module and all sub-modules.

    Args:
        module: The module to initialize (typically called as init_weights(self) from __init__).
        linear_init: Initialization scheme for nn.Linear weights.
                     Use "xavier_normal" for flow/predictor, "trunc_normal" for encoder.
        linear_std: Standard deviation for trunc_normal init (ignored for xavier_normal).
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if linear_init == "trunc_normal":
                nn.init.trunc_normal_(m.weight, std=linear_std)
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# [internal import removed]  from utils.models._base_trainer import (

# Expose model classes for package import
# [internal import removed]  from utils.models.motion_history_encoder import MotionHistoryEncoder
# [internal import removed]  from utils.models.flow_matching_predictor import FlowMatchingPredictor, LatentDecoder


# ========== models/motion_history_encoder.py ==========

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# [internal import removed]  from utils.config import Config, MotionHistoryEncoderConfig
# [internal import removed]  from utils.models import AdaLN, GatedMLP, TemporalCacheState, TemporalLayerCache, TemporalRoPEAttention, init_weights


class EncoderRoPEAttention(TemporalRoPEAttention):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__(config)
        self.num_registers = config.num_registers
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"TemporalRoPEAttention requires an even per-head dimension, got head_dim={self.head_dim}."
            )

    def forward(self, hidden_states: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = self._reshape_heads(self.k_proj(hidden_states))
        value = self._reshape_heads(self.v_proj(hidden_states))

        cos, sin = self._build_rope(
            seq_len=seq_len - self.num_registers,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query[:, :, self.num_registers :, :] = self._apply_rope(query[:, :, self.num_registers :, :], cos, sin)
        key[:, :, self.num_registers :, :] = self._apply_rope(key[:, :, self.num_registers :, :], cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


class EncoderMLP(GatedMLP):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation=config.hidden_act,
            dropout=config.dropout,
        )


class EncoderLayer(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()

        self.adaln_attn = AdaLN(d_model=config.hidden_size, d_cond=config.text_embedding_dim)

        self.self_attn = EncoderRoPEAttention(config)

        self.adaln_mlp = AdaLN(d_model=config.hidden_size, d_cond=config.text_embedding_dim)

        self.mlp = EncoderMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_emb: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, attn_gate = self.adaln_attn(hidden_states, text_emb)
        hidden_states = self.self_attn(hidden_states, is_causal=is_causal)
        hidden_states = residual + attn_gate * hidden_states

        residual = hidden_states
        hidden_states, mlp_gate = self.adaln_mlp(hidden_states, text_emb)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_gate * hidden_states

        return hidden_states


class MotionHistoryEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self.config = config.encoder_config
        enc_config = config.encoder_config

        self.frame_projection = nn.Linear(config.motion_dim, enc_config.hidden_size, bias=True)

        self.layers = nn.ModuleList([EncoderLayer(enc_config) for _ in range(enc_config.num_hidden_layers)])

        self.final_norm = nn.LayerNorm(enc_config.hidden_size, eps=enc_config.layer_norm_eps)

        self.register_tokens = nn.Parameter(torch.empty(enc_config.num_registers, enc_config.hidden_size))
        self.mask_token = nn.Parameter(torch.empty(1, enc_config.hidden_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.01)
        init_weights(self, linear_init="trunc_normal", linear_std=0.02)

    def _empty_cache_state(self) -> TemporalCacheState:
        return TemporalCacheState(layers=[TemporalLayerCache() for _ in range(self.config.num_hidden_layers)])

    def forward(
        self,
        motion_seq: torch.Tensor,
        text_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_layer_outputs: bool = False,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if motion_seq.ndim != 3 or motion_seq.shape[-1] != self._config.motion_dim:
            raise ValueError(
                f"Expected motion_seq shape (B, T, {self._config.motion_dim}), got {tuple(motion_seq.shape)}"
            )
        if text_emb.ndim != 2 or text_emb.shape[-1] != self._config.text_embedding_dim:
            raise ValueError(
                f"Expected text_emb shape (B, {self._config.text_embedding_dim}), got {tuple(text_emb.shape)}"
            )
        if text_emb.shape[0] != motion_seq.shape[0]:
            raise ValueError(
                "Batch size mismatch between motion_seq and text_emb: "
                f"{tuple(motion_seq.shape)} vs {tuple(text_emb.shape)}"
            )

        if mask is not None and mask.shape != motion_seq.shape[:2]:
            if mask.ndim != 2:
                raise ValueError(
                    "Expected mask shape (B, T) matching motion_seq, got "
                    f"{tuple(mask.shape)} vs {tuple(motion_seq.shape[:2])}"
                )

        # --------------------------

        batch_size, seq_len, _ = motion_seq.shape
        if seq_len == 0:
            raise ValueError("Expected motion_seq with at least one timestep.")

        hidden_states: torch.Tensor = self.frame_projection(motion_seq)

        if mask is not None:
            mask_flat = mask.flatten()  # (B*T,)
            hidden_states_flat = hidden_states.flatten(0, 1)  # (B*T, H)
            hidden_states_flat[mask_flat] = self.mask_token.to(hidden_states_flat)
            hidden_states = hidden_states_flat.view_as(hidden_states)  # (B, T, H)

        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        hidden_states = torch.cat([register_tokens, hidden_states], dim=1)

        all_hidden_states: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states = layer(hidden_states, text_emb, is_causal=is_causal)

            if return_layer_outputs:
                all_hidden_states.append(hidden_states[:, self.config.num_registers :, :])

        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states[:, self.config.num_registers :, :]

        if return_layer_outputs:
            all_hidden_states[-1] = hidden_states
            return torch.stack(all_hidden_states, dim=2)  # (B, N, L, H)

        return hidden_states  # (B, N, H)

    def step(
        self,
        x_t: torch.Tensor,
        text_emb: torch.Tensor,
        frame_buffer: Optional[torch.Tensor],
        cache_state: Optional[TemporalCacheState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, TemporalCacheState]:
        if x_t.ndim != 2 or x_t.shape[-1] != self._config.motion_dim:
            raise ValueError(f"Expected x_t shape (B, {self._config.motion_dim}), got {tuple(x_t.shape)}")

        if frame_buffer is None:
            next_frame_buffer = x_t.unsqueeze(1)
        else:
            if frame_buffer.ndim != 3 or frame_buffer.shape[-1] != self._config.motion_dim:
                raise ValueError(
                    f"Expected frame_buffer shape (B, T, {self._config.motion_dim}), got {tuple(frame_buffer.shape)}"
                )
            if frame_buffer.shape[0] != x_t.shape[0]:
                raise ValueError(
                    "Batch size mismatch between x_t and frame_buffer: "
                    f"{tuple(x_t.shape)} vs {tuple(frame_buffer.shape)}"
                )
            next_frame_buffer = torch.cat([frame_buffer, x_t.unsqueeze(1)], dim=1)

        next_cache_state = cache_state or self._empty_cache_state()
        if len(next_cache_state.layers) != self.config.num_hidden_layers:
            raise ValueError(
                "cache_state layer count mismatch: "
                f"expected {self.config.num_hidden_layers}, got {len(next_cache_state.layers)}"
            )
        return (
            self.forward(next_frame_buffer, text_emb),
            next_frame_buffer,
            next_cache_state,
        )


class LinearProbe(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        text_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, text_embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states shape (B, T, D), got {tuple(hidden_states.shape)}")
        seq_mean = hidden_states.mean(dim=1)
        x = self.norm(seq_mean)
        x = self.linear(x)
        return F.normalize(x, dim=-1)


class JepaMLPBlock(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.jp_config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = GatedMLP(
            hidden_size=config.jp_config.hidden_size,
            intermediate_size=config.jp_config.intermediate_size,
            bias=True,
            activation=config.hidden_act,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class JepaPredictor(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.jp_config.hidden_size
        self.intermediate_size = config.jp_config.intermediate_size
        self.num_layers = config.jp_config.num_hidden_layers

        self.z_proj = nn.Linear(self.hidden_size // 4, self.hidden_size)

        self.input_mlp = GatedMLP(
            hidden_size=config.num_hidden_layers * config.hidden_size,
            intermediate_size=2 * self.hidden_size,
            output_size=self.hidden_size,
            bias=True,
            activation=config.hidden_act,
            dropout=config.dropout,
        )

        self.input_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.blocks = nn.ModuleList([JepaMLPBlock(config) for _ in range(self.num_layers)])

        self.final_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        self.output_mlp = GatedMLP(
            hidden_size=self.hidden_size,
            intermediate_size=4 * self.hidden_size,
            output_size=self.hidden_size * config.num_hidden_layers,
            bias=True,
            activation=config.hidden_act,
            dropout=config.dropout,
        )

        self.output_proj = nn.ModuleList(
            [nn.Linear(self.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, motion_history_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_history_emb: (B, T, L, H) where
                B = batch size,
                T = sequence length,
                L = config.num_hidden_layers,
                H = config.hidden_size.

        Returns:
            (B, T, L, H) where
                B = batch size,
                T = sequence length,
                L = config.num_hidden_layers,
                H = config.hidden_size.
        """

        if motion_history_emb.ndim != 4 or motion_history_emb.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Expected motion_history_emb shape "
                f"(B, T, L, {self.config.hidden_size}), got {tuple(motion_history_emb.shape)}"
            )

        B, T, L, H_enc = motion_history_emb.shape

        motion_history_emb = motion_history_emb.reshape(B, T, -1)  # (B, T, L*H_enc)
        motion_history_emb = self.input_mlp(motion_history_emb)  # (B, T, H_enc)

        z = torch.randn(B, self.z_proj.in_features, device=motion_history_emb.device)  # (B, H//4)
        # VJEPA-style latent noise — random noise injected as a learnable conditioning signal.

        z_proj = self.z_proj(z).unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        combined = torch.cat([motion_history_emb, z_proj], dim=-1)
        x = self.input_proj(combined)  # (B, T, H)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = self.output_mlp(x)  # (B, T, L*H)

        x = x.view(B, T, L, -1)  # (B, T, L, H)

        layer_outputs = []
        for i in range(self.config.num_hidden_layers):
            layer_output = self.output_proj[i](x[:, :, i, :])  # (B, T, H_enc)
            layer_outputs.append(layer_output)

        output = torch.stack(layer_outputs, dim=2)  # (B, T, L, H_enc)

        return output


# ========== models/flow_matching_predictor.py ==========

from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# [internal import removed]  from utils.config import Config, FlowMatchingPredictorConfig
# [internal import removed]  from utils.models import AdaLN, GatedMLP, TemporalLayerCache, TemporalRoPEAttention, init_weights
# [internal import removed]  from utils.motion_utils import FeatureNormalizer, positions_to_x271, x72_to_positions


class SinusoidalEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        if half == 0:
            return torch.zeros((t.shape[0], dim), device=t.device, dtype=torch.float32)

        max_period_tensor = torch.tensor(max_period, device=t.device, dtype=torch.float32)
        freqs = torch.exp(
            -torch.log(max_period_tensor) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        out = self.mlp(t_freq)
        return out


class PredictorMLP(GatedMLP):
    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation="silu",
            dropout=0.0,
        )


class PredictorRopeCrossAttention(TemporalRoPEAttention):
    def __init__(self, config: Config, text_bias: Optional[float] = None) -> None:
        super().__init__(config.predictor_config)
        self.config = config
        self.text_bias = text_bias if text_bias is not None else getattr(config.predictor_config, "text_prior_bias", 0.8)

        self.k_proj = nn.Linear(config.encoder_config.hidden_size, config.predictor_config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.encoder_config.hidden_size, config.predictor_config.hidden_size, bias=True)

        self.kv_cache = TemporalLayerCache()
        self._init_weights()

    def clear_cache(self) -> None:
        self.kv_cache.key = None
        self.kv_cache.value = None

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def attn_weights(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_weights = torch.softmax(attn_scores, dim=-1)

        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N, H)
        encoder_hidden_states: torch.Tensor,  # (B, M, H_enc)
        _encoder_cache: Optional[TemporalLayerCache] = None,
        output_attentions: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        encoder_cache = self.kv_cache if _encoder_cache is None else _encoder_cache

        if encoder_cache.key is None or encoder_cache.value is None:
            key = self._reshape_heads(self.k_proj(encoder_hidden_states))
            value = self._reshape_heads(self.v_proj(encoder_hidden_states))
            encoder_cache.key = key
            encoder_cache.value = value

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = encoder_cache.key
        value = encoder_cache.value

        # Build float additive attention mask with text prior bias
        Total_Keys = key.shape[-2]
        T_hist = getattr(self.config, "history_length", 40)
        S_text = max(0, Total_Keys - T_hist)

        float_mask = torch.zeros((1, 1, 1, Total_Keys), device=query.device, dtype=query.dtype)
        if S_text > 0 and self.text_bias != 0.0:
            float_mask[..., :S_text] = self.text_bias

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                float_mask = float_mask.masked_fill(~attn_mask, float("-inf"))
            else:
                float_mask = float_mask + attn_mask

        attn_output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=float_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        if output_attentions:
            attn_weights = self.attn_weights(query, key, value, attn_mask=float_mask)
            return attn_output, attn_weights

        return attn_output, None


class PredictorSelfAttention(TemporalRoPEAttention):
    def __init__(self, config: Config, use_self_attn_rope: Optional[bool] = None) -> None:
        super().__init__(config.predictor_config)
        pred_config = config.predictor_config
        self.config = config
        self.use_self_attn_rope = (
            use_self_attn_rope
            if use_self_attn_rope is not None
            else getattr(pred_config, "use_self_attn_rope", True)
        )

        self.encoder_cond_proj = nn.Linear(config.encoder_config.hidden_size, pred_config.hidden_size, bias=True)
        self.gate_proj = nn.Linear(3 * pred_config.hidden_size, 1, bias=True)
        self.cond_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -3.0)  # sigmoid(-3) ≈ 0.047

    def attn_weights(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_cond: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = self._reshape_heads(self.k_proj(hidden_states))
        value = self._reshape_heads(self.v_proj(hidden_states))

        if self.use_self_attn_rope:
            cos, sin = self._build_rope(
                seq_len=seq_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            query = self._apply_rope(query, cos, sin)
            key = self._apply_rope(key, cos, sin)

        attn_output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        if output_attentions:
            attn_w = self.attn_weights(query, key)
            return attn_output, attn_w

        return attn_output, None


class PredictorLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        pred_config = config.predictor_config

        self.adaln_self_attn = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size)

        self.self_attn = PredictorSelfAttention(config)

        self.adaln_cross_attn = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size, init_gate_bias=-2.0)

        self.cross_attn = PredictorRopeCrossAttention(config)

        self.adaln_mlp = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size)

        self.mlp = PredictorMLP(pred_config)

    def clear_cache(self) -> None:
        self.cross_attn.clear_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        adaln_cond: torch.Tensor,
        output_attentions: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        history_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states, self_attn_gate = self.adaln_self_attn(hidden_states, adaln_cond)
        hidden_states, self_attn_weights = self.self_attn(hidden_states, output_attentions=output_attentions)
        hidden_states = residual + self_attn_gate * hidden_states

        residual = hidden_states
        hidden_states, cross_attn_gate = self.adaln_cross_attn(hidden_states, adaln_cond)
        hidden_states, cross_attn_weights = self.cross_attn(
            hidden_states, encoder_hidden_states, output_attentions=output_attentions, attn_mask=key_padding_mask
        )
        hidden_states = residual + cross_attn_gate * hidden_states

        residual = hidden_states
        hidden_states, mlp_gate = self.adaln_mlp(hidden_states, adaln_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_gate * hidden_states

        if output_attentions:
            return hidden_states, self_attn_weights, cross_attn_weights

        return hidden_states, None, None


class FlowMatchingPredictor(nn.Module):
    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        pred_config = config.predictor_config
        if isinstance(pred_config, dict):
            # [internal import removed]  from utils.config import FlowMatchingPredictorConfig

            pred_config = FlowMatchingPredictorConfig(**pred_config)
            config.predictor_config = pred_config
        if isinstance(config.encoder_config, dict):
            # [internal import removed]  from utils.config import MotionHistoryEncoderConfig

            config.encoder_config = MotionHistoryEncoderConfig(**config.encoder_config)

        self.text_proj = nn.Linear(config.text_embedding_dim, pred_config.hidden_size, bias=True)

        # Time embedding for denoising timestep
        self.time_embedder = SinusoidalEmbedder(pred_config.hidden_size)

        # Transformer layers with AdaLN
        self.layers = nn.ModuleList([PredictorLayer(config) for _ in range(pred_config.num_hidden_layers)])

        self.latent_in_proj = nn.Linear(config.encoder_config.hidden_size, pred_config.hidden_size, bias=True)
        self.latent_out_proj = nn.Linear(pred_config.hidden_size, config.encoder_config.hidden_size, bias=True)
        # Output prediction head
        self.output_adaln = nn.Sequential(
            nn.Linear(pred_config.hidden_size, config.encoder_config.hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Linear(config.encoder_config.hidden_size * 2, config.encoder_config.hidden_size * 2, bias=True),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clear_cache()
        if hasattr(self, "_last_track_features"):
            del self._last_track_features

    def forward(
        self,
        noisy_states: torch.Tensor,
        timesteps: torch.Tensor,
        track_features: torch.Tensor,
        text_embedding: torch.Tensor,
        output_attentions: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        history_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Args:
            noisy_states: [B, N, H_enc]
            timesteps: [B] or [B, 1]
            track_features: [B, M+N, H_enc]
            text_embedding: [B, D]
            output_attentions: bool
            key_padding_mask: Optional[torch.Tensor] of shape (B, 1, 1, S_cond) or boolean mask
            history_states: Optional[torch.Tensor] of shape (B, T_hist, H_enc)
        Returns:
            all_cross_attns: List of attention weights from each layer if output_attentions is True, else None
        """
        # Clear/reset KV cache if training or if track_features changes to avoid cross-batch leaking/graph errors
        if self.training or not hasattr(self, "_last_track_features") or self._last_track_features is not track_features:
            self.clear_cache()
            self._last_track_features = track_features

        B, N, H = noisy_states.shape

        time_cond = self.time_embedder(timesteps.squeeze(-1) if timesteps.dim() > 1 else timesteps)  # [B, H_pred]
        text_cond = self.text_proj(text_embedding)  # [B, H_pred]

        # Equalize time and text conditioning power (50% time / 50% text)
        time_norm = F.normalize(time_cond, dim=-1, eps=1e-6)
        text_norm = F.normalize(text_cond, dim=-1, eps=1e-6)
        norm_scale = float(self.config.predictor_config.hidden_size ** 0.5)
        adaln_cond = norm_scale * (0.5 * time_norm + 0.5 * text_norm)

        all_self_attns: list[torch.Tensor] = []
        all_cross_attns: list[torch.Tensor] = []

        hidden_states = self.latent_in_proj(noisy_states)  # (B, N, H_pred)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = hidden_states

            hidden_states, self_attn_weights, cross_attn_weights = layer(
                hidden_states,
                encoder_hidden_states=track_features,
                adaln_cond=adaln_cond,
                output_attentions=output_attentions,
                key_padding_mask=key_padding_mask,
                history_states=history_states,
            )

            if output_attentions:
                all_self_attns.append(self_attn_weights)
                all_cross_attns.append(cross_attn_weights)

        hidden_states = self.latent_out_proj(hidden_states)  # (B, N, H_enc)

        output_shift, output_scale = self.output_adaln(adaln_cond).chunk(2, dim=-1)
        output_shift = output_shift.unsqueeze(1)
        output_scale = output_scale.unsqueeze(1)
        flow_prediction = output_shift + output_scale * hidden_states

        return (
            flow_prediction,
            all_self_attns if output_attentions else None,
            all_cross_attns if output_attentions else None,
        )


class DecoderMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm = nn.RMSNorm(config.decoder_config.hidden_size, eps=config.predictor_config.rms_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(config.decoder_config.hidden_size, config.decoder_config.intermediate_size, bias=True),
            nn.GELU(),
            nn.Dropout(config.decoder_config.dropout),
            nn.Linear(config.decoder_config.intermediate_size, config.decoder_config.hidden_size, bias=True),
            nn.Dropout(config.decoder_config.dropout),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class LatentDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        H = config.decoder_config.hidden_size

        self.down_proj = nn.Linear(config.encoder_config.hidden_size, H, bias=True)
        self.blocks = nn.Sequential(*[DecoderMLP(config) for _ in range(config.decoder_config.num_layers)])

        self.head_norm = nn.RMSNorm(H, eps=config.predictor_config.rms_norm_eps)

        self.root_head_xz = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, 2),
        )

        self.root_head_y = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 1),
        )

        self.yaw_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 2),
        )  # delta_yaw sin, cos

        self.ric_head = nn.Sequential(
            nn.Linear(H, 2 * H),
            nn.GELU(),
            nn.Linear(2 * H, 2 * H),
            nn.GELU(),
            nn.Linear(2 * H, 63),
        )  # 21 joint direct RIC positions * 3

        self.contact_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 4),
        )  # 4 foot contact logits

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        init_weights(self, linear_init="xavier_normal")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, H_encoder) - the latent representation from the predictor
        output: (B, 72) - the predicted reduced features (root_y, root_xz_vel, delta_yaw sin, cos, joint_ric * 3, foot_contacts)
        """
        x = self.down_proj(latent)
        x = self.blocks(x)
        x = self.head_norm(x)

        root_xz = self.root_head_xz(x)  # (..., 2) - root_xz_vel
        root_y = self.root_head_y(x)  # (..., 1) - root_y
        yaw = self.yaw_head(x)  # (..., 2) - delta_yaw sin, cos
        yaw = nn.functional.normalize(yaw, dim=-1)  # normalize to unit vector
        ric = self.ric_head(x)  # (..., 63) - 21 joint direct RIC positions
        contacts = self.contact_head(x)  # (..., 4) - foot contact logits

        return torch.cat([root_y, root_xz, yaw, ric, contacts], dim=-1)

    def decode(
        self, latent: torch.Tensor, prev_pos: torch.Tensor, prev_frame: torch.Tensor, normalizer: FeatureNormalizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes the predicted flow output into new joint positions and relative shifts.
        Args:
            latent: The latent representation from the predictor (B, H_enc).
            prev_pos: The previous joint positions (B, 22, 3) - only the first joint is used for flow decoding.
            prev_frame: The previous frame's full features (B, 271) - normalized.
            normalizer: The feature normalizer to denormalize the outputs.
        """

        pred = self.forward(latent)

        new_pos = x72_to_positions(
            pred,
            normalizer,
            prev_frame,
            prev_pos,
        )
        new_frame, _ = positions_to_x271(new_pos, prev_pos, normalizer)

        relative_shift = new_pos - prev_pos

        return new_pos, relative_shift, new_frame


# ========== models/_base_trainer.py ==========

"""Shared mixin and training utilities for motion-history trainer classes."""


import copy
from pathlib import Path
from typing import Any, Dict, Generic, OrderedDict, TypeVar, Union
from typing import Mapping as MappingABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events, State
from ignite.handlers import (
    TerminateOnNan,
    Timer,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import RunningAverage
from scipy import interpolate as Interp

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.dataset import create_dataloader
# [internal import removed]  from utils.motion_utils import Features, x72_to_positions, x271_to_x72
# [internal import removed]  from utils.wandb_logger import WandbLogger

T = TypeVar("T", bound=nn.Module)


class PretrainState(State):
    """Custom Ignite State for JEPA pretraining."""

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = 40
        self.metrics: Dict[str, Any] = {
            "global_step": 0,
            "best_train_loss": float("inf"),
        }
        self.schedules = config.pre_conf.schedules
        self.num_epochs = config.get_num_epochs()
        self.timer: Any = None

    def epoch_progress(self) -> float:
        """Return progress through current epoch as a float in [0, 1]."""
        return self.epoch / self.num_epochs if self.num_epochs > 0 else 0.0

    def get_schedule_value(self, name: str, default: float, interp: str = "previous") -> float:
        """Get current value of a scheduled parameter based on epoch progress."""
        schedule = self.schedules.get(name)

        if not schedule:
            return default

        t = self.epoch_progress()
        t_list, v_list = zip(*schedule)
        t_list = np.array(t_list)
        v_list = np.array(v_list)
        t_list = t_list / np.max(t_list)

        f = Interp.interp1d(t_list, v_list, kind=interp, assume_sorted=True)

        return float(f(t))

    def state_dict(self) -> dict:
        """Return a dictionary containing the state of the trainer."""
        super_dict: OrderedDict = super().state_dict()
        super_dict.update(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "metrics": self.metrics,
                "schedules": self.schedules,
                "num_epochs": self.num_epochs,
                "horizon": self.horizon,
            }
        )
        return super_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the trainer from a dictionary."""
        super().load_state_dict(state_dict)
        self.epoch = state_dict.get("epoch", 0)
        self.iteration = state_dict.get("iteration", 0)
        self.metrics = state_dict.get("metrics", {})
        self.schedules = state_dict.get("schedules", {})
        self.num_epochs = state_dict.get("num_epochs", 0)
        self.horizon = state_dict.get("horizon", 40)


class PretrainEngine(Engine):
    """Custom Ignite Engine for JEPA pretraining."""

    def __init__(self, process_function: Any, config: Config) -> None:
        super().__init__(process_function)
        self.state = PretrainState(config=config)
        self.config = config

    def get_metric(self, name: str, *args, **kwargs):
        """Get a metric value by name."""
        return self.state.metrics.get(name, *args, **kwargs)

    def get_metrics(self, names: list[str], prefix: str = "") -> dict[str, Any]:
        """Get specified metrics as a dict."""
        return {f"{prefix}{name}": self.state.metrics.get(name) for name in names}

    def set_metrics(self, pairs: list[tuple[str, Any]]) -> None:
        """Set multiple metrics at once."""
        for name, value in pairs:
            self.state.metrics[name] = value

    def clear_metrics(self, names: list[str]) -> None:
        """Clear specified metrics."""
        for name in names:
            self.state.metrics.pop(name, None)

    def scale_metrics(self, names: list[str], scaler: float) -> None:
        """Scale specified metrics by a factor."""
        for name in names:
            self.state.metrics[name] *= scaler

    def csa_op_metrics(self, pairs: list[tuple[str, float]], scalers: float | list[float], clear: bool = False) -> None:
        """Add a value to an existing metric (useful for running totals)."""
        if clear:
            self.clear_metrics([name for name, _ in pairs])
        if not isinstance(scalers, list):
            scalers = [scalers for _ in pairs]
        for (name, value), scaler in zip(pairs, scalers):
            self.state.metrics[name] = self.state.metrics.get(name, 0.0) + value * scaler


def estimate_time_remaining(engine: PretrainEngine, step_time: float, config: Config) -> float:
    """Estimate remaining training time based on current progress and elapsed time."""
    num_epochs = config.get_num_epochs()
    epoch = engine.state.epoch
    global_step = engine.get_metric("global_step")
    steps_per_epoch = global_step / epoch
    total_steps = num_epochs * steps_per_epoch
    remaining_steps = total_steps - global_step

    remaining_time = remaining_steps * step_time
    return remaining_time


class EMAModel(Generic[T]):
    """
    Exponential Moving Average model wrapper.

    Maintains an EMA copy of a model for more stable evaluation.
    EMA is used for validation, sampling, and checkpointing.
    """

    def __init__(self, model: T, decay: float = 0.999):
        """
        Initialize EMA model.

        Args:
            model: The model to create EMA copy of
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.model: T = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def update(self, model: T) -> None:
        """
        Update EMA weights.

        Args:
            model: The source model to update from
        """
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device: Union[str, torch.device]) -> "EMAModel[T]":
        """Move EMA model to device."""
        self.model.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return state dict of wrapped model for checkpointing."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: MappingABC) -> None:
        """Load state dict into wrapped model."""
        self.model.load_state_dict(state_dict)


class CheckpointMetadata:
    """Wrapper for non-stateful metadata to be saved with checkpoints.

    Ignite's Checkpoint handler requires all values in the to_save dict to have
    ``state_dict`` / ``load_state_dict`` methods. This wrapper lets us store
    simple metadata (like session_id) alongside model checkpoints without
    triggering infinite recursion in ignite's _tree_map (which would happen
    with bare strings since they are Sequences of single-char strings).
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def state_dict(self) -> dict[str, Any]:
        return self.data

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data = state_dict


def _find_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> Path | None:
    """Find the latest checkpoint file with given prefix."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


class _BaseTrainer:
    """Shared mixin for PretrainTrainer and FinetuneTrainer.

    Rust-style composition: no __init__; concrete trainers keep their own
    construction and call mixin helpers via self.<method>().
    """

    # -- dataloader + scheduler + wandb helpers --

    def _create_dataloaders(self) -> None:
        self.train_loader, self.normalizer = create_dataloader(self.config, "train", shuffle=True)
        self.val_loader, _ = create_dataloader(self.config, "val", shuffle=False)

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer, num_epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        accumulation_steps = (
            self.accumulation_steps
            if hasattr(self, "accumulation_steps") and self.accumulation_steps is not None
            else 1
        )
        steps_per_epoch = max(len(self.train_loader) // accumulation_steps, 1)
        total_steps = max(num_epochs * steps_per_epoch, 1)
        warmup_epochs = int(self.config.lr_warmup_epochs)
        pct_start = min(max(warmup_epochs / num_epochs, 0.0), 1.0) if num_epochs > 0 else 0.0
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.config.learning_rate),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    def _init_wandb(self, phase: str, extra_config: dict[str, Any] | None = None, name: str | None = None) -> None:
        if not getattr(self, "wandb_project", None):
            return
        cfg: dict[str, Any] = {
            "lr": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "ema_decay": float(self.config.ema_decay),
            "batch_size": getattr(self.train_loader, "batch_size", self.config.batch_size),
            "effective_batch_size": self.config.effective_batch_size,
            "phase": phase,
        }
        if extra_config:
            cfg.update(extra_config)
        self.wandb_logger = WandbLogger(
            project=self.wandb_project,
            name=name,
            config=cfg,
            resume_id=self.resume_id if self.resume_id else None,
        )
        self.resume_id = self.wandb_logger.run.id if self.wandb_logger.run else None

    # -- batch prep --

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion = batch["motion"].to(self.device)
        motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)
        joints = batch["joints"].to(self.device)
        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError(f"Expected motion (B, T, 271), got {tuple(motion.shape)}")
        return motion, text, joints

    # -- decoder loss (shared, fixes latent bug in PretrainTrainer) --

    def _decoder_loss(
        self, engine: PretrainEngine, motion: torch.Tensor, joints: torch.Tensor, decoded: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prev_positions = joints[:, :-1, :].flatten(0, 1)
        prev_frames = motion[:, :-1, :].flatten(0, 1)
        y_frames = motion[:, 1:, :].flatten(0, 1)

        decoded_flat = decoded.flatten(0, 1)
        decoded_denorm = self.normalizer.denormalize_x72(decoded_flat)

        y_72d = x271_to_x72(y_frames, self.normalizer, prev_positions=prev_positions, prev_x271=prev_frames)
        y_72d_denorm = self.normalizer.denormalize_x72(y_72d)

        dec_positions = x72_to_positions(
            decoded_flat,
            self.normalizer,
            prev_x271=prev_frames,
            prev_positions=prev_positions,
        )

        loss_root_y = F.mse_loss(decoded_denorm[:, :1], y_72d_denorm[:, :1], reduction="mean")
        loss_root_xz = F.mse_loss(decoded_denorm[:, 1:3], y_72d_denorm[:, 1:3], reduction="mean")
        loss_yaw = F.mse_loss(decoded_denorm[:, 3:5], y_72d_denorm[:, 3:5], reduction="mean")
        loss_ric = F.mse_loss(decoded_denorm[:, 5:68], y_72d_denorm[:, 5:68], reduction="mean")
        loss_contact = F.binary_cross_entropy_with_logits(
            decoded_flat[:, 68:72], y_frames[:, Features.CONTACTS], reduction="mean"
        )
        loss_joint = F.mse_loss(dec_positions, joints[:, 1:, :].flatten(0, 1), reduction="mean")

        decoder_loss = (
            1.0 * loss_root_y
            + 1.0 * loss_root_xz
            + 1.0 * loss_yaw
            + 1.0 * loss_ric
            + 0.5 * loss_contact
            + 1.0 * loss_joint
        )

        if engine is not None:
            engine.set_metrics(
                [
                    ("decoder_loss", decoder_loss.detach().item()),
                    ("loss_root_y", loss_root_y.detach().item()),
                    ("loss_root_xz", loss_root_xz.detach().item()),
                    ("loss_yaw", loss_yaw.detach().item()),
                    ("loss_ric", loss_ric.detach().item()),
                    ("loss_contact", loss_contact.detach().item()),
                    ("loss_joint", loss_joint.detach().item()),
                ],
            )

        return {"decoder_loss": decoder_loss}

    # -- model parameter logging --

    @staticmethod
    def _log_model_parameters(*models: tuple[str, nn.Module]) -> None:
        for name, model in models:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            print(f"\n{'=' * 60}")
            print(f"Model: {name}")
            print(f"{'=' * 60}")
            print(f"Total parameters     : {total:,}")
            print(f"Trainable parameters : {trainable:,}")
            print(f"Frozen parameters    : {frozen:,}")
            if total > 0:
                print("-" * 60)
                print(f"{'Category':<30} {'Params':>12} {'%':>7}")
                print("-" * 60)
                categories: dict[str, int] = {}
                for pname, param in model.named_parameters():
                    cat = _assign_category(pname)
                    categories[cat] = categories.get(cat, 0) + param.numel()
                for cat, count in categories.items():
                    pct = count / total * 100
                    print(f"{cat:<30} {count:>12,} {pct:>6.2f}%")
            print("-" * 60)

    # -- handler hooks (override in subclasses) --

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach per-iteration training log handler."""

        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            timer = engine.state.timer
            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()
            step_time_avg = (
                engine.state.metrics["step_time_avg"] if "step_time_avg" in engine.state.metrics else step_time
            )
            remaining_time = estimate_time_remaining(engine, step_time_avg, self.config)
            engine.set_metrics([("step_time", step_time), ("remaining_time", remaining_time)])
            metrics = engine.get_metrics(["loss", "lr"], prefix="train/")
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach COMPLETED handler that logs validation metrics."""

        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            self.wandb_logger.log(
                engine.get_metrics(["val_loss", "loss"], prefix="val/"),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {"epoch": int(trainer.state.epoch), "global_step": int(trainer.get_metric("global_step", 0))},
                step=int(trainer.get_metric("global_step", 0)),
            )

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        """Attach per-iteration running-val-loss handler."""

        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            engine.csa_op_metrics([("val_loss", engine.get_metric("loss", 0.0))], 1.0)

    # -- common handler scaffolding --

    def _limit_val_batches(self, evaluator: PretrainEngine) -> None:
        val_batches = self.config.val_batches
        if val_batches > 0:

            @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
            def _handler(engine: PretrainEngine) -> None:
                if engine.state.iteration // self.accumulation_steps >= val_batches:
                    engine.terminate()

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach Ignite event handlers for training orchestration. Subclasses override hook methods to customize."""
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        ProgressBar(mininterval=10.0).attach(trainer, ["loss"])
        trainer.state.horizon = self.config.horizon

        timer = Timer(average=False)
        timer.attach(trainer, start=Events.STARTED, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED)
        RunningAverage(output_transform=lambda _: trainer.get_metric("step_time", 0.0)).attach(trainer, "step_time_avg")

        trainer.state.timer = timer

        @trainer.on(Events.GET_BATCH_STARTED)
        def _set_horizon(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(engine.state.horizon)

        @evaluator.on(Events.GET_BATCH_STARTED)
        def _set_horizon_eval(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(trainer.state.horizon)

        self._limit_val_batches(evaluator)

        @trainer.on(Events.EPOCH_COMPLETED(every=self.config.val_interval))
        def _run_validation(engine: PretrainEngine) -> None:
            evaluator.run(self.val_loader)

        self._log_train_step(trainer, evaluator)
        self._track_batch_loss(evaluator)
        self._log_best_validation(trainer, evaluator)

        @trainer.on(Events.EPOCH_STARTED)
        def _reset_train_loss(engine: PretrainEngine) -> None:
            engine.state.metrics["loss"] = 0.0

        @trainer.on(Events.EPOCH_COMPLETED)
        def _track_best_train_loss(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            if batch_count > 0:
                engine.scale_metrics(["loss"], 1.0 / batch_count)
            epoch_loss = float(engine.get_metric("loss", float("inf")))
            best_train_loss = float(engine.get_metric("best_train_loss", float("inf")))
            if epoch_loss < best_train_loss:
                engine.set_metrics([("best_train_loss", epoch_loss)])
                self.wandb_logger.log(
                    {"train/best_train_loss": epoch_loss},
                    step=int(trainer.get_metric("global_step", 0)),
                )

    # -- public run --

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the training loop."""
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self.evaluator = evaluator
        self._attach_handlers(trainer, evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


def _assign_category(name: str) -> str:
    name_lower = name.lower()
    if "cross_attn" in name_lower:
        return "cross_attn"
    if "self_attn" in name_lower or "attn" in name_lower:
        return "self_attn"
    if "mlp" in name_lower:
        return "mlp"
    if "adaln" in name_lower:
        return "adaln"
    if "layer_norm" in name_lower or "norm" in name_lower:
        return "layer_norm"
    if "register" in name_lower or "mask_token" in name_lower:
        return "learnable_tokens"
    if "linear" in name_lower or "proj" in name_lower:
        return "linear_proj"
    return "other"


# ========== models/flow_matching_trainer.py ==========

"""Flow Matching Predictor trainer — Phase 3.

Loads a pretrained MotionHistoryEncoder from checkpoint, freezes it,
and trains a FlowMatchingPredictor using rectified flow matching in
the encoder's latent space.

Training objective: learn v_theta(z_t, t, c) such that integrating
the ODE from t=0 to t=1 transports Gaussian noise to the encoder's
latent distribution, conditioned on text c and motion history.

Key design:
  - Encoder frozen throughout (both online and EMA copies)
  - Predictor trained with separate AdamW + OneCycleLR
  - Power-schedule timestep sampling: t = u^{1/(p+1)}, p=config.t_sampling_power
  - CFG dropout: config.cfg_dropout fraction replace text with a learned null embedding
  - Rectified flow target: v* = z1 - z0, where z_t = (1-t)*z0 + t*z1
  - EMA maintained for the predictor
  - Can run standalone via FlowMatchingTrainer.run() or supply models to a combined loop

Leakage-free latent construction (requires dataset to emit separate tensors):
  - target_motion  (= batch["motion"])         : the sequence being generated
  - history_motion (= batch["history_motion"]) : frames strictly preceding the target,
                                                 zero-padded when unavailable

  z1             = EMA_encoder(target_motion,  zeros_text, return_layer_outputs=True)[:, 1:, -1, :]
  track_features = EMA_encoder(history_motion, zeros_text, return_layer_outputs=False)

  Because target_motion and history_motion are non-overlapping by construction in
  dataset.py, the encoder running on history_motion cannot see any frame that is
  part of the target being generated.  This is true regardless of whether the
  encoder is causal or bidirectional.

  Text conditioning reaches the predictor exclusively through the AdaLN text_embedding
  argument, which is also the sole location of CFG dropout.
"""


import os
import pathlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
)

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.models import (
# [internal import removed]  from utils.models.flow_matching_predictor import FlowMatchingPredictor
# [internal import removed]  from utils.models.motion_history_encoder import MotionHistoryEncoder


# ── Windows checkpoint compat ──────────────────────────────────────────────────


@contextmanager
def _windows_checkpoint_path_compat():
    """Allow checkpoints pickled with PosixPath to load on Windows."""
    original_posix_path = pathlib.PosixPath
    should_patch_posix = os.name == "nt"
    if should_patch_posix:
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        if should_patch_posix:
            pathlib.PosixPath = original_posix_path


def _torch_load_with_compat(path, map_location, weights_only):
    """Load checkpoint with Windows path compatibility."""
    with _windows_checkpoint_path_compat():
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
        finally:
            pass
    return checkpoint


# ── Latent space stats container ───────────────────────────────────────────────


class _LatentStats(nn.Module):
    """Container for latent statistics so Ignite can serialize them cleanly."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.norm_std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.initialized = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)



# ── Learned null embedding ─────────────────────────────────────────────────────


class _NullTextEmbedding(nn.Module):
    """Learned unconditional text embedding for Classifier-Free Guidance.

    Owned by the trainer (not the predictor model) so the predictor
    architecture remains unchanged. Saved in checkpoints as a separate key
    and included in the predictor optimizer's parameter group.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return null embedding expanded to (batch_size, dim)."""
        return self.weight.expand(batch_size, -1)


# ── Trainer ────────────────────────────────────────────────────────────────────


class FlowMatchingTrainer(_BaseTrainer):
    """Phase 3: Flow Matching Predictor trainer.

    Trains FlowMatchingPredictor using rectified flow matching while keeping
    the MotionHistoryEncoder fully frozen.

    Usage (standalone)::

        trainer = FlowMatchingTrainer(
            config=config,
            pretrained_checkpoint_path="checkpoints/pretrain_best_val_xxx.pt",
        )
        trainer.run()

    Usage (combined notebook loop)::

        trainer = FlowMatchingTrainer(config=config, pretrained_checkpoint_path=ckpt)
        # Access trainer.predictor, trainer.ema_predictor, trainer.null_text_embedding,
        # trainer.normalizer, trainer.ema_encoder directly from your notebook loop.
        loss = trainer.compute_flow_loss(motion, text_emb)
    """

    def __init__(
        self,
        config: Config,
        pretrained_checkpoint_path: str | Path,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.pretrained_checkpoint_path = Path(pretrained_checkpoint_path) if pretrained_checkpoint_path is not None else None
        self.wandb_project = wandb_project

        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.use_amp)
        )

        self.session_id = datetime.now(timezone(timedelta(hours=6))).strftime("%Y%m%d_%H%M%S")
        self.phase3_session_id = self.session_id
        self.resume_id: str | None = None

        # ── Load frozen encoder from checkpoint ──
        self._load_pretrained_encoder_state()

        # ── Build predictor + EMA ──
        self.predictor = FlowMatchingPredictor(config).to(self.device)
        self.ema_predictor: EMAModel[FlowMatchingPredictor] = EMAModel(
            self.predictor, decay=float(config.ema_decay)
        ).to(self.device)

        # ── Learned null text embedding for CFG ──
        self.null_text_embedding = _NullTextEmbedding(config.text_embedding_dim).to(self.device)

        # ── Instantiate CLIP Text Sequence Encoder ──
        # [internal import removed]  from utils.text_encoder import CLIPEncoder
        self.clip_encoder = CLIPEncoder().to(self.device)

        # ── Instantiate Latent statistics container ──
        self.latent_stats = _LatentStats(config.encoder_config.hidden_size).to(self.device)

        self.encoder.to(self.device)
        self.ema_encoder.to(self.device)
        self.predictor.to(self.device)

        # ── Freeze encoder completely ──
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ema_encoder.model.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.ema_encoder.model.eval()

        self._log_model_parameters(
            ("MotionHistoryEncoder (frozen)", self.encoder),
            ("FlowMatchingPredictor", self.predictor),
        )

        self.wandb_logger = None
        self._initialize()

    # ── Initialization helpers ─────────────────────────────────────────────────

    def _initialize(self) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._create_dataloaders()
        self._compute_latent_statistics()

        self.accumulation_steps = ceil(self.config.effective_batch_size / self.config.batch_size) or 1

        # Optimizer: predictor params + learned null embedding trained jointly
        self.optimizer = torch.optim.AdamW(
            list(self.predictor.parameters()) + list(self.null_text_embedding.parameters()),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.lr_scheduler = self._setup_scheduler(
            self.optimizer, num_epochs=int(self.config.get_num_epochs())
        )

        self._init_wandb(
            "phase3_flow_matching",
            extra_config={
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "predictor_params": sum(p.numel() for p in self.predictor.parameters()),
                "pretrained_checkpoint": str(self.pretrained_checkpoint_path),
                "cfg_dropout": self.config.cfg_dropout,
                "t_sampling_power": self.config.t_sampling_power,
            },
            name=self.phase3_session_id,
        )

    def _load_pretrained_encoder_state(self) -> None:
        """Load encoder (online + EMA) from a pretrain checkpoint and freeze both."""
        if self.pretrained_checkpoint_path is None:
            raise FileNotFoundError(
                "Pretrained checkpoint path is None. No pretrained checkpoint was found matching prefix 'pretrain_best_val_*.pt' in './checkpoints/pretrain'."
            )
        if not self.pretrained_checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_checkpoint_path}")

        checkpoint = _torch_load_with_compat(
            self.pretrained_checkpoint_path, self.device, weights_only=False
        )

        metadata = checkpoint.get("metadata")
        if isinstance(metadata, CheckpointMetadata):
            metadata = metadata.state_dict()
        self.pretraining_session_id = metadata.get("session_id", "unknown") if metadata else "unknown"

        config_raw = checkpoint.get("config")
        config: Config = Config()
        if isinstance(config_raw, CheckpointMetadata):
            config_raw = config_raw.state_dict()
        if isinstance(config_raw, dict):
            config.load_state_dict(config_raw)

        encoder_state = checkpoint.get("encoder")
        encoder_ema_state = checkpoint.get("encoder_ema")
        if encoder_state is None and encoder_ema_state is None:
            raise KeyError(
                f"Checkpoint {self.pretrained_checkpoint_path} does not contain "
                "encoder or encoder_ema weights"
            )

        primary_state = encoder_state if encoder_state is not None else encoder_ema_state
        ema_state = encoder_ema_state if encoder_ema_state is not None else encoder_state
        assert primary_state is not None

        self.encoder = MotionHistoryEncoder(config).to(self.device)
        self.ema_encoder = EMAModel(self.encoder, decay=float(config.ema_decay)).to(self.device)
        self.encoder.load_state_dict(primary_state)
        self.ema_encoder.load_state_dict(ema_state)

    # ── Core helpers ───────────────────────────────────────────────────────────

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps from power schedule.

        Density: p(t) ∝ t^power → inverse CDF: t = u^{1/(power+1)}.
        Concentrates samples near t→1 (harder high-noise region) where the
        flow matching loss is most informative.

        Returns:
            t: (batch_size,) float32 tensor in [0, 1].
        """
        u = torch.rand(batch_size, device=self.device)
        power = float(self.config.t_sampling_power)
        return u.pow(1.0 / (power + 1.0))

    def _apply_cfg_dropout(
        self,
        text_emb: torch.Tensor,
        batch_size: int,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Stochastically replace text with learned null embedding for CFG training.

        10% of samples use the unconditional (null) embedding so the predictor
        learns both text-conditioned and text-free trajectory distributions.
        No dropout is applied during validation.
        """
        dropout_prob = float(self.config.cfg_dropout)
        if dropout_prob <= 0.0 or not is_training:
            return text_emb
        cfg_mask = torch.rand(batch_size, device=self.device) < dropout_prob  # (B,) bool
        null_emb = self.null_text_embedding(batch_size).to(dtype=text_emb.dtype)  # (B, 512)

        # If text_emb is a sequence (B, S, 512), expand the null embedding along sequence dimension
        if text_emb.ndim == 3:
            S = text_emb.shape[1]
            null_seq_emb = null_emb.unsqueeze(1).expand(-1, S, -1)
            return torch.where(cfg_mask.unsqueeze(-1).unsqueeze(-1), null_seq_emb, text_emb)

        return torch.where(cfg_mask.unsqueeze(-1), null_emb, text_emb)

    def _compute_latent_statistics(self) -> None:
        """Compute running mean and std of the encoder's latents over the training dataset."""
        if self.latent_stats.initialized.item():
            print("Latent space statistics already loaded from checkpoint.")
            return

        print("Pre-computing latent space normalization statistics...")

        def _collect_z1(loader, num_batches):
            all_z1 = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= num_batches:
                        break
                    motion_raw = batch["motion"].to(self.device)
                    target_motion = self.normalizer.normalize(motion_raw)
                    zeros_text = torch.zeros(target_motion.shape[0], self.config.text_embedding_dim,
                                             device=self.device, dtype=target_motion.dtype)
                    raw_target = self.ema_encoder.model(target_motion, zeros_text, mask=None, return_layer_outputs=True)
                    z1 = raw_target[:, 1:, -1, :].detach()  # (B, T-1, H)
                    all_z1.append(z1.cpu())
            return torch.cat(all_z1, dim=0).flatten(0, 1)  # (N, H)

        combined_z1 = _collect_z1(self.train_loader, num_batches=20)
        latent_mean = combined_z1.mean(dim=0).to(self.device)
        latent_std = torch.clamp(combined_z1.std(dim=0).to(self.device), min=1e-5)

        # Compute per-channel std on independent held-out validation sample
        holdout_z1 = _collect_z1(self.val_loader, num_batches=5)
        normalized_holdout = (holdout_z1.to(self.device) - latent_mean) / latent_std
        norm_std = torch.clamp(normalized_holdout.std(dim=0), min=1e-5)

        self.latent_stats.mean.copy_(latent_mean)
        self.latent_stats.std.copy_(latent_std)
        if hasattr(self.latent_stats, "norm_std"):
            self.latent_stats.norm_std.copy_(norm_std)
        self.latent_stats.initialized.copy_(torch.tensor(True, device=self.device))
        print("Latent statistics computed successfully.")


    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize target latent z to have mean 0 and std 1 channel-wise."""
        mean = self.latent_stats.mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_stats.std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Denormalize latent z back to its original scale and shift."""
        mean = self.latent_stats.mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_stats.std.to(device=z.device, dtype=z.dtype)
        return z * std + mean

    # ── Latent computation (usable from external combined loops) ───────────────

    @torch.no_grad()
    def compute_target_latents(
        self,
        target_motion: torch.Tensor,
        history_motion: torch.Tensor,
        text_emb: torch.Tensor,
        history_valid_length: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frozen encoder outputs for flow matching, with no leakage.

        target_motion and history_motion are non-overlapping by construction
        (dataset.py guarantees this).  The encoder is bidirectional, so the only
        way to prevent leakage is to run it on two disjoint input sequences.

        Args:
            target_motion:  Normalised target window (B, T_target, 271).  This is the
                            sequence being generated.  Used to produce z1.
            history_motion: Normalised history window (B, T_history, 271).  Frames
                            strictly preceding the target window.  Used to produce
                            track_features.  Zero-padded at front when unavailable.
            text_emb:       Text embedding (B, D).  Used only to build a zeros tensor
                            of the right shape/dtype; NOT passed to the encoder.
            history_valid_length: Optional (B,) tensor indicating number of real history frames.

        Returns:
            z1:             Target latents (B, T_target-1, H_enc) — last encoder layer,
                            token index 1 onwards (matches decoder_trainer convention).
            track_features: History context (B, T_history, H_enc) — last encoder layer.
                            Encoder saw only history frames; no target frame information.
        """
        target_enc = self.ema_encoder.model
        target_enc.eval()

        zeros_text = torch.zeros_like(text_emb)

        # z1 from target_motion (what the predictor must learn to generate)
        raw_target = target_enc(
            target_motion, zeros_text, mask=None, return_layer_outputs=True
        )  # (B, T_target, L, H_enc)
        z1 = raw_target[:, 1:, -1, :].detach()  # (B, T_target-1, H_enc)

        # track_features from history_motion (disjoint from target — no leakage possible)
        # Text is zeroed: conditioning lives exclusively in AdaLN (text_embedding arg).
        zeros_text_hist = torch.zeros(history_motion.shape[0], text_emb.shape[1],
                                      device=history_motion.device, dtype=history_motion.dtype)
        track_features = target_enc(
            history_motion, zeros_text_hist, mask=None, return_layer_outputs=False
        ).detach()  # (B, T_history, H_enc)

        return z1, track_features

    def compute_flow_loss(
        self,
        target_motion: torch.Tensor | None = None,
        history_motion: torch.Tensor | None = None,
        text_emb: torch.Tensor | None = None,
        is_training: bool = True,
        z1: torch.Tensor | None = None,
        track_features: torch.Tensor | None = None,
        history_valid_length: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rectified flow matching loss.

        Suitable for use from external (non-Ignite) combined training loops.

        Args:
            target_motion:  Normalised target window (B, T_target, 271).
            history_motion: Normalised history window (B, T_history, 271), non-overlapping
                            with target_motion.
            text_emb:       Text embedding (B, D) or sequence (B, S, D).
            is_training:    If False, skip CFG dropout; use EMA predictor.
            z1:             Pre-computed target latents (B, T_target-1, H_enc).
                            Computed from frozen encoder if not supplied.
            track_features: Pre-computed history context (B, T_history, H_enc).
                            Computed from frozen encoder if not supplied.
            history_valid_length: Optional (B,) tensor with valid history counts.

        Returns:
            loss:  Scalar flow matching MSE loss.
            t_vec: Timestep vector (B,) for diagnostics/logging.
        """
        # Ensure text_emb has a sequence dimension (B, S, D)
        if text_emb is not None and text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(1)

        if z1 is None or track_features is None:
            if target_motion is None or history_motion is None or text_emb is None:
                raise ValueError("target_motion, history_motion, and text_emb must be provided if z1 or track_features is None.")
            # Encoder expects a 2D text embedding for shapes (B, 512).
            # We use the mean pooled text embedding to avoid breaking encoder's check.
            encoder_text_ref = text_emb.mean(dim=1)
            z1, track_features = self.compute_target_latents(target_motion, history_motion, encoder_text_ref, history_valid_length=history_valid_length)

        batch_size = z1.shape[0]

        # Apply target latent normalization channel-wise
        z1_normalized = self.normalize_latent(z1)

        z0 = torch.randn_like(z1_normalized)
        t = self._sample_timesteps(batch_size)
        t_bc = t.view(batch_size, 1, 1)

        z_t = (1.0 - t_bc) * z0 + t_bc * z1_normalized
        v_target = z1_normalized - z0

        text_with_cfg = self._apply_cfg_dropout(text_emb, batch_size, is_training=is_training)

        # Concatenate text token sequence and history context along the sequence dimension: (B, S + M, 512)
        combined_cond = torch.cat([text_with_cfg, track_features], dim=1)

        # Build boolean key-padding mask for combined_cond = [text_with_cfg, track_features]
        S_text = text_with_cfg.shape[1]
        T_hist = track_features.shape[1]
        S_cond = S_text + T_hist

        if history_valid_length is not None:
            device = track_features.device
            hist_idx = torch.arange(T_hist, device=device).unsqueeze(0)  # (1, T_hist)
            valid_start = (T_hist - history_valid_length.to(device=device)).unsqueeze(1)  # (B, 1)
            hist_mask = hist_idx >= valid_start  # (B, T_hist) bool
            text_mask = torch.ones((batch_size, S_text), dtype=torch.bool, device=device)  # (B, S_text) bool
            cond_mask = torch.cat([text_mask, hist_mask], dim=1)  # (B, S_cond) bool
            attn_mask = cond_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_cond) bool
        else:
            attn_mask = None

        # Mean pool the text sequence to get a global vector for the predictor's global AdaLN path: (B, 512)
        text_pooled = text_with_cfg.mean(dim=1)

        active_predictor = self.predictor if is_training else self.ema_predictor.model
        v_pred, _, _ = active_predictor(
            noisy_states=z_t,
            timesteps=t,
            track_features=combined_cond,       # Pass combined condition as cross-attention target
            text_embedding=text_pooled,         # Pass pooled vector for global AdaLN text projection
            output_attentions=False,
            key_padding_mask=attn_mask,
            history_states=track_features,      # Pass pure history frames for self-attention masked_cond
        )

        # Stage 3: Inverse Latent Channel Standard Deviation Loss Reweighting (using normalized-space std)
        if hasattr(self, "latent_stats") and self.latent_stats is not None and getattr(self.latent_stats, "initialized", False):
            if hasattr(self.latent_stats, "norm_std"):
                ch_std = self.latent_stats.norm_std.view(1, 1, -1).detach() + 1e-4
            else:
                ch_std = z1_normalized.detach().flatten(0, 1).std(dim=0).view(1, 1, -1) + 1e-4
            ch_weight = 1.0 / ch_std
            ch_weight = ch_weight / ch_weight.mean()  # Normalize mean weight to 1.0
            loss = ((v_pred - v_target) ** 2 * ch_weight).mean()
        else:
            loss = F.mse_loss(v_pred, v_target, reduction="mean")
        return loss, t


    # ── Ignite step functions ──────────────────────────────────────────────────

    def _train_step(
        self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one Ignite training step."""
        self.predictor.train()
        self.null_text_embedding.train()

        target_motion, history_motion, text_emb = self._prepare_batch_phase3(batch)
        hist_valid_len = batch.get("history_valid_length", None)

        with torch.amp.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            loss, _t = self.compute_flow_loss(target_motion, history_motion, text_emb, is_training=True, history_valid_length=hist_valid_len)

        self.scaler.scale(loss / self.accumulation_steps).backward()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.predictor.parameters()) + list(self.null_text_embedding.parameters()),
                max_norm=float(self.config.gradient_clip),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.ema_predictor.update(self.predictor)
            self.lr_scheduler.step()

            engine.state.metrics["global_step"] = engine.state.iteration // self.accumulation_steps

        engine.csa_op_metrics(
            [("loss", loss.detach().item())],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,
        )

        return {
            "loss": loss.detach(),
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
        }

    def _val_step(
        self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one Ignite validation step using EMA predictor."""
        self.ema_predictor.model.eval()

        target_motion, history_motion, text_emb = self._prepare_batch_phase3(batch)
        hist_valid_len = batch.get("history_valid_length", None)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss, _t = self.compute_flow_loss(target_motion, history_motion, text_emb, is_training=False, history_valid_length=hist_valid_len)

        return {
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
            "val_loss": loss,
        }

    def _prepare_batch_phase3(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and normalize target_motion, history_motion, and text_emb from a batch.

        Returns:
            target_motion:  Normalised (B, T_target, 271)
            history_motion: Normalised (B, T_history, 271)
            text_emb:       (B, S, D) sequence embeddings (dynamically computed from captions)
                            or (B, D) pooled fallback.
        """
        motion_raw  = batch["motion"].to(self.device)          # target window, raw
        history_raw = batch["history_motion"].to(self.device)  # history window, raw

        target_motion  = self.normalizer.normalize(motion_raw)
        history_motion = self.normalizer.normalize(history_raw)

        if "captions" in batch:
            text_emb = self.clip_encoder.encode_sequence(batch["captions"])
        else:
            text_raw = batch["text_clip"].to(self.device)
            text_emb = text_raw[:, -1, :] if text_raw.ndim == 3 else text_raw  # (B, D)

        return target_motion, history_motion, text_emb

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            timer = engine.state.timer
            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()
            step_time_avg = (
                engine.state.metrics["step_time_avg"]
                if "step_time_avg" in engine.state.metrics
                else step_time
            )
            remaining_time = estimate_time_remaining(engine, step_time_avg, self.config)
            engine.set_metrics([("step_time", step_time), ("remaining_time", remaining_time)])
            metrics = engine.get_metrics(["loss", "lr"], prefix="train/")
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))
            self.wandb_logger.log(
                {
                    "train/horizon": engine.state.horizon,
                    "epoch": int(engine.state.epoch),
                    "global_step": int(engine.get_metric("global_step", 0)),
                    "step_time": step_time,
                    "remaining_time": remaining_time,
                },
                step=engine.get_metric("global_step", 0),
            )

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            engine.scale_metrics(
                ["val_loss"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )
            self.wandb_logger.log(
                engine.get_metrics(["val_loss"], prefix="val/"),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {
                    "epoch": int(trainer.state.epoch),
                    "global_step": int(trainer.get_metric("global_step", 0)),
                },
                step=int(trainer.get_metric("global_step", 0)),
            )
            val_loss = float(engine.get_metric("val_loss", float("inf")))
            best_val_loss = float(engine.get_metric("best_val_loss", float("inf")))
            if val_loss < best_val_loss:
                engine.set_metrics([("best_val_loss", val_loss)])
                self.wandb_logger.log(
                    {"val/best_val_loss": val_loss},
                    step=int(trainer.get_metric("global_step", 0)),
                )

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            output = engine.state.output if isinstance(engine.state.output, dict) else {}
            engine.csa_op_metrics(
                [("val_loss", float(output.get("val_loss", 0.0)))],
                1.0,
            )

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        super()._attach_handlers(trainer, evaluator)

        checkpoint_mapping = {
            "trainer": trainer,
            "encoder": self.encoder,
            "encoder_ema": self.ema_encoder,
            "predictor": self.predictor,
            "predictor_ema": self.ema_predictor,
            "null_text_embedding": self.null_text_embedding,
            "latent_stats": self.latent_stats,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "metadata": CheckpointMetadata(
                {
                    "pretrain_id": self.pretraining_session_id,
                    "session_id": self.session_id,
                    "resume_id": self.resume_id,
                }
            ),
            "config": CheckpointMetadata(asdict(self.config)),
        }

        def _global_step_transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"phase3_best_val_{self.phase3_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=_global_step_transform,
        )
        self.val_best_checkpoint = val_best_checkpoint

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"phase3_latest_{self.phase3_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=_global_step_transform,
        )
        self.latest_checkpoint = latest_checkpoint

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval), latest_checkpoint
        )
        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)

    # ── Public run ─────────────────────────────────────────────────────────────

    def run(self, max_epochs: int | None = None) -> None:
        """Run standalone flow matching training (uses Ignite internally)."""
        print(f"Starting Phase 3 (flow matching predictor) session: {self.session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self._attach_handlers(trainer, self.evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


# ── Convenience function ───────────────────────────────────────────────────────


def train_flow_matching(
    config: Config,
    pretrained_checkpoint_path: str | Path,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel[MotionHistoryEncoder], EMAModel[FlowMatchingPredictor], Path]:
    """Train the flow matching predictor while reusing a pretrained encoder checkpoint.

    Returns:
        ema_encoder:   Frozen EMA encoder (for use in inference or further training).
        ema_predictor: Trained EMA predictor.
        checkpoint_path: Path to the best-val checkpoint saved to disk.
    """
    trainer = FlowMatchingTrainer(
        config=config,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = trainer.val_best_checkpoint.last_checkpoint
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path()
    return trainer.ema_encoder, trainer.ema_predictor, checkpoint_path


__all__ = ["FlowMatchingTrainer", "train_flow_matching"]


# ========== models/pretrain_trainer.py ==========

"""
JEPA-style Pretraining Trainer for Motion History Encoder.

Standalone, self-contained trainer using PyTorch Ignite.
Handles all model building, training loop configuration, and execution.
Designed for direct use in Kaggle notebooks with minimal boilerplate.
"""


from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
)

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.models import CheckpointMetadata, EMAModel, PretrainEngine, _BaseTrainer, estimate_time_remaining
# [internal import removed]  from utils.models.flow_matching_predictor import LatentDecoder
# [internal import removed]  from utils.models.motion_history_encoder import JepaPredictor, LinearProbe, MotionHistoryEncoder


class InterpEnum(Enum):
    """Interpolation types for ProgressScheduler."""

    NONE = "none"
    LINEAR = "linear"
    CUBIC = "cubic"


class ProgressScheduler:
    """Utility for scheduling progress through curriculum phases."""

    def __init__(self, schedule: list[tuple[float, float]]):
        progress, value = zip(*schedule)
        max_progress = max(progress)
        self.progress = [p / max_progress for p in progress]
        self.value = value

    def get_value(self, progress: float, interp: InterpEnum = InterpEnum.NONE) -> float:
        """Return progress through current curriculum phase as a float in [0, 1]."""

        return self.value[-1]


def random_span_mask(
    seq_len: int,
    num_spans: int = 2,
    min_span: int = 8,
    max_span: int = 20,
    engine: PretrainEngine | None = None,
    device: torch.device | str | None = None,
):
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    span_lengths = []

    free_intervals = [(0, seq_len)]

    for _ in range(num_spans):
        valid = []
        for idx, (left, right) in enumerate(free_intervals):
            length = right - left
            if length >= min_span:
                valid.append((idx, left, right, length))

        if not valid:
            break

        weights = torch.tensor([v[3] for v in valid], dtype=torch.float)
        choice_idx = int(torch.multinomial(weights, 1).item())

        idx, left, right, length = valid[choice_idx]
        span = int(torch.randint(min_span, min(max_span, length) + 1, ()).item())
        start = int(torch.randint(left, right - span + 1, ()).item())
        end = start + span

        mask[start:end] = True
        span_lengths.append(span)

        new_intervals = []
        for j, (free_left, free_right) in enumerate(free_intervals):
            if j != idx:
                new_intervals.append((free_left, free_right))
                continue
            if free_left < start:
                new_intervals.append((free_left, start))
            if end < free_right:
                new_intervals.append((end, free_right))
        free_intervals = new_intervals

    total_mask_len = sum(span_lengths)
    min_mask_len = min(span_lengths) if span_lengths else 0
    max_mask_len = max(span_lengths) if span_lengths else 0

    metrics = {
        "total_mask_len": total_mask_len,
        "min_mask_len": min_mask_len,
        "max_mask_len": max_mask_len,
        "num_spans": len(span_lengths),
    }

    if engine is not None:
        engine.set_metrics(list(metrics.items()))

    return mask


class PretrainTrainer(_BaseTrainer):
    """
    JEPA-style pretraining trainer with masked frame reconstruction.

    Fully self-contained:, and Ignite engine.
    Usage:
        trainer = PretrainTrainer(config=config, train_loader=train_loader, val_loader=val_loader)
        trainer.run()
    """

    def __init__(
        self,
        config: Config,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.wandb_project = wandb_project

        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.use_amp)
        )

        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config)
        self.ema_encoder: EMAModel = EMAModel(self.encoder, decay=float(config.ema_decay))
        self.jepa_predictor: JepaPredictor = JepaPredictor(config.encoder_config)
        self.decoder: LatentDecoder = LatentDecoder(config)

        self.linear_probe: LinearProbe = LinearProbe(
            hidden_size=config.encoder_config.hidden_size,
            text_embedding_dim=config.text_embedding_dim,
        ).to(self.device)

        self._log_model_parameters(
            ("MotionHistoryEncoder", self.encoder),
            ("JepaPredictor", self.jepa_predictor),
            ("LinearProbe", self.linear_probe),
            ("LatentDecoder", self.decoder),
        )

        self.wandb_logger = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components: models, optimizer, W&B."""
        config = self.config

        utc_plus_6 = timezone(timedelta(hours=6))
        self.session_id = datetime.now(utc_plus_6).strftime("%Y%m%d_%H%M%S")
        self.pretraining_session_id = self.session_id

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._create_dataloaders()

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.jepa_predictor.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.aux_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.linear_probe.parameters()) + list(self.decoder.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )

        self.encoder.to(self.device)
        self.jepa_predictor.to(self.device)
        self.linear_probe.to(self.device)
        self.decoder.to(self.device)
        self.ema_encoder.to(self.device)

        self.accumulation_steps = ceil(config.effective_batch_size / config.batch_size) or 1

        num_epochs = int(config.get_num_epochs())
        self.lr_scheduler = self._setup_scheduler(self.optimizer, num_epochs)
        self.aux_lr_scheduler = self._setup_scheduler(self.aux_optimizer, num_epochs)

        self.resume_id = None

        self._init_wandb(
            "pretrain",
            extra_config={
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "jepa_params": sum(p.numel() for p in self.jepa_predictor.parameters()),
                "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
            },
            name=self.pretraining_session_id,
        )
        if self.wandb_logger:
            print(f"Initialized W&B run with ID: {self.resume_id}")

    def _train_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""
        self.encoder.train()
        self.jepa_predictor.train()
        self.linear_probe.train()
        self.decoder.train()

        motion, text, joints = self._prepare_batch(batch)

        losses = self._compute_loss(engine, motion, joints, text)

        loss = losses["loss"]
        probe_loss = losses["probe_loss"]
        decoder_loss = losses["decoder_loss"]

        self.scaler.scale(loss / self.accumulation_steps).backward()
        self.scaler.scale(probe_loss / self.accumulation_steps).backward()
        self.scaler.scale(decoder_loss / self.accumulation_steps).backward()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.step(self.aux_optimizer)
            self.scaler.update()
            self.aux_optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)

            self.ema_encoder.update(self.encoder)
            self.lr_scheduler.step()
            self.aux_lr_scheduler.step()

            engine.state.metrics["global_step"] = engine.state.iteration // self.accumulation_steps

        return {
            "loss": loss.detach(),
            "lr": losses["lr"],
            "probe_loss": probe_loss.detach(),
            "decoder_loss": decoder_loss.detach(),
        }

    def _val_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one validation step."""
        if self.config.val_use_ema:
            self.ema_encoder.model.eval()
        else:
            self.encoder.eval()
        self.jepa_predictor.eval()
        self.linear_probe.eval()
        self.decoder.eval()

        motion, text, joints = self._prepare_batch(batch)

        with torch.no_grad():
            losses = self._compute_loss(engine, motion, joints, text)

        return {
            "lr": losses["lr"],
            "val_loss": losses["loss"],
            "val_probe_loss": losses["probe_loss"],
            "val_decoder_loss": losses["decoder_loss"],
        }

    def _compute_loss(
        self, engine: PretrainEngine, motion: torch.Tensor, joints: torch.Tensor, text: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = motion.shape

        num_spans = round(engine.state.get_schedule_value("num_spans", default=2, interp="linear"))
        min_span = self.config.pre_conf.mask_min_span
        max_span = self.config.pre_conf.mask_max_span

        mask_bool = (
            random_span_mask(seq_len, num_spans, min_span, max_span, engine).unsqueeze(0).expand(batch_size, -1)
        ).to(self.device)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            text_emb = text[:, -1, :] if text.ndim == 3 else text
            masked_context = self.encoder.forward(
                motion, torch.zeros_like(text_emb), mask=mask_bool, return_layer_outputs=True
            )

            with torch.no_grad():
                target_encoder = self.ema_encoder.model
                target_encoder.eval()
                target_context = target_encoder(
                    motion, torch.zeros_like(text_emb), mask=None, return_layer_outputs=True
                ).detach()

            _, _, L, H = masked_context.shape

            predicted = self.jepa_predictor(masked_context)

            _loss = F.smooth_l1_loss(predicted, target_context, reduction="none").mean(dim=(2, 3))
            mask_loss = (_loss * mask_bool.float()).sum() / mask_bool.sum().clamp(min=1)
            context_loss = self._context_loss(engine, _loss, mask_bool)
            loss = mask_loss + context_loss * self.config.jepa_ctx_weight

            probe_out = self.linear_probe(target_context[:, :, -1, :])
            probe_out = F.normalize(probe_out, dim=-1)
            normalized_text = F.normalize(text_emb, dim=-1)
            probe_loss = 1 - F.cosine_similarity(probe_out, normalized_text, dim=-1).mean()

            latent = target_context[:, 1:, -1, :]
            decoded = self.decoder(latent)

            decoder_losses = self._decoder_loss(engine, motion, joints, decoded)
            decoder_loss = decoder_losses["decoder_loss"]

        engine.csa_op_metrics(
            [
                ("loss", loss.detach().item()),
                ("mask_loss", mask_loss.detach().item()),
                ("context_loss", context_loss.detach().item()),
                ("probe_loss", probe_loss.detach().item()),
            ],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,
        )

        return {
            "loss": loss,
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
            "probe_loss": probe_loss,
            "decoder_loss": decoder_loss,
        }

    def _context_loss(self, engine: PretrainEngine, _loss: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask_bool.shape
        all_mask_indices = [mask_bool[b].nonzero(as_tuple=False).squeeze(1) for b in range(batch_size)]

        pos = torch.arange(seq_len, device=self.device)
        weights = torch.zeros(batch_size, seq_len, device=self.device)

        pos_exp = pos.unsqueeze(0).unsqueeze(2)
        mask_idx_exp = torch.stack(all_mask_indices).unsqueeze(1)
        distances = (pos_exp - mask_idx_exp).abs()
        min_distances = distances.min(dim=2).values
        weights = 1.0 / torch.sqrt(min_distances + 1e-8)
        weights[mask_bool] = 0.0

        unmasked_bool = ~mask_bool
        context_loss = (_loss * unmasked_bool.float() * weights.detach()).sum() / (unmasked_bool.sum().float() + 1e-8)

        return context_loss

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            timer = engine.state.timer
            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()
            step_time_avg = (
                engine.state.metrics["step_time_avg"] if "step_time_avg" in engine.state.metrics else step_time
            )
            remaining_time = estimate_time_remaining(engine, step_time_avg, self.config)
            engine.set_metrics([("step_time", step_time), ("remaining_time", remaining_time)])
            metrics = engine.get_metrics(
                [
                    "loss",
                    "mask_loss",
                    "context_loss",
                    "probe_loss",
                    "decoder_loss",
                    "loss_root",
                    "loss_yaw",
                    "loss_ric",
                    "loss_vel",
                    "loss_joint",
                    "lr",
                ],
                prefix="train/",
            )
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))
            self.wandb_logger.log(
                {
                    "train/horizon": engine.state.horizon,
                    "epoch": int(engine.state.epoch),
                    "global_step": int(engine.get_metric("global_step", 0)),
                    "step_time": step_time,
                    "remaining_time": remaining_time,
                },
                step=engine.get_metric("global_step", 0),
            )

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            engine.scale_metrics(
                ["val_loss", "val_decoder_loss", "val_probe_loss"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )
            self.wandb_logger.log(
                engine.get_metrics(
                    [
                        "val_loss",
                        "val_decoder_loss",
                        "val_probe_loss",
                    ],
                    prefix="val/",
                ),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {"epoch": int(trainer.state.epoch), "global_step": int(trainer.get_metric("global_step", 0))},
                step=int(trainer.get_metric("global_step", 0)),
            )
            val_loss = float(engine.get_metric("val_loss", float("inf")))
            best_val_loss = float(engine.get_metric("best_val_loss", float("inf")))
            if val_loss < best_val_loss:
                engine.set_metrics([("best_val_loss", val_loss)])
                self.wandb_logger.log({"val/best_val_loss": val_loss}, step=int(trainer.get_metric("global_step", 0)))
            eval_loss = float(engine.get_metric("val_decoder_loss", float("inf")))
            best_eval_loss = float(engine.get_metric("best_eval_loss", float("inf")))
            if eval_loss < best_eval_loss:
                engine.set_metrics([("best_eval_loss", eval_loss)])
                self.wandb_logger.log({"val/best_eval_loss": eval_loss}, step=int(trainer.get_metric("global_step", 0)))

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            # Ignite stores process_function return dict in state.output, not state.metrics
            output = engine.state.output if isinstance(engine.state.output, dict) else {}
            engine.csa_op_metrics(
                [
                    ("val_loss", float(output.get("val_loss", 0.0))),
                    ("val_decoder_loss", float(output.get("val_decoder_loss", 0.0))),
                    ("val_probe_loss", float(output.get("val_probe_loss", 0.0))),
                ],
                1.0,
            )

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        super()._attach_handlers(trainer, evaluator)

        checkpoint_mapping = {
            "trainer": trainer,
            "encoder": self.encoder,
            "linear_probe": self.linear_probe,
            "jepa_predictor": self.jepa_predictor,
            "encoder_ema": self.ema_encoder,
            "decoder": self.decoder,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "aux_lr_scheduler": self.aux_lr_scheduler,
            "metadata": CheckpointMetadata({"session_id": self.pretraining_session_id, "resume_id": self.resume_id}),
            "config": CheckpointMetadata(asdict(self.config)),
        }

        global_step_transform = self._create_global_step_transform(trainer)

        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_best_val_{self.pretraining_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=global_step_transform,
        )

        eval_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_best_eval_{self.pretraining_session_id}",
            score_function=lambda engine: -float(engine.state.metrics["val_decoder_loss"]),
            score_name="eval_loss",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=global_step_transform,
        )

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_latest_{self.pretraining_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=global_step_transform,
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval), latest_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, eval_best_checkpoint)

    @staticmethod
    def _create_global_step_transform(trainer: PretrainEngine):
        def _transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        return _transform

    def run(self, max_epochs: int | None = None) -> None:
        print(f"Starting pretraining session: {self.pretraining_session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self._attach_handlers(trainer, self.evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


def train_pretrain(
    config: Config,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel, None, Path]:
    """
    Train encoder with JEPA objective in a single call.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        normalizer: Feature normalizer
        wandb_project: Optional W&B project name
        max_epochs: Override number of epochs (uses config default if None)

    Returns:
        Tuple of (ema_encoder, ema_jepa_predictor, checkpoint_path)
    """
    trainer = PretrainTrainer(
        config=config,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = config.checkpoint_dir / "pretrain_latest.pt"
    return trainer.ema_encoder, None, checkpoint_path


# ========== models/finetune_trainer.py ==========

"""Decoder-only finetuning trainer for motion latent reconstruction.

Loads a pretrained MotionHistoryEncoder and its EMA copy from checkpoint,
freezes them, and trains a fresh LatentDecoder on the same reconstruction loss
used during pretraining.
"""


import os
import pathlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import torch
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
)

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.models import CheckpointMetadata, EMAModel, PretrainEngine, _BaseTrainer, estimate_time_remaining
# [internal import removed]  from utils.models.flow_matching_predictor import LatentDecoder
# [internal import removed]  from utils.models.motion_history_encoder import MotionHistoryEncoder


@contextmanager
def _windows_checkpoint_path_compat():
    """Allow checkpoints pickled with PosixPath to load on Windows."""
    original_posix_path = pathlib.PosixPath
    should_patch_posix = os.name == "nt"
    if should_patch_posix:
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        if should_patch_posix:
            pathlib.PosixPath = original_posix_path


def _torch_load_with_compat(path, map_location, weights_only):
    """Load checkpoint with Windows path and module path compatibility."""
    with _windows_checkpoint_path_compat():
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
        finally:
            pass
    return checkpoint


class FinetuneTrainer(_BaseTrainer):
    """Decoder-only finetuning trainer."""

    def __init__(
        self,
        config: Config,
        pretrained_checkpoint_path: str | Path,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.pretrained_checkpoint_path = Path(pretrained_checkpoint_path)
        self.wandb_project = wandb_project

        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.use_amp)
        )

        self.session_id = datetime.now(timezone(timedelta(hours=6))).strftime("%Y%m%d_%H%M%S")
        self.finetuning_session_id = self.session_id
        self.resume_id: str | None = None

        self._load_pretrained_encoder_state()

        self.decoder = LatentDecoder(config).to(self.device)
        self.ema_decoder: EMAModel[LatentDecoder] = EMAModel(self.decoder, decay=float(config.ema_decay)).to(
            self.device
        )

        self.encoder.to(self.device)
        self.ema_encoder.to(self.device)
        self.decoder.to(self.device)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ema_encoder.model.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.ema_encoder.model.eval()

        self._log_model_parameters(
            ("MotionHistoryEncoder", self.encoder),
            ("LatentDecoder", self.decoder),
        )

        self.wandb_logger = None

        self._initialize()

    def _initialize(self) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._create_dataloaders()

        self.accumulation_steps = ceil(self.config.effective_batch_size / self.config.batch_size) or 1

        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.lr_scheduler = self._setup_scheduler(self.optimizer, num_epochs=int(self.config.get_num_epochs()))

        self._init_wandb(
            "finetune_decoder",
            extra_config={
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
                "pretrained_checkpoint": str(self.pretrained_checkpoint_path),
            },
            name=self.finetuning_session_id,
        )

    def _load_pretrained_encoder_state(self) -> None:
        if not self.pretrained_checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_checkpoint_path}")

        checkpoint = _torch_load_with_compat(self.pretrained_checkpoint_path, self.device, weights_only=False)

        metadata = checkpoint.get("metadata")
        if isinstance(metadata, CheckpointMetadata):
            metadata = metadata.state_dict()
        self.pretraining_session_id = metadata.get("session_id", "unknown") if metadata else "unknown"

        config_raw = checkpoint.get("config")
        config: Config = Config()
        if isinstance(config_raw, CheckpointMetadata):
            config_raw = config_raw.state_dict()
        if isinstance(config_raw, dict):
            config.load_state_dict(config_raw)

        encoder_state = checkpoint.get("encoder")
        encoder_ema_state = checkpoint.get("encoder_ema")
        if encoder_state is None and encoder_ema_state is None:
            raise KeyError(
                f"Checkpoint {self.pretrained_checkpoint_path} does not contain encoder or encoder_ema weights"
            )

        primary_state = encoder_state if encoder_state is not None else encoder_ema_state
        ema_state = encoder_ema_state if encoder_ema_state is not None else encoder_state
        assert primary_state is not None

        self.encoder = MotionHistoryEncoder(config).to(self.device)
        self.ema_encoder = EMAModel(self.encoder, decay=float(config.ema_decay)).to(self.device)

        self.encoder.load_state_dict(primary_state)
        self.ema_encoder.load_state_dict(ema_state)

    def _train_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""
        self._decoder = self.decoder
        self._decoder.train()

        motion, text, joints = self._prepare_batch(batch)

        losses = self._compute_loss(engine, motion, joints, text)

        loss = losses["loss"]

        self.scaler.scale(loss / self.accumulation_steps).backward()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.ema_decoder.update(self.decoder)
            self.lr_scheduler.step()

            engine.state.metrics["global_step"] = engine.state.iteration // self.accumulation_steps

        return {
            "loss": loss.detach(),
            "lr": losses["lr"],
        }

    def _val_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one validation step."""
        self._encoder = self.ema_encoder.model
        self._encoder.eval()
        self._decoder = self.ema_decoder.model
        self._decoder.eval()

        motion, text, joints = self._prepare_batch(batch)

        with torch.no_grad():
            losses = self._compute_loss(engine, motion, joints, text)

        return {
            "lr": losses["lr"],
            "val_loss": losses["loss"],
        }

    def _compute_loss(
        self, engine: PretrainEngine, motion: torch.Tensor, joints: torch.Tensor, text: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = motion.shape

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            text_emb = text[:, -1, :] if text.ndim == 3 else text

            with torch.no_grad():
                target_encoder = self.ema_encoder.model
                target_encoder.eval()
                target_context = target_encoder(
                    motion, torch.zeros_like(text_emb), mask=None, return_layer_outputs=True
                ).detach()

            latent = target_context[:, 1:, -1, :]
            decoded = self._decoder(latent)

            decoder_losses = self._decoder_loss(engine, motion, joints, decoded)
            loss = decoder_losses["decoder_loss"]
            decoder_loss = decoder_losses["decoder_loss"]

        engine.csa_op_metrics(
            [("loss", loss.detach().item())],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,
        )

        return {
            "loss": loss,
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
        }

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            timer = engine.state.timer
            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()
            step_time_avg = (
                engine.state.metrics["step_time_avg"] if "step_time_avg" in engine.state.metrics else step_time
            )
            remaining_time = estimate_time_remaining(engine, step_time_avg, self.config)
            engine.set_metrics([("step_time", step_time), ("remaining_time", remaining_time)])
            metrics = engine.get_metrics(
                [
                    "decoder_loss",
                    "loss_root_y",
                    "loss_root_xz",
                    "loss_yaw",
                    "loss_vel",
                    "loss_joint",
                    "lr",
                ],
                prefix="train/",
            )
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))
            self.wandb_logger.log(
                {
                    "train/horizon": engine.state.horizon,
                    "epoch": int(engine.state.epoch),
                    "global_step": int(engine.get_metric("global_step", 0)),
                    "step_time": step_time,
                    "remaining_time": remaining_time,
                },
                step=engine.get_metric("global_step", 0),
            )

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            engine.scale_metrics(
                ["val_loss", "val_decoder_loss"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )
            self.wandb_logger.log(
                engine.get_metrics(
                    ["val_loss", "val_decoder_loss", "decoder_loss"],
                    prefix="val/",
                ),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {"epoch": int(trainer.state.epoch), "global_step": int(trainer.get_metric("global_step", 0))},
                step=int(trainer.get_metric("global_step", 0)),
            )
            val_loss = float(engine.get_metric("val_loss", float("inf")))
            best_val_loss = float(engine.get_metric("best_val_loss", float("inf")))
            if val_loss < best_val_loss:
                engine.set_metrics([("best_val_loss", val_loss)])
                self.wandb_logger.log({"val/best_loss": val_loss}, step=int(trainer.get_metric("global_step", 0)))
            eval_loss = float(engine.get_metric("val_decoder_loss", float("inf")))
            best_eval_loss = float(engine.get_metric("best_eval_loss", float("inf")))
            if eval_loss < best_eval_loss:
                engine.set_metrics([("best_eval_loss", eval_loss)])
                self.wandb_logger.log({"val/best_eval_loss": eval_loss}, step=int(trainer.get_metric("global_step", 0)))

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            engine.csa_op_metrics(
                [
                    ("val_loss", engine.get_metric("val_loss", 0.0)),
                    ("val_decoder_loss", engine.get_metric("decoder_loss", 0.0)),
                ],
                1.0,
            )

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        super()._attach_handlers(trainer, evaluator)

        checkpoint_mapping = {
            "trainer": trainer,
            "encoder": self.encoder,
            "encoder_ema": self.ema_encoder,
            "decoder": self.decoder,
            "decoder_ema": self.ema_decoder,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "metadata": CheckpointMetadata(
                {"pretrain_id": self.pretraining_session_id, "session_id": self.session_id, "resume_id": self.resume_id}
            ),
            "config": CheckpointMetadata(asdict(self.config)),
        }

        def _global_step_transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"finetune_best_val_{self.finetuning_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["loss"]),
            score_name="val_loss",
            global_step_transform=_global_step_transform,
        )
        self.val_best_checkpoint = val_best_checkpoint

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"finetune_latest_{self.finetuning_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=_global_step_transform,
        )
        self.latest_checkpoint = latest_checkpoint

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval), latest_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        print(f"Starting finetuning session: {self.session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self._attach_handlers(trainer, self.evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


def train_finetune(
    config: Config,
    pretrained_checkpoint_path: str | Path,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel[MotionHistoryEncoder], EMAModel[LatentDecoder], Path]:
    """Train the decoder while reusing a pretrained encoder checkpoint."""

    trainer = FinetuneTrainer(
        config=config,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = trainer.val_best_checkpoint.last_checkpoint
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path()
    return trainer.ema_encoder, trainer.ema_decoder, checkpoint_path


__all__ = ["FinetuneTrainer", "train_finetune"]
