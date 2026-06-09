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
from torch.amp.grad_scaler import GradScaler
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


# ========== config.py ==========

"""
Configuration file for Human Motion Animation Generation Pipeline.

This configuration uses the custom 271D feature format:
- Input: 271D feature vectors from motion_utils.py
- Output: Joint positions (nframe, 22, 3) → BVH files

Feature Layout (271D) - Updated per normalization plan:
- [0:3]   Root height Y, Root velocity X, Root velocity Z (velocity form)
- [3:69]  22 RIC positions (22 * 3)
- [69:201] 22 6D rotations (22 * 6)
- [201:267] 22 local velocities (22 * 3)
- [267:271] Foot contacts (4D)

Note: Root X,Z are stored as velocities for autoregressive stability.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class FlowMatchingPredictorConfig:
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    track_dimensionality: int = 3
    global_cond_dim: int = 512  # clip embedding : 512D
    head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class MotionHistoryEncoderConfig:
    frame_feature_dim: int = 271
    text_embedding_dim: int = 512
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    dropout: float = 0.1
    per_joint_output_dim: int = 64
    joint_count: int = 22
    num_registers: int = 2
    text_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "MotionHistoryEncoderConfig.hidden_size must be divisible by "
                f"num_attention_heads, got {self.hidden_size} and "
                f"{self.num_attention_heads}."
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                "MotionHistoryEncoderConfig requires an even per-head dimension "
                f"for RoPE, got hidden_size={self.hidden_size}, "
                f"num_attention_heads={self.num_attention_heads}, head_dim={head_dim}."
            )
        self.intermediate_size = max(int(self.intermediate_size), self.hidden_size)


@dataclass
class Config:
    """Configuration class for the motion generation pipeline."""

    # Device settings
    device: Any = "cuda"  # "cuda" or "cpu" or torch.device
    seed: int = 42

    # Data paths and directories
    dataset_path: Path = Path("./dataset/humanml3d-subset")
    output_path: Path = Path("./output")
    checkpoint_dir: Path = Path("./checkpoints")

    checkpoint_interval: int = 50  # Save checkpoint every N epochs

    # Motion format settings (271D custom format from motion_utils.py)
    motion_dim: int = 271  # Custom 271D feature dimension
    num_joints: int = 22  # Number of joints in skeleton
    joint_dim: int = 3  # 3D coordinates per joint
    max_motion_length: int = 200  # Maximum motion length in frames
    fps: int = 20  # Frames per second

    # Feature dimension subsetting for training
    feature_dims: tuple = (
        slice(0, 3),  # root height Y, velocity X, velocity Z (3D)
        slice(3, 69),  # RIC positions (22*3 = 66D)
        slice(69, 201),  # 6D rotations (22*6 = 132D)
        slice(201, 267),  # local velocities (22*3 = 66D)
        slice(267, 271),  # foot contacts (4D)
    )

    # =============================================================================
    # MotionHistoryEncoder Configuration
    # =============================================================================
    encoder_config: MotionHistoryEncoderConfig = field(
        default_factory=lambda: MotionHistoryEncoderConfig(
            frame_feature_dim=271,
            text_embedding_dim=512,
            hidden_size=512,
            intermediate_size=2 * 512,
            num_hidden_layers=4,
            num_attention_heads=16,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            attention_bias=True,
            attention_dropout=0.1,
            mlp_bias=True,
            dropout=0.1,
            per_joint_output_dim=64,
            joint_count=22,
            text_scale=1.0,
        )
    )

    # =============================================================================
    # FlowMatchingPredictor Configuration (Spatial-Only with Flow Matching Timestep)
    # =============================================================================
    # Uses new FlowMatchingPredictorConfig dataclass for structured configuration
    # Time embedding is handled internally via SinusoidalEmbedder(hidden_size)
    predictor_config: FlowMatchingPredictorConfig = field(
        default_factory=lambda: FlowMatchingPredictorConfig(
            hidden_size=128,
            intermediate_size=4 * 128,
            num_hidden_layers=3,
            num_attention_heads=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_bias=True,
            attention_dropout=0.1,
            mlp_bias=True,
            track_dimensionality=3,
            head_dim=None,
        )
    )

    text_embedding_dim = 512  # Dimension of text embeddings for conditioning

    # Training settings
    effective_batch_size: int = 400
    batch_size: int = 200
    learning_rate: float = 0.5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 30.0
    ema_decay: float = 0.999

    # Curriculum learning settings
    # Set to None to disable curriculum (use fixed horizon from horizon field)
    curriculum: Optional[list[dict[str, int]]] = field(
        default_factory=lambda: [
            {"horizon": 5, "epochs": 100},
            {"horizon": 10, "epochs": 200},
            {"horizon": 20, "epochs": 300},
            {"horizon": 40, "epochs": 1000},
        ]
    )

    horizon: int = 40  # Maximum/target horizon for training
    _num_epochs: int = 200

    jepa_ctx_weight: float = 0.2

    # CFG (Classifier-Free Guidance) settings
    cfg_dropout: float = 0.1  # Dropout probability for CFG

    # Training-time timestep sampling
    t_sampling_mode: str = "power"  # "uniform" or "power"
    t_sampling_power: float = 3.0  # Power-law exponent k in p(t)=(k+1)t^k
    t_sampling_power_warmup_fraction: float = 1  # Fraction of training used to ramp k from 0 to target

    use_fk: bool = False  # Whether to compute FK loss during training
    # Rollout scheduling settings
    rollout_prob_start: float = 0.1  # Rollout probability at first epoch
    rollout_prob_end: float = 0.3  # Rollout probability at final epoch
    rollout_warmup_fraction: float = 0.15  # Fraction of training with rollout disabled before schedule starts
    rollout_block_len_start: int = 1  # Rollout block length at schedule start
    rollout_block_len_end: int = 4  # Rollout block length at schedule end
    rollout_integration_steps: int = 3  # Number of ODE integration steps for rollout branch
    rollout_subset_fraction: float = 0.25  # Fraction of batch for rollout branch
    rollout_loss_weight: float = 0.25  # Weight of rollout-conditioned loss branch
    rollout_block_len_bias_power: float = 2.0  # Power > 1 biases sampled rollout lengths toward the scheduled max

    use_consistency_loss: bool = True  # Enable endpoint consistency loss after no-grad rollout
    consistency_loss_t_threshold: float = 0.5  # Only apply consistency loss for t > threshold
    consistency_loss_weight: float = 10  # Weight for consistency loss in total loss

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Inference settings
    num_inference_steps: int = 20  # Number of flow matching steps
    inference_t_schedule_power: float = 3.0  # End-bias power p in t=1-(1-s)^p for inference ODE time boundaries
    guidance_scale: float = 1.0  # CFG scale for inference

    # Validation settings
    val_interval: int = 5  # Run validation every N epochs
    val_batches: int = 20  # Number of validation batches per run (-1 for all)
    val_use_ema: bool = True  # Use EMA models for validation
    save_best_val: bool = True  # Save separate checkpoint for best validation loss

    # Profiling settings
    enable_profiling: bool = False  # Enable timing instrumentation
    timing_log_interval: int = 100  # Log timings every N batches
    tqdm_log_per_batch: bool = False  # Show per-batch tqdm progress during training

    unit_length = 5

    enable_linear_probe: bool = True
    probe_loss_weight: float = 1.0

    def __post_init__(self):
        self.t_sampling_mode = str(self.t_sampling_mode).lower()
        if self.t_sampling_mode not in {"uniform", "power"}:
            raise ValueError(f"t_sampling_mode must be 'uniform' or 'power', got {self.t_sampling_mode!r}")
        self.t_sampling_power = max(0.0, float(self.t_sampling_power))
        self.t_sampling_power_warmup_fraction = min(
            max(float(self.t_sampling_power_warmup_fraction), 0.0),
            1.0,
        )
        self.tqdm_log_per_batch = bool(self.tqdm_log_per_batch)
        self.rollout_warmup_fraction = min(
            max(float(self.rollout_warmup_fraction), 0.0),
            1.0 - 1e-6,
        )
        self.rollout_block_len_start = max(1, int(self.rollout_block_len_start))
        self.rollout_block_len_end = max(
            self.rollout_block_len_start,
            int(self.rollout_block_len_end),
        )
        self.rollout_subset_fraction = min(
            max(float(self.rollout_subset_fraction), 0.0),
            1.0,
        )
        self.rollout_loss_weight = max(float(self.rollout_loss_weight), 0.0)
        self.rollout_block_len_bias_power = float(self.rollout_block_len_bias_power)
        if self.rollout_block_len_bias_power <= 1.0:
            raise ValueError(
                f"rollout_block_len_bias_power must be greater than 1, got {self.rollout_block_len_bias_power}"
            )
        self.inference_t_schedule_power = float(self.inference_t_schedule_power)
        if self.inference_t_schedule_power <= 0.0:
            raise ValueError(f"inference_t_schedule_power must be positive, got {self.inference_t_schedule_power}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # self.encoder_config = MotionHistoryEncoderConfig(
        #     frame_feature_dim=271,
        #     text_embedding_dim=512,
        #     hidden_size=192,
        #     intermediate_size=2 * 192,
        #     num_hidden_layers=5,
        #     num_attention_heads=8,
        #     hidden_act="silu",
        #     layer_norm_eps=1e-05,
        #     attention_bias=True,
        #     attention_dropout=0.1,
        #     mlp_bias=True,
        #     dropout=0.1,
        #     per_joint_output_dim=64,
        #     joint_count=22,
        #     text_scale=1.0,
        # )

        # self.predictor_config = FlowMatchingPredictorConfig(
        #     hidden_size=128,
        #     intermediate_size=4 * 128,
        #     num_hidden_layers=3,
        #     num_attention_heads=4,
        #     hidden_act="silu",
        #     rms_norm_eps=1e-06,
        #     attention_bias=True,
        #     attention_dropout=0.1,
        #     mlp_bias=True,
        #     track_dimensionality=3,
        #     head_dim=None,
        # )

    def get_num_epochs(self) -> int:
        """Return the total number of training epochs, accounting for curriculum."""
        return self.curriculum[-1]["epochs"] if self.curriculum else self._num_epochs

    def to_dict(self) -> dict:
        """Export the configuration as a serializable dictionary."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value) and not isinstance(value, type):
                return {k: _convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return {k: _convert(v) for k, v in self.__dict__.items()}


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

    @property
    def embedding_dim(self) -> int:
        """Output dimension of the CLIP text model."""
        return self.model.config.hidden_size


# ========== dataset.py ==========

"""
Dataset loading and Text2Motion dataset implementation.

Handles loading HumanML3D dataset with text-motion pairs.
"""

import torch
import numpy as np
from os.path import join as pjoin
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
                with open(
                    pjoin(str(text_dir), name + ".txt"), "r", encoding="utf-8"
                ) as f:
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
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
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
            except Exception as e:
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
            for i in tqdm(
                range(0, len(missing_captions), batch_size), desc="Encoding Texts"
            ):
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

    def __getitem__(self, item) -> Tuple[str, torch.Tensor, torch.Tensor, int, torch.Tensor, str]:
        """
        Returns a single sample from the dataset.
        GUARANTEES: All returned tensors have shape (max_motion_length, features)

        NOTE: Returns RAW (unnormalized) features. Normalization should be done
        externally using FeatureNormalizer before passing to models.
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

        # ===== CONVERT TO TENSORS =====
        motion = torch.from_numpy(motion.copy()).float()  # (T, 271) - RAW features
        joints = torch.from_numpy(joints.copy()).float()  # (T, 22, 3)

        # ===== NO NORMALIZATION HERE =====
        # Normalization is handled externally via FeatureNormalizer
        # motion = (motion - self.mean) / self.std  # REMOVED

        # ===== PAD OR TRUNCATE TO MAX_MOTION_LENGTH =====
        target_len = self.current_horizon
        current_len = original_length

        if current_len < target_len:
            # Pad with zeros
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
        elif current_len > target_len:
            start_idx = random.randint(0, current_len - self.current_horizon)
            motion = motion[start_idx : start_idx + self.current_horizon]
            joints = joints[start_idx : start_idx + self.current_horizon]

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
        # motion shape: (target_len, 271) - full 271D features (used as both motion and history_features)
        # valid_length is the number of real frames before zero-padding, capped at target_len.
        sample_id = self.name_list[idx]

        return caption, motion, joints, valid_length, text_embedding, sample_id

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
    batch: List[Tuple[str, torch.Tensor, torch.Tensor, int, torch.Tensor, str]],
) -> Dict[str, Any]:
    """
    Collate function for Text2MotionDataset.
    Expects each sample to be a tuple:
      - caption: str
      - motion: (max_T, 271) torch.Tensor - full 271D features
      - joints: (max_T, J, 3) torch.Tensor
      - length: int number of valid frames before padding, capped at the returned horizon
    - text_embedding: (1, 512) torch.Tensor - pooled CLIP embeddings
    - sample_id: str sample identifier
    """
    # Lists of items
    captions = [b[0] for b in batch]
    motions_list = [b[1] for b in batch]
    joints_list = [b[2] for b in batch]
    lengths = [b[3] for b in batch]
    text_embs_list = [b[4] for b in batch]
    sample_ids = [b[5] for b in batch]

    # Stack tensors directly
    motion_batch = torch.stack(motions_list, dim=0)  # (B, T, 271)
    joints_batch = torch.stack(joints_list, dim=0)  # (B, T, J, 3)
    length_batch = torch.tensor(lengths, dtype=torch.long)  # (B,)

    text_emb_batch = torch.stack(text_embs_list, dim=0)  # (B, 1, 512)

    return {
        "captions": captions,
        "sample_ids": sample_ids,
        "motion": motion_batch,
        "joints": joints_batch,
        "lengths": length_batch,
        "text_clip": text_emb_batch,
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Any
# [internal import removed]  from utils.motion_utils import T2M_KINEMATIC_CHAIN


def probe_camera_state(ax) -> dict:
    """
    Print and return matplotlib 3D camera + scene state for use when
    configuring a Plotly backend to match this view.

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
    max_range = max(xr, yr, zr)

    state = dict(
        elev=elev,
        azim=azim,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        x_range=xr,
        y_range=yr,
        z_range=zr,
        # Plotly aspectratio dict derived from data ranges
        plotly_aspectratio=dict(
            x=xr / max_range,
            y=yr / max_range,
            z=zr / max_range,
        ),
        # Plotly camera eye derived from elev/azim
        plotly_camera_eye=dict(
            x=float(1.75 * np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim))),
            y=float(1.75 * np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim))),
            z=float(1.75 * np.sin(np.deg2rad(elev))),
        ),
    )

    print("=" * 50)
    print(f"  elev       : {elev:.2f} deg")
    print(f"  azim       : {azim:.2f} deg")
    print(f"  xlim       : [{xlim[0]:.3f}, {xlim[1]:.3f}]  range={xr:.3f}")
    print(f"  ylim       : [{ylim[0]:.3f}, {ylim[1]:.3f}]  range={yr:.3f}")
    print(f"  zlim       : [{zlim[0]:.3f}, {zlim[1]:.3f}]  range={zr:.3f}")
    print(f"  plotly aspectratio : {state['plotly_aspectratio']}")
    print(f"  plotly camera eye  : {state['plotly_camera_eye']}")
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
):
    import imageio
    import io
    import base64
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
    return HTML(
        f'<video controls width="600"><source src="data:video/mp4;base64,{b64}"></video>'
    )


def plot_3d_motion_plotly(
    motion: np.ndarray,
    fps: float = 20,
    radius: float = 1.0,
    title: str = "Motion Visualization",
    follow_root: bool = False,
):
    """
    Create an optimized 3D animation of motion joint positions using Plotly,
    specifically tuned for Jupyter notebooks and matching the Matplotlib probe.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")

    n_frames = len(motion)

    pos_min = motion.min(axis=(0, 1))
    pos_max = motion.max(axis=(0, 1))

    # Set matching ranges based on matplotlib logic
    if follow_root:
        root0 = motion[0, 0, :]
        x_range = [root0[0] - radius, root0[0] + radius]
        y_range = [root0[2] - radius, root0[2] + radius]
        z_range = [pos_min[1], pos_max[1] + 0.5]
    else:
        x_range = [pos_min[0] - radius, pos_max[0] + radius]
        y_range = [pos_min[2] - radius, pos_max[2] + radius]
        z_range = [pos_min[1], pos_max[1] + 0.5]

    xr = x_range[1] - x_range[0]
    yr = y_range[1] - y_range[0]
    zr = z_range[1] - z_range[0]
    max_range = max(xr, yr, zr)

    # Exactly matching the probed aspect ratio
    aspectratio = dict(
        x=xr / max_range,
        y=yr / max_range,
        z=zr / max_range,
    )

    # Exactly matching the probed camera eye (elev=15, azim=65, dist=1.75)
    elev = 15.0
    azim = 65.0
    dist = 1.75
    camera_eye = dict(
        x=dist * np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
        y=dist * np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
        z=dist * np.sin(np.deg2rad(elev)),
    )

    colors = ["#2980b9", "#c0392b", "#27ae60", "#f39c12", "#8e44ad"]

    # Pre-build frames efficiently by updating only coordinate data
    initial_data = []
    for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
        joints = motion[0, c_indices, :]
        initial_data.append(
            go.Scatter3d(
                x=joints[:, 0],
                y=joints[:, 2],
                z=joints[:, 1],
                mode="lines+markers",
                marker=dict(size=2.5, color=colors[i]),
                line=dict(width=3, color=colors[i]),
                name=f"Chain {i}",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    frames = []
    for frame_idx in range(n_frames):
        frame_data = []
        for i, c_indices in enumerate(T2M_KINEMATIC_CHAIN):
            joints = motion[frame_idx, c_indices, :]
            frame_data.append(
                go.Scatter3d(x=joints[:, 0], y=joints[:, 2], z=joints[:, 1])
            )

        layout_update = {}
        if follow_root:
            root = motion[frame_idx, 0, :]
            layout_update = dict(
                scene=dict(
                    xaxis=dict(range=[root[0] - radius, root[0] + radius]),
                    yaxis=dict(range=[root[2] - radius, root[2] + radius]),
                )
            )

        frames.append(
            go.Frame(data=frame_data, name=str(frame_idx), layout=layout_update)
        )

    fig = go.Figure(data=initial_data, frames=frames)

    fig.update_layout(
        title=title,
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(
                title="X (Side)",
                range=x_range,
                autorange=False,
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgray",
                zerolinecolor="gray",
            ),
            yaxis=dict(
                title="Z (Forward)",
                range=y_range,
                autorange=False,
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgray",
                zerolinecolor="gray",
            ),
            zaxis=dict(
                title="Y (Height)",
                range=z_range,
                autorange=False,
                showbackground=True,
                backgroundcolor="white",
                gridcolor="lightgray",
                zerolinecolor="gray",
            ),
            aspectmode="manual",
            aspectratio=aspectratio,
            camera=dict(
                eye=camera_eye,
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),
            ),
        ),
        # in a scope where n_frames and fps are defined
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.0,
                y=0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    # Play / Pause toggle
                    dict(
                        label="▶/⏸",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1000 / fps, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0, easing="linear"),
                            ),
                        ],
                    ),
                    # Restart from beginning
                    dict(
                        label="⟲",
                        method="animate",
                        args=[
                            [str(0)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Frame: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0, easing="linear"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [str(k)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(k),
                        method="animate",
                    )
                    for k in range(n_frames)
                ],
            )
        ],
        margin=dict(l=0, r=20, t=40, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


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
        probe: If True, print camera + scene state (for Plotly parity work)
        backend: Visualization backend - "plotly" (default) or "matplotlib"
    """
    fps = fps / skip_frames
    motion_subsampled = joint_positions[::skip_frames]

    if backend == "plotly":
        try:
            fig = plot_3d_motion_plotly(
                motion_subsampled,
                fps=fps,
                radius=radius,
                title=title,
            )

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                html_path = save_path.with_suffix(".html")
                fig.write_html(str(html_path))
                print(f"Saved interactive animation to {html_path}")

            if notebook:
                return fig
            return fig

        except ImportError:
            print("Plotly not available, falling back to matplotlib...")
            backend = "matplotlib"

    if backend == "matplotlib":
        html = plot_3d_motion(
            motion_subsampled, radius=radius, fps=fps, title=title, probe=probe
        )

        # if save_path:
        #     # imageio needs a real file for saving; re-render to disk
        #     save_path.parent.mkdir(parents=True, exist_ok=True)
        #     import imageio
        #     # render frames again to save_path directly
        #     # simplest: call a thin wrapper that writes to file
        #     _plot_3d_motion_to_file(
        #         motion_subsampled, save_path, radius=radius, fps=fps, title=title
        #     )
        #     print(f"Saved animation to {save_path}")

        return html


def compare_motions(
    generated_joints: np.ndarray,
    ground_truth_joints: np.ndarray,
    save_path: Optional[Path] = None,
    backend: str = "plotly",
) -> None:
    """
    Compare generated motion with ground truth.
    """
    # Simply call the visualization logic with the chosen backend
    visualize_motion(
        generated_joints,
        title="Generated vs Ground Truth",
        save_path=save_path,
        backend=backend,
    )


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
from datetime import datetime
from typing import Optional, Dict, Any

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
        resume: Optional[str] = None,
        kaggle_secret_name: str = "WANDB_API_KEY",
        enabled: bool = True,
    ):
        self.project = project
        self.config = config or {}
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.resume_id = resume_id
        self.resume = resume

        # Auto-generate run name from datetime if not provided
        if name is None:
            self.name = "motion-generation-buet"
        else:
            self.name = name

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
                resume=self.resume,
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

        if api_key:
            try:
                wandb.login(key=api_key)
                print("[WandbLogger] Authenticated successfully.")
            except Exception as e:
                print(f"[WandbLogger] Authentication failed: {e}")
        else:
            print(
                "[WandbLogger] No API key found. Using existing login or anonymous mode."
            )

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

import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

# [internal import removed]  from utils.config import Config, FlowMatchingPredictorConfig, MotionHistoryEncoderConfig
# [internal import removed]  from utils.models import FlowMatchingPredictor, MotionHistoryEncoder
# [internal import removed]  from utils.motion_utils import (


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

    def __init__(self, d_model: int, d_cond: int):
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
        *,
        bias: bool,
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
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


def _build_inference_time_boundaries(
    num_steps: int,
    *,
    power: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build end-biased ODE boundaries on [0, 1] using t = 1 - (1 - s)^p."""
    steps = max(1, int(num_steps))
    s = torch.linspace(0.0, 1.0, steps=steps + 1, device=device, dtype=dtype)
    return 1.0 - (1.0 - s).pow(power)


def integrate_flow_ode(
    *,
    predictor: FlowMatchingPredictor,
    track_features: torch.Tensor,
    current_frame_features: torch.Tensor,
    text_embedding: torch.Tensor,
    num_steps: int,
    time_schedule_power: float = 2.0,
    initial_state: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    unconditional_text_embedding: Optional[torch.Tensor] = None,
    unconditional_track_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Integrate the inference-time flow ODE using an end-biased power grid and Heun.
    """
    batch_size = track_features.shape[0]
    device = track_features.device
    dtype = track_features.dtype
    if time_schedule_power <= 0.0:
        raise ValueError(f"time_schedule_power must be positive, got {time_schedule_power}")

    if initial_state is None:
        x_t = torch.randn(
            (batch_size, predictor.flow_dim),
            device=device,
            dtype=dtype,
        )
    else:
        if initial_state.shape != (batch_size, predictor.flow_dim):
            raise ValueError(
                f"Expected initial_state shape ({batch_size}, {predictor.flow_dim}), got {tuple(initial_state.shape)}"
            )
        x_t = initial_state.to(device=device, dtype=dtype)

    tau = _build_inference_time_boundaries(
        num_steps,
        power=float(time_schedule_power),
        device=device,
        dtype=dtype,
    )

    use_cfg = float(guidance_scale) != 1.0 and (
        unconditional_text_embedding is not None or unconditional_track_features is not None
    )
    if unconditional_text_embedding is not None:
        if unconditional_text_embedding.shape != text_embedding.shape:
            raise ValueError(
                "Expected unconditional_text_embedding shape "
                f"{tuple(text_embedding.shape)}, got "
                f"{tuple(unconditional_text_embedding.shape)}"
            )
        unconditional_text_embedding = unconditional_text_embedding.to(
            device=device,
            dtype=text_embedding.dtype,
        )
    if unconditional_track_features is not None:
        if unconditional_track_features.shape != track_features.shape:
            raise ValueError(
                "Expected unconditional_track_features shape "
                f"{tuple(track_features.shape)}, got "
                f"{tuple(unconditional_track_features.shape)}"
            )
        unconditional_track_features = unconditional_track_features.to(
            device=device,
            dtype=track_features.dtype,
        )

    def _predict_velocity(
        noisy_features: torch.Tensor,
        timesteps_batch: torch.Tensor,
    ) -> torch.Tensor:
        cond_velocity = predictor(
            track_features=track_features,
            noisy_features=noisy_features,
            timesteps=timesteps_batch,
            current_frame_features=current_frame_features,
            text_embedding=text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        if not use_cfg:
            return cond_velocity

        cfg_text_embedding = (
            unconditional_text_embedding if unconditional_text_embedding is not None else text_embedding
        )
        cfg_track_features = (
            unconditional_track_features if unconditional_track_features is not None else track_features
        )
        uncond_velocity = predictor(
            track_features=cfg_track_features,
            noisy_features=noisy_features,
            timesteps=timesteps_batch,
            current_frame_features=current_frame_features,
            text_embedding=cfg_text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        return uncond_velocity + float(guidance_scale) * (cond_velocity - uncond_velocity)

    for step in range(tau.shape[0] - 1):
        t_start = tau[step]
        t_end = tau[step + 1]
        dt = t_end - t_start

        t_start_batch = t_start.expand(batch_size)
        k1 = _predict_velocity(x_t, t_start_batch)

        x_euler = x_t + dt * k1
        t_end_batch = t_end.expand(batch_size)
        k2 = _predict_velocity(x_euler, t_end_batch)

        x_t = x_t + 0.5 * dt * (k1 + k2)

    return x_t


@dataclass
class TemporalLayerCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


@dataclass
class TemporalCacheState:
    layers: List[TemporalLayerCache]


class HumanMotionGenerator:
    """
    Top-level wrapper for the Human Motion Generation pipeline.
    Integrates MotionHistoryEncoder (Context) and FlowMatchingPredictor (Spatial Generation).

    Updated to use 271D features and proper reduced-state → 271D conversion for autoregressive generation.
    """

    def __init__(
        self,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
        config: Config,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalizer = normalizer
        self.config = config

    def eval(self) -> "HumanMotionGenerator":
        """Set models to evaluation mode."""
        self.encoder.eval()
        self.predictor.eval()
        return self

    def train(self, mode: bool = True) -> "HumanMotionGenerator":
        """Set models to training mode."""
        self.encoder.train(mode)
        self.predictor.train(mode)
        return self

    def parameters(self):
        """Yield parameters from both encoder and predictor."""
        for p in self.encoder.parameters():
            yield p
        for p in self.predictor.parameters():
            yield p

    def to(self, device):
        """Move models to device."""
        self.encoder = self.encoder.to(device)
        self.predictor = self.predictor.to(device)
        return self

    def generate_sequence(
        self,
        text: Union[str, List[str], torch.Tensor],  # type: ignore
        num_frames: int = 200,
        num_steps: int = 10,
        horizon: int | None = None,
        input_positions: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        guidance_drop_text: bool = True,
        guidance_drop_context: bool = False,
        dataset_type: str = "t2m",
        use_fk=True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate n consecutive animation frames autoregressively.

        Uses absolute global joint positions as the autoregressive state and
        generated_positions_to_271d for incremental feature extraction. This ensures
        O(n) complexity and Markov-safe autoregressive generation.

        Args:
            text: Text prompt(s) - str, List[str], or pre-encoded tensor (B, 1, 512)
            num_frames: Number of frames to generate
            num_steps: Number of flow matching ODE steps
            horizon: Number of recent frames to use for context encoding
            input_positions: Optional initial global positions (B, N, 22, 3) or (B, 22, 3)
            total_duration: Optional duration tensor (not used)
            guidance_scale: CFG strength used during inference
            guidance_drop_text: Zero the predictor text embedding in the CFG branch
            guidance_drop_context: Zero the encoder context in the CFG branch
            dataset_type: Dataset type for feature extraction
            use_fk: Whether to use FK positions for relative shift computation
        Returns:
            position_history: (B, N+num_frames, 22, 3) - Absolute global joint positions including initial history
            feature_history: (B, N+num_frames, 271) - 271D features derived from position history
            relative_shift_history: (B, N+num_frames, 22, 3) - Relative shifts between frames
        """
        self.eval()
        with torch.no_grad():
            # Handle text encoding
            if isinstance(text, str):
                # Single string - encode and use
                # [internal import removed]  from utils.text_encoder import CLIPEncoder

                clip_encoder = CLIPEncoder()
                text = clip_encoder(text)  # (1, 1, 512)
                B = 1
            elif isinstance(text, list):
                # List of strings - encode all
                # [internal import removed]  from utils.text_encoder import CLIPEncoder

                clip_encoder = CLIPEncoder()
                text: torch.Tensor = clip_encoder(text)  # (B, 1, 512)
                B = text.shape[0]
            else:
                # Already a tensor
                if text.ndim != 3 or text.shape[1] != 1:
                    raise ValueError(f"Pre-encoded text must have shape (B, 1, 512); got {tuple(text.shape)}")
                B = text.shape[0]
            device = next(self.parameters()).device
            text = text.to(device=device)

            # ========================================
            # History Initialization
            # ========================================
            # Keep absolute global positions as the primary AR state.
            # Feature history is derived incrementally from position history.

            if input_positions is None:
                # Cold start from a zero pose frame.
                joint_count = int(self.config.num_joints)
                joint_dim = int(self.config.joint_dim)
                position_history = torch.zeros(
                    (B, 1, joint_count, joint_dim),
                    device=device,
                )
                feature_history = sequence_joints_to_features(
                    position_history, dataset_type=dataset_type
                )  # (B, 1, 271)
            else:
                input_positions = input_positions.to(device=device)
                if input_positions.ndim == 3:
                    # Single frame (B, N, 3)
                    seed_positions = input_positions.unsqueeze(1).clone()  # (B, 1, N, 3)
                elif input_positions.ndim == 4:
                    # Sequence (B, T, N, 3)
                    seed_positions = input_positions.clone()  # (B, T, N, 3)
                else:
                    raise ValueError("input_positions must be shape (B, N, 3) or (B, T, N, 3)")

                position_history = seed_positions
                # Seeded: convert global positions to 271D feature history
                feature_history = sequence_joints_to_features(seed_positions, dataset_type=dataset_type)  # (B, T, 271)

            fk_offsets = get_fk_offsets(position_history) if use_fk else None  # (B, 22, 3)

            feature_history = (
                self.normalizer.normalize(feature_history) if self.normalizer is not None else feature_history
            )

            relative_shift_history = torch.zeros(
                (B, 1, self.config.num_joints, self.config.joint_dim),
                device=device,
            )

            if position_history.shape[1] > 1:
                relative_shift_history = torch.cat(
                    [
                        relative_shift_history,
                        position_history[:, 1:] - position_history[:, :-1],
                    ],
                    dim=1,
                )

            text_emb = text[:, 0, :]
            use_cfg = float(guidance_scale) != 1.0
            if use_cfg and not (guidance_drop_text or guidance_drop_context):
                raise ValueError(
                    "guidance_scale requires at least one unconditional branch input; "
                    "set guidance_drop_text and/or guidance_drop_context."
                )
            predictor_uncond_text_emb = torch.zeros_like(text_emb) if use_cfg and guidance_drop_text else None
            frame_buffer = feature_history[:, :-1] if feature_history.shape[1] > 1 else None
            cache_state: Optional[TemporalCacheState] = None

            for frame_idx in range(num_frames):
                # ========================================
                # Step A: Extract last frame from position history
                # ========================================
                current_positions = position_history[:, -1]  # (B, 22, 3)
                current_frame = feature_history[:, -1]

                # ========================================
                # Step B: Encode context from last horizon frames
                # Feature history buffer is maintained outside the encoder.
                # ========================================
                if frame_buffer is not None:
                    if horizon is not None:
                        frame_buffer = frame_buffer[:, -horizon - 1 :]

                context_cond, frame_buffer, cache_state = self.encoder.step(
                    current_frame,
                    text_emb,
                    frame_buffer=frame_buffer,
                    cache_state=cache_state,
                )
                predictor_uncond_context = torch.zeros_like(context_cond) if use_cfg and guidance_drop_context else None
                current_frame_features = extract_prev_frame_features(
                    current_frame,
                    normalizer=self.normalizer,
                    normalize_output=self.normalizer is not None,
                )

                # ========================================
                # Step C: Flow matching ODE loop
                # x_t starts as random noise in normalized reduced flow space.
                # ========================================
                x_t = integrate_flow_ode(
                    predictor=self.predictor,
                    track_features=context_cond,
                    current_frame_features=current_frame_features,
                    text_embedding=text_emb,
                    num_steps=num_steps,
                    time_schedule_power=self.config.inference_t_schedule_power,
                    guidance_scale=guidance_scale,
                    unconditional_text_embedding=predictor_uncond_text_emb,
                    unconditional_track_features=predictor_uncond_context,
                )

                # ========================================
                # Step E: Convert reduced-state prediction -> positions -> 271D (incremental)
                # ========================================
                flow_output_raw = self.normalizer.denormalize_flow_output(x_t) if self.normalizer is not None else x_t
                current_frame_raw = (
                    self.normalizer.denormalize(current_frame) if self.normalizer is not None else current_frame
                )
                new_positions = flow_output_to_positions(
                    flow_output_raw,
                    prev_root_pos=current_positions[:, 0],
                    prev_root_rot_6d=current_frame_raw[:, 69:75],
                )
                relative_shift = new_positions - current_positions
                new_frame, _, fk_positions = generated_positions_to_271d(
                    new_positions=new_positions,
                    prev_positions=current_positions,
                    dataset_type=dataset_type,
                    normalizer=self.normalizer,
                    fk_offsets=fk_offsets,
                )  # (B, 271), (B, 3), (B, 22, 3)

                if fk_positions is not None:
                    new_positions = fk_positions  # Override with FK-corrected positions if available
                    relative_shift = new_positions - current_positions  # Recompute relative shift after FK correction

                # ========================================
                # Step F: Update tracker and history
                # ========================================
                position_history = torch.cat([position_history, new_positions.unsqueeze(1)], dim=1)  # (B, T+1, 22, 3)
                feature_history = torch.cat([feature_history, new_frame.unsqueeze(1)], dim=1)  # (B, N+1, 271)
                relative_shift_history = torch.cat(
                    [relative_shift_history, relative_shift.unsqueeze(1)], dim=1
                )  # (B, T, 22, 3)

                if (frame_idx + 1) % 50 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            return position_history, feature_history, relative_shift_history

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Config,  # type: ignore
        device: str = "cpu",
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> "HumanMotionGenerator":
        """
        Load the generator from a checkpoint file.
        Prefers EMA weights if available.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Config object with model configuration
            device: Device to load model on
            normalizer: Optional FeatureNormalizer for raw feature normalization
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        with _windows_checkpoint_path_compat():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "config" in checkpoint:
            config: Config = checkpoint["config"]

        encoder = MotionHistoryEncoder(config.encoder_config).to(device)

        # Initialize Flow Matching Predictor with new config-based interface
        predictor_config = config.predictor_config
        feature_size = config.encoder_config.hidden_size

        predictor = FlowMatchingPredictor(
            feature_size=feature_size,
            config=predictor_config,
        ).to(device)

        # Load weights (Prefer EMA)
        if "encoder_ema" in checkpoint and "predictor_ema" in checkpoint:
            print("Loading EMA weights for generation...")
            encoder.load_state_dict(checkpoint["encoder_ema"])
            predictor.load_state_dict(checkpoint["predictor_ema"])
        else:
            print("Loading standard weights (EMA not found)...")
            encoder.load_state_dict(checkpoint["encoder"])
            predictor.load_state_dict(checkpoint["predictor"])

        encoder.to(device)
        predictor.to(device)
        encoder.eval()
        predictor.eval()

        return cls(encoder, predictor, config, normalizer=normalizer)


# Export all public symbols defined in this module.
__all__ = sorted(  # pyright: ignore[reportUnsupportedDunderAll]
    name for name, obj in globals().items() if not name.startswith("_") and getattr(obj, "__module__", None) == __name__
)


# ========== models/motion_history_encoder.py ==========

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# [internal import removed]  from utils.config import MotionHistoryEncoderConfig
# [internal import removed]  from utils.models import AdaLN, GatedMLP, TemporalCacheState, TemporalLayerCache, TemporalRoPEAttention


class EncoderRoPEAttention(TemporalRoPEAttention):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__(config)
        self.num_registers = config.num_registers
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"TemporalRoPEAttention requires an even per-head dimension, got head_dim={self.head_dim}."
            )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = config.attention_dropout

    def forward(self, hidden_states: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
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

        self.self_attn = TemporalRoPEAttention(config)

        self.adaln_mlp = AdaLN(d_model=config.hidden_size, d_cond=config.text_embedding_dim)

        self.mlp = EncoderMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_emb: torch.Tensor,
        is_causal: bool = True,
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
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.frame_projection = nn.Linear(config.frame_feature_dim, config.hidden_size, bias=True)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.register_tokens = nn.Parameter(torch.empty(config.num_registers, config.hidden_size))
        self.mask_token = nn.Parameter(torch.empty(1, config.hidden_size))

        self._init_weights()

    def _init_weights(self):
        # Register tokens:
        # slightly larger init so they participate in attention early
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)

        # Mask token:
        # slightly smaller helps continuous motion stability
        nn.init.normal_(self.mask_token, mean=0.0, std=0.01)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # ViT-style transformer init
                nn.init.trunc_normal_(module.weight, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _empty_cache_state(self) -> TemporalCacheState:
        return TemporalCacheState(layers=[TemporalLayerCache() for _ in range(self.config.num_hidden_layers)])

    def forward(
        self,
        motion_seq: torch.Tensor,
        text_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_layer_outputs: bool = False,
        is_causal: bool = True,
    ) -> torch.Tensor:
        if motion_seq.ndim != 3 or motion_seq.shape[-1] != self.config.frame_feature_dim:
            raise ValueError(
                f"Expected motion_seq shape (B, T, {self.config.frame_feature_dim}), got {tuple(motion_seq.shape)}"
            )
        if text_emb.ndim != 2 or text_emb.shape[-1] != self.config.text_embedding_dim:
            raise ValueError(
                f"Expected text_emb shape (B, {self.config.text_embedding_dim}), got {tuple(text_emb.shape)}"
            )
        if text_emb.shape[0] != motion_seq.shape[0]:
            raise ValueError(
                "Batch size mismatch between motion_seq and text_emb: "
                f"{tuple(motion_seq.shape)} vs {tuple(text_emb.shape)}"
            )

        if mask is not None:
            if mask.ndim != 2 or mask.shape != motion_seq.shape[:2]:
                raise ValueError(
                    "Expected mask shape (B, T) matching motion_seq, got "
                    f"{tuple(mask.shape)} vs {tuple(motion_seq.shape[:2])}"
                )

        # --------------------------

        batch_size, seq_len, _ = motion_seq.shape
        if seq_len == 0:
            raise ValueError("Expected motion_seq with at least one timestep.")

        hidden_states = self.frame_projection(motion_seq)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            hidden_states = torch.where(mask, self.mask_token, hidden_states)

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
            all_hidden_states.append(hidden_states)
            return torch.stack(all_hidden_states, dim=2)  # (B, N, L, H)

        return hidden_states  # (B, N, H)

    def step(
        self,
        x_t: torch.Tensor,
        text_emb: torch.Tensor,
        frame_buffer: Optional[torch.Tensor],
        cache_state: Optional[TemporalCacheState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, TemporalCacheState]:
        if x_t.ndim != 2 or x_t.shape[-1] != self.config.frame_feature_dim:
            raise ValueError(f"Expected x_t shape (B, {self.config.frame_feature_dim}), got {tuple(x_t.shape)}")

        if frame_buffer is None:
            next_frame_buffer = x_t.unsqueeze(1)
        else:
            if frame_buffer.ndim != 3 or frame_buffer.shape[-1] != self.config.frame_feature_dim:
                raise ValueError(
                    "Expected frame_buffer shape "
                    f"(B, T, {self.config.frame_feature_dim}), got {tuple(frame_buffer.shape)}"
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

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 2,
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

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class JepaPredictor(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.z_proj = nn.Linear(config.hidden_size // 4, config.hidden_size)

        self.input_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.blocks = nn.ModuleList([JepaMLPBlock(config) for _ in range(2)])

        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.output_head = nn.Linear(config.hidden_size, config.hidden_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, motion_history_emb: torch.Tensor) -> torch.Tensor:
        if motion_history_emb.ndim != 2 or motion_history_emb.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Expected motion_history_emb shape "
                f"(B, {self.config.hidden_size}), got {tuple(motion_history_emb.shape)}"
            )

        batch_size = motion_history_emb.shape[0]

        z = torch.randn(batch_size, self.z_proj.in_features, device=motion_history_emb.device)

        z_proj = self.z_proj(z)
        combined = torch.cat([motion_history_emb, z_proj], dim=-1)
        x = self.input_proj(combined)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        output = self.output_head(x)

        return output


# ========== models/flow_matching_predictor.py ==========

from typing import List, Optional, Tuple

import torch
from torch import nn

# [internal import removed]  from utils.config import Config, FlowMatchingPredictorConfig
# [internal import removed]  from utils.models import AdaLN, GatedMLP, TemporalLayerCache, TemporalRoPEAttention


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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
    def __init__(self, config: Config) -> None:
        super().__init__(config.predictor_config)
        self.config = config

        self.k_proj = nn.Linear(config.encoder_config.hidden_size, config.predictor_config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.encoder_config.hidden_size, config.predictor_config.hidden_size, bias=True)

        self.kv_cache = TemporalLayerCache()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def attn_weights(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N, H)
        encoder_hidden_states: torch.Tensor,  # (B, M, H_enc)
        _encoder_cache: Optional[TemporalLayerCache] = None,
        output_attentions: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        query = self._reshape_heads(self.q_proj(hidden_states))

        encoder_cache = self.kv_cache if _encoder_cache is None else _encoder_cache

        if encoder_cache.key is None or encoder_cache.value is None:
            key = self._reshape_heads(self.k_proj(encoder_hidden_states))
            value = self._reshape_heads(self.v_proj(encoder_hidden_states))
            encoder_cache.key = key
            encoder_cache.value = value

        query = self._reshape_heads(self.q_proj(hidden_states))
        key = encoder_cache.key
        value = encoder_cache.value

        cos, sin = self._build_rope(
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query = self._apply_rope(query, cos, sin)

        attn_output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        if output_attentions:
            attn_weights = self.attn_weights(query, key, value)
            return attn_output, attn_weights

        return attn_output


class PredictorLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        pred_config = config.predictor_config

        self.adaln_attn = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.cross_attn = PredictorRopeCrossAttention(config)

        self.adaln_mlp = AdaLN(d_model=pred_config.hidden_size, d_cond=pred_config.hidden_size * 2)

        self.mlp = PredictorMLP(pred_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        adaln_cond: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states, attn_gate = self.adaln_attn(hidden_states, adaln_cond)
        hidden_states, attn_weights = self.cross_attn(
            hidden_states, encoder_hidden_states, output_attentions=output_attentions
        )
        hidden_states = residual + attn_gate * hidden_states

        residual = hidden_states
        hidden_states, mlp_gate = self.adaln_mlp(hidden_states, adaln_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mlp_gate * hidden_states

        if output_attentions:
            return hidden_states, attn_weights

        return hidden_states, None


class FlowMatchingPredictor(nn.Module):
    def __init__(
        self,
        config: Config,  # Model configuration
        **kwargs,
    ):
        super().__init__()
        self.config = config
        pred_config = config.predictor_config

        self.text_proj = nn.Linear(config.text_embedding_dim, pred_config.hidden_size, bias=True)

        # Time embedding for denoising timestep
        self.time_embedder = SinusoidalEmbedder(pred_config.hidden_size)

        # Transformer layers with AdaLN
        self.layers = nn.ModuleList([PredictorLayer(config) for _ in range(pred_config.num_hidden_layers)])

        self.latent_proj = nn.Linear(pred_config.hidden_size, config.encoder_config.hidden_size, bias=True)
        # Output prediction head
        self.output_adaln = nn.Sequential(
            nn.Linear(pred_config.hidden_size * 2, config.encoder_config.hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Linear(config.encoder_config.hidden_size * 2, config.encoder_config.hidden_size * 2, bias=True),
        )
        self.output_norm = nn.LayerNorm(config.encoder_config.hidden_size)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        noisy_states: torch.Tensor,  # (B, N, H_enc) - noised reduced-state features
        timesteps: torch.Tensor,  # (B,) or (B, 1) - denoising timesteps in [0,1]
        encoder_hidden_states: torch.Tensor,  # (B, M, H_enc) - from MotionHistoryEncoder
        text_embedding: torch.Tensor,  # (B, F) - Text embedding for global conditioning
        output_attentions: bool = False,
        **kwargs,  # Ignore attention_mask, position_ids, etc.
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Build static per-joint kinematic tokens once and reuse across layers.
        B, N, H = noisy_states.shape

        text_cond = self.text_proj(text_embedding)  # (B, H)
        time_cond = self.time_embedder(timesteps.squeeze(-1) if timesteps.dim() > 1 else timesteps)  # (B, H)

        adaln_cond = torch.cat([text_cond, time_cond], dim=-1)  # (B, 2H)

        # 4. Transformer processing (NO ATTENTION MASKING)
        all_cross_attns: list[torch.Tensor] = []

        hidden_states = noisy_states

        for layer_idx, layer in enumerate(self.layers):
            # Bounded signed gate allows add/subtract structural prior per layer.
            hidden_states = hidden_states

            hidden_states, attn_weights = layer(
                hidden_states,
                encoder_hidden_states,
                adaln_cond=adaln_cond,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_cross_attns.append(attn_weights)

        hidden_states = self.latent_proj(hidden_states)  # (B, N, H_enc)

        output_shift, output_scale = self.output_adaln(adaln_cond).chunk(2, dim=-1)
        output_shift = output_shift.unsqueeze(1)  # (B, 1, H_enc)
        output_scale = output_scale.unsqueeze(1)  # (B, 1, H_enc)
        flow_prediction = self.output_norm(output_shift + output_scale * hidden_states)

        return (
            flow_prediction,
            all_cross_attns if output_attentions else None,
        )


# ========== models/pretrain_trainer.py ==========

"""
JEPA-style Pretraining Trainer for Motion History Encoder.

Standalone, self-contained trainer using PyTorch Ignite.
Handles all model building, training loop configuration, and execution.
Designed for direct use in Kaggle notebooks with minimal boilerplate.
"""


import copy
import sys
from typing import Any, Dict, Tuple, Generic, TypeVar, Union, Mapping as MappingABC
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events, State
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    TerminateOnNan,
    global_step_from_engine,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import RunningAverage

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.models import MotionHistoryEncoder
# [internal import removed]  from utils.models.motion_history_encoder import JepaPredictor, LinearProbe
# [internal import removed]  from utils.motion_utils import FeatureNormalizer
# [internal import removed]  from utils.wandb_logger import WandbLogger


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
        if progress <= self.progress[0]:
            return self.value[0]

        for i in range(1, len(self.progress)):
            if progress <= self.progress[i]:
                if interp == InterpEnum.NONE:
                    return self.value[i - 1]
                elif interp == InterpEnum.LINEAR:
                    ratio = (progress - self.progress[i - 1]) / (
                        self.progress[i] - self.progress[i - 1]
                    )
                    return self.value[i - 1] + ratio * (
                        self.value[i] - self.value[i - 1]
                    )
                elif interp == InterpEnum.CUBIC:
                    ratio = (progress - self.progress[i - 1]) / (
                        self.progress[i] - self.progress[i - 1]
                    )
                    ratio_cubic = (
                        3 * ratio**2 - 2 * ratio**3
                    )  # Smooth cubic interpolation
                    return self.value[i - 1] + ratio_cubic * (
                        self.value[i] - self.value[i - 1]
                    )
                else:
                    raise ValueError(f"Unsupported interpolation type: {interp}")

        return self.value[-1]


class PretrainState(State):
    """Custom Ignite State for JEPA pretraining."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = 40
        self.metrics: Dict[str, Any] = {
            "global_step": 0,
            "best_train_loss": float("inf"),
        }

    def epoch_progress(self, config: Config) -> float:
        """Return progress through current epoch as a float in [0, 1]."""
        return (
            self.epoch / float(config.get_num_epochs())
            if config.get_num_epochs() > 0
            else 0.0
        )


class PretrainEngine(Engine):
    """Custom Ignite Engine for JEPA pretraining."""

    def __init__(self, process_function: Any) -> None:
        super().__init__(process_function)
        self.state = PretrainState()


T = TypeVar("T", bound=nn.Module)


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


class PretrainTrainer:
    """
    JEPA-style pretraining trainer with masked frame reconstruction.

    Fully self-contained: builds models, optimizer, and Ignite engine.
    Usage:
        trainer = PretrainTrainer(config=config, train_loader=train_loader, val_loader=val_loader)
        trainer.run()
    """

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: FeatureNormalizer | None = None,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.wandb_project = wandb_project

        # Device and AMP setup
        self.device = torch.device(config.device)
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # Models - built internally
        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config.encoder_config)
        self.jepa_predictor: JepaPredictor = JepaPredictor(config.encoder_config)

        # EMA models
        self.ema_encoder: EMAModel = EMAModel(
            self.encoder, decay=float(config.ema_decay)
        )
        self.ema_jepa: EMAModel = EMAModel(
            self.jepa_predictor, decay=float(config.ema_decay)
        )

        probe_hidden = getattr(config.encoder_config, "hidden_size", 512)
        probe_text_dim = getattr(config.encoder_config, "text_embedding_dim", 512)
        self.linear_probe: LinearProbe = LinearProbe(
            hidden_size=probe_hidden,
            text_embedding_dim=probe_text_dim,
        ).to(self.device)

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.jepa_predictor.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.probe_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.linear_probe.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.scaler: GradScaler = GradScaler("cuda", enabled=self.use_amp)

        # W&B logger
        self.wandb_logger: WandbLogger | None = None

        self.accumulation_steps = (
            config.effective_batch_size // config.batch_size
        ) or 1
        # Initialize all objects
        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components: models, optimizer, W&B."""
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move to device
        self.encoder.to(self.device)
        self.jepa_predictor.to(self.device)
        self.ema_encoder.to(self.device)
        self.ema_jepa.to(self.device)

        # Setup W&B
        if self.wandb_project:
            self.wandb_logger = WandbLogger(
                project=self.wandb_project,
                config={
                    "lr": float(self.config.learning_rate),
                    "weight_decay": float(self.config.weight_decay),
                    "ema_decay": float(self.config.ema_decay),
                    "batch_size": self.train_loader.batch_size,
                    "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                    "jepa_params": sum(
                        p.numel() for p in self.jepa_predictor.parameters()
                    ),
                    "phase": "pretrain",
                },
            )

    def _train_step(
        self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""
        # Prepare batch
        motion = batch["motion"].to(self.device)
        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)

        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError(
                f"Expected motion (B, torch.Tensor, 271), got {tuple(motion.shape)}"
            )

        batch_size, seq_len, _ = motion.shape
        num_masked = max(1, int(seq_len * 0.25))

        # Build mask
        mask_indices = torch.stack(
            [
                torch.randperm(seq_len, device=self.device)[:num_masked]
                for _ in range(batch_size)
            ]
        )
        mask_bool = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=self.device
        )
        for b in range(batch_size):
            mask_bool[b, mask_indices[b]] = True

        # Forward pass
        self.encoder.train()
        self.jepa_predictor.train()
        self.linear_probe.train()

        self.optimizer.zero_grad(set_to_none=True)
        self.probe_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            text_emb = text[:, 0, :] if text.ndim == 3 else text
            masked_context = self.encoder(
                motion, torch.zeros_like(text_emb), mask=mask_bool, return_all=True
            )

            with torch.no_grad():
                target_encoder = self.ema_encoder.model
                target_encoder.eval()
                target_context = target_encoder(
                    motion, torch.zeros_like(text_emb), mask=None, return_all=True
                ).detach()

            # Extract masked tokens
            b_idx = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                .expand(-1, num_masked)
            )
            masked_tokens = masked_context[b_idx, mask_indices, :].reshape(
                batch_size * num_masked, -1
            )
            target_masked = target_context[b_idx, mask_indices, :].reshape(
                batch_size * num_masked, -1
            )

            predicted = self.jepa_predictor(masked_tokens)
            mask_loss = F.smooth_l1_loss(predicted, target_masked)

            token_diff = F.smooth_l1_loss(
                masked_context, target_context, reduction="none"
            ).mean(dim=-1)

            # Compute distance of each position to nearest masked position
            # pos: (1, seq_len, 1), mask_idx_exp: (B, 1, num_masked)
            # distances: (B, seq_len, num_masked)
            pos = torch.arange(seq_len, device=self.device)[
                None, :, None
            ]  # (1, seq_len, 1)
            mask_idx_exp = mask_indices.unsqueeze(1)  # (B, 1, num_masked)
            distances = torch.abs(
                pos - mask_idx_exp
            )  # (1, seq_len, 1) - (B, 1, num_masked) -> (B, seq_len, num_masked)
            min_distances = distances.min(dim=2).values  # (B, seq_len)

            weights = 1.0 / torch.sqrt(min_distances + 1.0)  # (B, seq_len)

            unmasked_bool = ~mask_bool  # (B, seq_len)
            context_loss = (token_diff * unmasked_bool.float() * weights).sum()
            context_loss = context_loss / (unmasked_bool.sum().float() + 1e-8)

            loss = mask_loss + context_loss * self.config.jepa_ctx_weight

            probe_out = self.linear_probe(target_context)
            probe_out = F.normalize(probe_out, dim=-1)
            normalized_text = F.normalize(text_emb, dim=-1)
            probe_loss = (
                1 - F.cosine_similarity(probe_out, normalized_text, dim=-1).mean()
            )

        engine.state.metrics["mask_loss"] = float(mask_loss.detach().item())
        engine.state.metrics["context_loss"] = float(context_loss.detach().item())

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scaler.scale(probe_loss).backward()
            self.scaler.step(self.probe_optimizer)
            self.scaler.update()

            # EMA update
            self.ema_encoder.update(self.encoder)
            self.ema_jepa.update(self.jepa_predictor)

            # Update engine state
            loss_val = float(loss.detach().item())
            engine.state.metrics["global_step"] = (
                engine.state.iteration // self.accumulation_steps
            )
            engine.state.metrics["best_train_loss"] = min(
                float(engine.state.metrics.get("best_train_loss", float("inf"))),
                loss_val,
            )

        return {
            "loss": loss.detach(),
            "lr": torch.tensor(float(self.config.learning_rate), device=self.device),
            "probe_loss": probe_loss.detach(),
        }

    def _val_step(
        self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one validation step."""
        motion = batch["motion"].to(self.device)
        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)

        self.encoder.eval()
        self.linear_probe.eval()

        with torch.no_grad():
            text_emb = text[:, 0, :] if text.ndim == 3 else text
            hidden_states = self.encoder(
                motion, torch.zeros_like(text_emb), return_all=True
            ).detach()
            probe_out = self.linear_probe(hidden_states)
            probe_out = F.normalize(probe_out, dim=-1)
            normalized_text = F.normalize(text_emb, dim=-1)
            probe_loss = (
                1 - F.cosine_similarity(probe_out, normalized_text, dim=-1).mean()
            )

        metrics: Dict[str, torch.Tensor] = {"val_loss": probe_loss.detach()}

        return metrics

    def _attach_handlers(
        self, trainer: PretrainEngine, evaluator: PretrainEngine
    ) -> None:
        """Attach Ignite event handlers for training orchestration."""
        # Running averages
        RunningAverage(output_transform=lambda o: o["loss"]).attach(trainer, "loss")
        RunningAverage(output_transform=lambda o: o["val_loss"]).attach(
            evaluator, "val_loss"
        )

        # NaN termination
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        # Progress bar for console display
        pbar = ProgressBar(
            file=sys.stdout,
            mininterval=10.0,
        )
        pbar.attach(trainer, ["loss"])

        # Step logging
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _log_train_step(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            output = engine.state.output
            if not isinstance(output, dict):
                return
            metrics = {
                "train/loss": float(output.get("loss", 0.0)),
                "train/loss_avg": float(engine.state.metrics.get("loss", 0.0)),
                "train/lr": float(output.get("lr", 0.0)),
                "train/global_step": int(engine.state.metrics.get("global_step", 0)),
            }
            self.wandb_logger.log(metrics, step=engine.state.metrics["global_step"])

        @trainer.on(Events.GET_BATCH_STARTED)
        def _set_horizon(engine: PretrainEngine) -> None:
            from typing import cast

            engine.state.dataloader.dataset.set_horizon(engine.state.horizon)

        @evaluator.on(Events.ITERATION_COMPLETED)
        def _set_horizon_eval(engine: PretrainEngine) -> None:
            from typing import cast

            engine.state.dataloader.dataset.set_horizon(trainer.state.horizon)

        # Best checkpoint handler
        best_checkpoint = Checkpoint(
            {
                "encoder": self.encoder,
                "jepa_predictor": self.jepa_predictor,
                "encoder_ema": self.ema_encoder,
                "jepa_ema": self.ema_jepa,
                "optimizer": self.optimizer,
                "scaler": self.scaler,
            },
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="pretrain_best_val",
            score_function=lambda engine: -float(engine.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=global_step_from_engine(trainer),
        )

        latest_checkpoint = Checkpoint(
            {
                "encoder": self.encoder,
                "jepa_predictor": self.jepa_predictor,
                "encoder_ema": self.ema_encoder,
                "jepa_ema": self.ema_jepa,
                "optimizer": self.optimizer,
                "scaler": self.scaler,
            },
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="pretrain_latest",
            filename_pattern="{filename_prefix}.pt",
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval),
            latest_checkpoint,
        )

        @trainer.on(
            Events.EPOCH_COMPLETED(every=getattr(self.config, "val_interval", 10))
        )
        def _run_validation(engine: PretrainEngine) -> None:
            evaluator.run(self.val_loader)

        val_batches = getattr(self.config, "val_batches", -1)
        if val_batches > 0:

            @evaluator.on(Events.ITERATION_COMPLETED)
            def _limit_val_batches(engine: PretrainEngine) -> None:
                if engine.state.iteration >= val_batches:
                    engine.terminate()

        if getattr(self.config, "save_best_val", True):

            @evaluator.on(Events.COMPLETED)
            def _log_best_validation(engine: PretrainEngine) -> None:
                val_loss = float(engine.state.metrics.get("val_loss", float("nan")))
                if self.wandb_logger:
                    self.wandb_logger.log(
                        {
                            "val/loss": val_loss,
                            "val/epoch": int(trainer.state.epoch),
                            "train/global_step": int(
                                trainer.state.metrics.get("global_step", 0)
                            ),
                        },
                        step=int(trainer.state.metrics.get("global_step", 0)),
                    )

            evaluator.add_event_handler(Events.COMPLETED, best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the pretraining loop."""
        trainer = PretrainEngine(self._train_step)
        self.evaluator = PretrainEngine(self._val_step)
        self._attach_handlers(trainer, self.evaluator)

        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)

        if self.wandb_logger:
            self.wandb_logger.finish()


# Convenience function for notebook usage
def train_pretrain(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: FeatureNormalizer | None = None,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel, EMAModel, Path]:
    """
    Train encoder with JEPA objective in a single call.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        normalizer: Optional feature normalizer
        wandb_project: Optional W&B project name
        max_epochs: Override number of epochs (uses config default if None)

    Returns:
        Tuple of (ema_encoder, ema_jepa_predictor, checkpoint_path)
    """
    trainer = PretrainTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = config.checkpoint_dir / "pretrain_latest.pt"
    return trainer.ema_encoder, trainer.ema_jepa, checkpoint_path


__all__ = ["EMAModel", "PretrainTrainer", "train_pretrain"]


# ========== models/finetune_trainer.py ==========

"""
Flow Matching Fine-tuning Trainer for Motion Generation.

Standalone, self-contained trainer using PyTorch Ignite.
Handles all model building, training loop configuration, and execution.
Designed for direct use in Kaggle notebooks with minimal boilerplate.
"""


import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan, global_step_from_engine
from ignite.metrics import RunningAverage

# [internal import removed]  from utils.config import Config
# [internal import removed]  from utils.models import MotionHistoryEncoder, FlowMatchingPredictor
# [internal import removed]  from utils.motion_utils import (
# [internal import removed]  from utils.wandb_logger import WandbLogger as _WandbLogger

T = torch.Tensor

# Constants for flow diagnostics
LOSS_VS_T_NUM_BINS = 100


def _compute_per_sample_flow_loss(pred: T, target_flow: T) -> T:
    """Compute one mean-MSE flow loss value per flattened training sample."""
    if pred.shape != target_flow.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target_flow.shape)}")
    if pred.ndim < 2:
        raise ValueError(f"Expected at least 2D, got {tuple(pred.shape)}")
    return F.mse_loss(pred, target_flow, reduction="none").mean(dim=-1)


def _aggregate_loss_vs_t_bins(
    t_values: T,
    per_sample_flow_loss: T,
    num_bins: int = LOSS_VS_T_NUM_BINS,
) -> Dict[str, T]:
    """Aggregate per-sample flow loss into fixed bins over t in [0, 1]."""
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if t_values.ndim != 1 or per_sample_flow_loss.ndim != 1:
        raise ValueError(f"Expected 1D tensors, got {tuple(t_values.shape)} and {tuple(per_sample_flow_loss.shape)}")

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, dtype=torch.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    counts = torch.zeros(num_bins, dtype=torch.long)
    mean_losses = torch.full((num_bins,), float("nan"), dtype=torch.float64)

    t_cpu = t_values.detach().to(dtype=torch.float64, device="cpu").clamp_(0.0, 1.0)
    loss_cpu = per_sample_flow_loss.detach().to(dtype=torch.float64, device="cpu")
    bin_indices = torch.clamp((t_cpu * num_bins).to(torch.long), max=num_bins - 1)

    counts = torch.bincount(bin_indices, minlength=num_bins)
    sums = torch.bincount(bin_indices, weights=loss_cpu, minlength=num_bins)
    nonempty = counts > 0
    mean_losses[nonempty] = sums[nonempty] / counts[nonempty].to(torch.float64)

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": counts,
        "mean_flow_loss": mean_losses,
    }


class EMAModel(nn.Module):
    """Exponential Moving Average model wrapper for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def update(self, model: nn.Module) -> None:
        """Update EMA weights from source model."""
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device: torch.device) -> "EMAModel":
        self.model.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)


class FiTrainer:
    """
    Flow matching fine-tuning trainer with curriculum support.
    
    Fully self-contained: builds models, optimizer, and Ignite engine.
    Usage:
        trainer = FiTrainer(config=config, train_loader=train_loader, val_loader=val_loader)
        trainer.run()
    """

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: FeatureNormalizer | None = None,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.wandb_project = wandb_project

        # Device and AMP setup
        self.device = torch.device(config.device)
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # Models - built internally
        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config.encoder_config)
        self.predictor: FlowMatchingPredictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
        )

        # EMA models
        self.ema_encoder: EMAModel = EMAModel(self.encoder, decay=float(config.ema_decay))
        self.ema_predictor: EMAModel = EMAModel(self.predictor, decay=float(config.ema_decay))

        # Optimizer and scaler
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.scaler: GradScaler = GradScaler("cuda", enabled=self.use_amp)

        # Ignite engines
        self.trainer: Engine | None = None
        self.evaluator: Engine | None = None

        # W&B logger
        self.wandb_logger: _WandbLogger | None = None

        # State tracking
        self.global_step = 0
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")

        # Initialize all objects
        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components: models, optimizer, W&B."""
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move to device
        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.ema_encoder.to(self.device)
        self.ema_predictor.to(self.device)

        # Setup W&B
        if self.wandb_project:
            self.wandb_logger = _WandbLogger(
                project=self.wandb_project,
                config={
                    "lr": float(self.config.learning_rate),
                    "weight_decay": float(self.config.weight_decay),
                    "ema_decay": float(self.config.ema_decay),
                    "batch_size": self.train_loader.batch_size,
                    "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                    "predictor_params": sum(p.numel() for p in self.predictor.parameters()),
                    "phase": "finetune",
                },
            )

    def _sample_timesteps(self, batch_size: int, epoch: int) -> T:
        """Sample training timesteps according to config (power or uniform)."""
        u = torch.rand(batch_size, device=self.device, dtype=torch.float32)

        if self.config.t_sampling_mode == "uniform":
            return u

        if self.config.t_sampling_power <= 0 or self.config.get_num_epochs <= 1:
            return u

        return u.pow(1.0 / (float(self.config.t_sampling_power) + 1.0))

    def _compute_losses(self, batch: Dict[str, T], epoch: int) -> Tuple[T, T, T]:
        """Compute flow and consistency losses."""
        motion = batch["motion"].to(self.device)
        text = batch["text_clip"].to(self.device)

        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)

        # History and target
        history = motion[:, :-1]
        target_motion = motion[:, -1]
        current_frame = history[:, -1]

        # Text embedding with CFG dropout
        if self.config.cfg_dropout > 0.0 and torch.rand(1, device=self.device).item() <= self.config.cfg_dropout:
            text_emb = torch.zeros(
                motion.shape[0],
                self.config.encoder_config.text_embedding_dim,
                device=self.device,
                dtype=text.dtype,
            )
        else:
            text_emb = text[:, 0, :] if text.ndim == 3 else text

        # Forward pass
        track_features = self.encoder(history, text_emb, return_all=False).unsqueeze(1)
        current_frame_features = extract_prev_frame_features(
            current_frame,
            normalizer=self.normalizer,
            normalize_output=self.normalizer is not None,
        )

        x1 = subset_271d_to_68d(target_motion, prev_frame=current_frame, normalizer=self.normalizer)
        x0 = torch.randn_like(x1)
        t = self._sample_timesteps(target_motion.shape[0], epoch)
        xt = t.unsqueeze(1) * x1 + (1 - t.unsqueeze(1)) * x0

        predicted_flow, _, _ = self.predictor(
            track_features=track_features,
            noisy_features=xt,
            timesteps=t,
            text_embedding=text_emb,
            current_frame_features=current_frame_features,
            output_attentions=False,
            output_hidden_states=False,
        )

        flow_loss = F.mse_loss(predicted_flow, x1 - x0)
        per_sample_loss = _compute_per_sample_flow_loss(predicted_flow, x1 - x0)

        # Consistency loss
        consistency_loss = predicted_flow.new_zeros(())
        if self.config.use_consistency_loss:
            t_thresh_mask = t > self.config.consistency_loss_t_threshold
            if t_thresh_mask.any():
                pred_x1 = xt[t_thresh_mask] + predicted_flow[t_thresh_mask] * (1 - t.unsqueeze(1)[t_thresh_mask])

                flow_raw = self.normalizer.denormalize_flow_output(pred_x1) if self.normalizer else pred_x1
                x1_raw = self.normalizer.denormalize_flow_output(x1[t_thresh_mask]) if self.normalizer else x1[t_thresh_mask]

                # Root loss
                root_loss = F.mse_loss(flow_raw[:, :3], x1_raw[:, :3])

                # Yaw loss
                x1_dyaw = sin_cos_to_yaw(x1_raw[:, 3:5])
                pred_dyaw = sin_cos_to_yaw(flow_raw[:, 3:5])
                yaw_error = wrap_angle(pred_dyaw - x1_dyaw)
                yaw_loss = yaw_error.pow(2).mean()

                # RIC loss
                ric_loss = F.mse_loss(flow_raw[:, 5:], x1_raw[:, 5:])

                consistency_loss = ric_loss * 3 + root_loss * 20 + yaw_loss * 10

        total_loss = flow_loss + self.config.consistency_loss_weight * consistency_loss
        return total_loss, flow_loss, consistency_loss

    def _train_step(self, engine: Engine, batch: Dict[str, T]) -> Dict[str, T]:
        """Execute one training step."""
        self.encoder.train()
        self.predictor.train()

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            loss, flow_loss, consistency_loss = self._compute_losses(batch, epoch=int(engine.state.epoch))

        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            float(self.config.gradient_clip),
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA update
        self.ema_encoder.update(self.encoder)
        self.ema_predictor.update(self.predictor)

        # Update state
        self.global_step = int(engine.state.iteration)
        self.best_train_loss = min(self.best_train_loss, float(loss.detach().item()))

        return {
            "loss": loss.detach(),
            "flow_loss": torch.tensor(flow_loss.detach().item(), device=self.device),
            "consistency_loss": torch.tensor(consistency_loss.detach().item(), device=self.device),
            "lr": torch.tensor(float(self.config.learning_rate), device=self.device),
        }

    def _val_step(self, engine: Engine, batch: Dict[str, T]) -> Dict[str, T]:
        """Execute one validation step."""
        motion = batch["motion"].to(self.device)
        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)

        with torch.no_grad():
            self.ema_encoder.model.eval()
            self.ema_predictor.model.eval()

            history = motion[:, :-1]
            target_motion = motion[:, -1]
            current_frame = history[:, -1]
            text_emb = text[:, 0, :] if text.ndim == 3 else text

            track_features = self.ema_encoder.model(history, text_emb, return_all=False).unsqueeze(1)
            current_frame_features = extract_prev_frame_features(
                current_frame,
                normalizer=self.normalizer,
                normalize_output=self.normalizer is not None,
            )

            x1 = subset_271d_to_68d(target_motion, prev_frame=current_frame, normalizer=self.normalizer)
            t = torch.rand(target_motion.shape[0], device=self.device, dtype=torch.float32)
            x0 = torch.randn_like(x1)
            xt = t.unsqueeze(1) * x1 + (1 - t.unsqueeze(1)) * x0

            predicted_flow, _, _ = self.ema_predictor.model(
                track_features=track_features,
                noisy_features=xt,
                timesteps=t,
                text_embedding=text_emb,
                current_frame_features=current_frame_features,
                output_attentions=False,
                output_hidden_states=False,
            )

            loss = F.mse_loss(predicted_flow, x1 - x0)

        return {"val_loss": loss.detach()}

    def _attach_handlers(self, trainer: Engine, evaluator: Engine) -> None:
        """Attach Ignite event handlers for training orchestration."""
        # Running averages
        RunningAverage(output_transform=lambda o: o["loss"]).attach(trainer, "loss")
        RunningAverage(output_transform=lambda o: o["val_loss"]).attach(evaluator, "val_loss")

        # NaN termination
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        # Step logging
        @trainer.on(Events.ITERATION_COMPLETED(every=50))
        def _log_train_step(engine: Engine) -> None:
            if not self.wandb_logger:
                return
            output = engine.state.output
            if not isinstance(output, dict):
                return
            metrics = {
                "train/loss": float(output.get("loss", 0.0)),
                "train/loss_avg": float(engine.state.metrics.get("loss", 0.0)),
                "train/lr": float(output.get("lr", 0.0)),
                "train/loss_flow": float(output.get("flow_loss", 0.0)),
                "train/loss_consistency": float(output.get("consistency_loss", 0.0)),
            }
            self.wandb_logger.log(metrics, step=int(engine.state.iteration))

        # Latest checkpoint
        latest_checkpoint = Checkpoint(
            {
                "encoder": self.encoder,
                "predictor": self.predictor,
                "encoder_ema": self.ema_encoder,
                "predictor_ema": self.ema_predictor,
                "optimizer": self.optimizer,
                "scaler": self.scaler,
            },
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="finetune_latest",
            global_step_transform=global_step_from_engine(trainer),
        )

        # Best checkpoint
        best_checkpoint = Checkpoint(
            {
                "encoder": self.encoder,
                "predictor": self.predictor,
                "encoder_ema": self.ema_encoder,
                "predictor_ema": self.ema_predictor,
                "optimizer": self.optimizer,
                "scaler": self.scaler,
            },
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="finetune_best",
            score_function=lambda e: -float(e.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=global_step_from_engine(trainer),
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, latest_checkpoint)

        @evaluator.on(Events.COMPLETED)
        def _log_val(engine: Engine) -> None:
            val_loss = float(engine.state.metrics.get("val_loss", float("nan")))
            if self.wandb_logger:
                self.wandb_logger.log({"val/loss": val_loss}, step=int(self.global_step))

        evaluator.add_event_handler(Events.COMPLETED, best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the fine-tuning loop."""
        if self.trainer is None:
            self.trainer = Engine(self._train_step)
            self.evaluator = Engine(self._val_step)
            self._attach_handlers(self.trainer, self.evaluator)

        epochs = max_epochs or int(self.config.get_num_epochs)
        self.trainer.run(self.train_loader, max_epochs=epochs)

        if self.wandb_logger:
            self.wandb_logger.finish()


# Convenience function for notebook usage
def train_finetune(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: FeatureNormalizer | None = None,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel, EMAModel]:
    """
    Fine-tune with flow matching objective in a single call.
    
    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        normalizer: Optional feature normalizer
        wandb_project: Optional W&B project name
        max_epochs: Override number of epochs (uses config default if None)
    
    Returns:
        Tuple of (ema_encoder, ema_predictor)
    """
    trainer = FiTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    return trainer.ema_encoder, trainer.ema_predictor


__all__ = ["EMAModel", "FiTrainer", "train_finetune", "_compute_per_sample_flow_loss", "_aggregate_loss_vs_t_bins"]