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
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

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
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

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
