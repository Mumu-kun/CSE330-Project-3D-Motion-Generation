# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

_EPS4 = np.finfo(float).eps * 4.0

_FLOAT_EPS = np.finfo(np.float64).eps


# PyTorch-backed implementations
def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return qinv(torch.from_numpy(q).float()).numpy()


def qnormalize(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.norm(q, dim=-1, keepdim=True)


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


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
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0, deg=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    if deg:
        return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
    else:
        return torch.stack((x, y, z), dim=1).view(original_shape)


# Numpy-backed implementations


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda().float()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous().float()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def euler2quat(e, order, deg=True):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.view(-1, 3)

    ## if euler angles in degrees
    if deg:
        e = e * np.pi / 180.0

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack(
        (torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)),
        dim=1,
    )
    ry = torch.stack(
        (torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)),
        dim=1,
    )
    rz = torch.stack(
        (torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)),
        dim=1,
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.view(original_shape)


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


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


def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def cont6d_to_matrix(cont6d):
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


def cont6d_to_matrix_np(cont6d):
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


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


def matrix_to_quaternion_np(rotation_matrix):
    """Numpy version of matrix_to_quaternion."""
    mat = torch.from_numpy(rotation_matrix).contiguous().float()
    return matrix_to_quaternion(mat).numpy()


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


def cont6d_to_quaternion_np(cont6d):
    """Numpy version of cont6d_to_quaternion."""
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_quaternion(q).numpy()


def qpow(q0, t, dtype=torch.float):
    """q0 : tensor of quaternions
    t: tensor of powers
    """
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., 0])

    ## if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / torch.sin(theta0).view(-1, 1)

    if isinstance(t, torch.Tensor):
        q = torch.zeros(t.shape + q0.shape)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:  ## if t is a number
        q = torch.zeros(q0.shape)
        theta = t * theta0

    q[..., 0] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta).unsqueeze(-1)

    return q.to(dtype)


def qslerp(q0, q1, t):
    """
    q0: starting quaternion
    q1: ending quaternion
    t: array of points along the way

    Returns:
    Tensor of Slerps: t.shape + q0.shape
    """

    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)

    return qmul(
        q_,
        q0.contiguous()
        .view(torch.Size([1] * len(t.shape)) + q0.shape)
        .expand(t.shape + q0.shape)
        .contiguous(),
    )


def qbetween(v0, v1):
    """
    find the quaternion used to rotate v0 to v1
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1)
    w = torch.sqrt(
        (v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)
    ) + (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0, v1):
    """
    find the quaternion used to rotate v0 to v1
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


def lerp(p0, p1, t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor([t])

    new_shape = t.shape + p0.shape
    new_view_t = t.shape + torch.Size([1] * len(p0.shape))
    new_view_p = torch.Size([1] * len(t.shape)) + p0.shape
    p0 = p0.view(new_view_p).expand(new_shape)
    p1 = p1.view(new_view_p).expand(new_shape)
    t = t.view(new_view_t).expand(new_shape)

    return p0 + t * (p1 - p0)
