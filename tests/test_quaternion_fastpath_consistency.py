"""Regression tests for optimized quaternion fast paths."""

import os
import sys

import torch

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.quaternion import qinv, qmul, quaternion_to_cont6d, quaternion_to_matrix


def test_quaternion_to_cont6d_matches_matrix_columns() -> None:
    """Direct quaternion->6D path should match matrix-column extraction."""
    torch.manual_seed(0)

    q = torch.randn(64, 22, 4, dtype=torch.float32)
    q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-10)

    cont6d_fast = quaternion_to_cont6d(q)
    rot_mat = quaternion_to_matrix(q)
    cont6d_ref = torch.cat([rot_mat[..., 0], rot_mat[..., 1]], dim=-1)

    assert cont6d_fast.shape == cont6d_ref.shape
    assert torch.allclose(cont6d_fast, cont6d_ref, atol=1e-6, rtol=1e-5)


def test_qmul_identity_and_qinv_consistency() -> None:
    """Optimized qmul/qinv should preserve quaternion identity properties."""
    torch.manual_seed(1)

    q = torch.randn(128, 4, dtype=torch.float32)
    q = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-10)

    identity = torch.zeros_like(q)
    identity[:, 0] = 1.0

    left = qmul(identity, q)
    right = qmul(q, identity)
    inv = qinv(q)
    recon = qmul(q, inv)

    assert torch.allclose(left, q, atol=1e-6, rtol=1e-5)
    assert torch.allclose(right, q, atol=1e-6, rtol=1e-5)
    assert torch.allclose(recon, identity, atol=1e-5, rtol=1e-4)
