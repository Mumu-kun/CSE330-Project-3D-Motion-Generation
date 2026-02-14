"""
Unit tests for training utility functions.
Tests helper functions used in the training loop.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
from utils.train_utils import build_prev_and_clean_diffs


def test_build_prev_and_clean_diffs():
    """Test feature extraction from motion sequences."""
    print("\n=== Test 1: Feature extraction ===")

    B, T_hist, C = 2, 10, 263
    hist = torch.randn(B, T_hist, C)
    future = torch.randn(B, 1, C)

    prev_pos, prev_rot6d, prev_v, clean_v = build_prev_and_clean_diffs(hist, future)

    # Check shapes
    assert prev_pos.shape == (B, 22, 3), f"Expected (B, 22, 3), got {prev_pos.shape}"
    assert prev_rot6d.shape == (
        B,
        22,
        6,
    ), f"Expected (B, 22, 6), got {prev_rot6d.shape}"
    assert prev_v.shape == (B, 22, 3), f"Expected (B, 22, 3), got {prev_v.shape}"
    assert clean_v.shape == (B, 22, 3), f"Expected (B, 22, 3), got {clean_v.shape}"

    print(f"✓ prev_pos shape: {prev_pos.shape}")
    print(f"✓ prev_rot6d shape: {prev_rot6d.shape}")
    print(f"✓ prev_v shape: {prev_v.shape}")
    print(f"✓ clean_v shape: {clean_v.shape}")


def test_root_joint_identity():
    """Test that root joint has identity rotation."""
    print("\n=== Test 2: Root joint identity ===")

    B, T_hist, C = 2, 10, 263
    hist = torch.randn(B, T_hist, C)
    future = torch.randn(B, 1, C)

    prev_pos, prev_rot6d, prev_v, clean_v = build_prev_and_clean_diffs(hist, future)

    # Root rotation should be [1, 0, 0, 0, 1, 0]
    expected_root_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    for b in range(B):
        assert torch.allclose(
            prev_rot6d[b, 0], expected_root_rot, atol=1e-5
        ), f"Root rotation mismatch: {prev_rot6d[b, 0]}"

    print(f"✓ Root joint has identity rotation")


def test_root_position_zero():
    """Test that root position is zero in RIC space."""
    print("\n=== Test 3: Root position zero ===")

    B, T_hist, C = 2, 10, 263
    hist = torch.randn(B, T_hist, C)
    future = torch.randn(B, 1, C)

    prev_pos, prev_rot6d, prev_v, clean_v = build_prev_and_clean_diffs(hist, future)

    # Root position should be [0, 0, 0]
    expected_root_pos = torch.zeros(3)

    for b in range(B):
        assert torch.allclose(
            prev_pos[b, 0], expected_root_pos, atol=1e-5
        ), f"Root position mismatch: {prev_pos[b, 0]}"

    print(f"✓ Root position is zero")


def test_different_batch_sizes():
    """Test with various batch sizes."""
    print("\n=== Test 4: Different batch sizes ===")

    T_hist, C = 10, 263

    for B in [1, 4, 8]:
        hist = torch.randn(B, T_hist, C)
        future = torch.randn(B, 1, C)

        prev_pos, prev_rot6d, prev_v, clean_v = build_prev_and_clean_diffs(hist, future)

        assert prev_pos.shape == (B, 22, 3)
        assert prev_rot6d.shape == (B, 22, 6)
        assert prev_v.shape == (B, 22, 3)
        assert clean_v.shape == (B, 22, 3)

        print(f"✓ Batch size {B}: All shapes correct")


if __name__ == "__main__":
    test_build_prev_and_clean_diffs()
    test_root_joint_identity()
    test_root_position_zero()
    test_different_batch_sizes()
    print("\n✅ All training utility tests passed!")
