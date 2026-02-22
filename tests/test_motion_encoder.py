"""
Unit tests for MotionHistoryEncoder.
Tests the encoder in isolation with various input configurations.

Uses 271D feature format from motion_utils.py:
- [0:3]   Global root position (XYZ)
- [3:69]  RIC positions (22 * 3)
- [69:201] 6D rotations (22 * 6)
- [201:267] Local velocities (22 * 3)
- [267:271] Foot contacts (4D)
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models import MotionHistoryEncoder


def test_encoder_with_all_inputs():
    """Test encoder with text, motion history, and duration."""
    print("\n=== Test 1: Encoder with all inputs ===")

    B, T_hist, C = 2, 10, 271  # 271D feature format
    encoder = MotionHistoryEncoder(
        frame_feature_dim=C,
        text_embedding_dim=512,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    text = torch.randn(B, 2, 512)
    motion_history = torch.randn(B, T_hist, C)
    duration = torch.rand(B, 1)

    output = encoder(text=text, input_features=motion_history, total_duration=duration)

    expected_shape = (B, 22, 64)  # (B, joint_count, per_joint_out_dim)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape: {output.shape}")


def test_encoder_zero_shot():
    """Test encoder with no motion history (zero-shot generation)."""
    print("\n=== Test 2: Zero-shot (no history) ===")

    B = 2
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,  # 271D feature format
        text_embedding_dim=512,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    text = torch.randn(B, 2, 512)

    # No motion history, no duration
    output = encoder(text=text, input_features=None, total_duration=None)

    expected_shape = (B, 22, 64)
    assert output.shape == expected_shape
    print(f"✓ Zero-shot output shape: {output.shape}")


def test_encoder_unconditional():
    """Test encoder with no text (unconditional generation)."""
    print("\n=== Test 3: Unconditional (no text) ===")

    B, T_hist = 2, 10
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,  # 271D feature format
        text_embedding_dim=512,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    motion_history = torch.randn(B, T_hist, 271)

    # No text conditioning
    output = encoder(text=None, input_features=motion_history, total_duration=None)

    expected_shape = (B, 22, 64)
    assert output.shape == expected_shape
    print(f"✓ Unconditional output shape: {output.shape}")


def test_encoder_batch_broadcasting():
    """Test encoder with batch size 1 text broadcast to larger batch."""
    print("\n=== Test 4: Text broadcasting ===")

    B, T_hist = 4, 10
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,  # 271D feature format
        text_embedding_dim=512,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    text = torch.randn(1, 2, 512)  # Batch size 1
    motion_history = torch.randn(B, T_hist, 271)  # Batch size 4

    output = encoder(text=text, input_features=motion_history, total_duration=None)

    expected_shape = (B, 22, 64)
    assert output.shape == expected_shape
    print(f"✓ Broadcasting works: {output.shape}")


def test_encoder_with_clip_sequence():
    """Test encoder with CLIP sequence embeddings (B, l_seq, 512)."""
    print("\n=== Test 5: CLIP sequence embeddings ===")

    B, T_hist, l_seq = 2, 10, 77  # CLIP max sequence length
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,  # 271D feature format
        text_embedding_dim=512,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
        max_text_seq_len=77,
    )

    text = torch.randn(B, l_seq, 512)  # CLIP sequence embeddings
    motion_history = torch.randn(B, T_hist, 271)

    output = encoder(text=text, input_features=motion_history, total_duration=None)

    expected_shape = (B, 22, 64)
    assert output.shape == expected_shape
    print(f"✓ CLIP sequence output shape: {output.shape}")


if __name__ == "__main__":
    test_encoder_with_all_inputs()
    test_encoder_zero_shot()
    test_encoder_unconditional()
    test_encoder_batch_broadcasting()
    test_encoder_with_clip_sequence()
    print("\n✅ All MotionHistoryEncoder tests passed!")
