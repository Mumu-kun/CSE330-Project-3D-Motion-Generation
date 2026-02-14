"""
Unit tests for MotionHistoryEncoder.
Tests the encoder in isolation with various input configurations.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
from models import MotionHistoryEncoder


def test_encoder_with_all_inputs():
    """Test encoder with text, motion history, and duration."""
    print("\n=== Test 1: Encoder with all inputs ===")

    B, T_hist, C = 2, 10, 263
    encoder = MotionHistoryEncoder(
        frame_feature_dim=C,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=16,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    text = torch.randn(B, 512)
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
        frame_feature_dim=263,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=16,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    text = torch.randn(B, 512)

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
        frame_feature_dim=263,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=16,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    motion_history = torch.randn(B, T_hist, 263)

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
        frame_feature_dim=263,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=16,
        per_joint_out_dim=64,
        model_dim=128,
        joint_count=22,
    )

    text = torch.randn(1, 512)  # Batch size 1
    motion_history = torch.randn(B, T_hist, 263)  # Batch size 4

    output = encoder(text=text, input_features=motion_history, total_duration=None)

    expected_shape = (B, 22, 64)
    assert output.shape == expected_shape
    print(f"✓ Broadcasting works: {output.shape}")


if __name__ == "__main__":
    test_encoder_with_all_inputs()
    test_encoder_zero_shot()
    test_encoder_unconditional()
    test_encoder_batch_broadcasting()
    print("\n✅ All MotionHistoryEncoder tests passed!")
