"""
Unit tests for FlowMatchingPredictor.
Tests the flow predictor in isolation with various input configurations.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
from models import FlowMatchingPredictor


def test_predictor_basic():
    """Test predictor with all inputs."""
    print("\n=== Test 1: Predictor with all inputs ===")

    B, joint_count = 2, 22
    per_joint_dim = 64
    model_dim = 128

    predictor = FlowMatchingPredictor(
        per_joint_dim=per_joint_dim,
        model_dim=model_dim,
        num_layers=2,
        joint_count=joint_count,
    )

    # Inputs
    history_features = torch.randn(B, joint_count, per_joint_dim)
    noise_level = torch.rand(B)
    noisy_target_diffs = torch.randn(B, joint_count, 3)
    prev_frame_features = torch.randn(B, joint_count, 12)  # pos(3) + rot6d(6) + vel(3)
    temporal_progress = torch.rand(B)

    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target_diffs=noisy_target_diffs,
        prev_frame_features=prev_frame_features,
        temporal_progress=temporal_progress,
    )

    expected_shape = (B, joint_count, 3)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape: {output.shape}")


def test_predictor_zero_shot():
    """Test predictor without previous frame (zero-shot)."""
    print("\n=== Test 2: Zero-shot (no prev frame) ===")

    B, joint_count = 2, 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    history_features = torch.randn(B, joint_count, 64)
    noise_level = torch.rand(B)
    noisy_target_diffs = torch.randn(B, joint_count, 3)

    # No previous frame, no temporal progress
    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target_diffs=noisy_target_diffs,
        prev_frame_features=None,
        temporal_progress=None,
    )

    expected_shape = (B, joint_count, 3)
    assert output.shape == expected_shape
    print(f"✓ Zero-shot output shape: {output.shape}")


def test_predictor_different_batch_sizes():
    """Test predictor with various batch sizes."""
    print("\n=== Test 3: Different batch sizes ===")

    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    for B in [1, 4, 8]:
        history_features = torch.randn(B, joint_count, 64)
        noise_level = torch.rand(B)
        noisy_target_diffs = torch.randn(B, joint_count, 3)

        output = predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target_diffs=noisy_target_diffs,
            prev_frame_features=None,
            temporal_progress=None,
        )

        assert output.shape == (B, joint_count, 3)
        print(f"✓ Batch size {B}: {output.shape}")


if __name__ == "__main__":
    test_predictor_basic()
    test_predictor_zero_shot()
    test_predictor_different_batch_sizes()
    print("\n✅ All FlowMatchingPredictor tests passed!")
