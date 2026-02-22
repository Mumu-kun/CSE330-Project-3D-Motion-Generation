"""
Unit tests for FlowMatchingPredictor.
Tests the flow predictor in isolation with various input configurations.

FlowMatchingPredictor I/O Scheme (Option A - 261D prev_frame):
- history_features: (B, 22, per_joint_dim) - Context from MotionHistoryEncoder
- noise_level: (B,) - Flow time t in [0,1]
- noisy_target: (B, 72) - Root (9D) + Joint RIC positions (63D)
- prev_frame_features: (B, 261) - Root (9D) + Joint features (252D)
- Output: (B, 72) - Root (9D) + Joint RIC positions (63D)

Feature Layout:
- noisy_target[0:9]: Root features (height 1D + velocity 2D + rotation_6d 6D)
- noisy_target[9:72]: Joint RIC positions (21 joints x 3D = 63D)
- prev_frame_features[0:9]: Root features (height 1D + velocity 2D + rotation_6d 6D)
- prev_frame_features[9:261]: Joint features (21 joints x 12D = 252D)
  - Per joint: RIC position (3D) + rotation_6d (6D) + local_velocity (3D) = 12D
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models import FlowMatchingPredictor


def test_predictor_basic():
    """Test predictor with all inputs using correct I/O shapes."""
    print("\n=== Test 1: Predictor with all inputs ===")

    B = 2
    joint_count = 22
    per_joint_dim = 64
    model_dim = 128

    predictor = FlowMatchingPredictor(
        per_joint_dim=per_joint_dim,
        model_dim=model_dim,
        num_layers=2,
        joint_count=joint_count,
    )

    # Inputs with correct shapes
    history_features = torch.randn(B, joint_count, per_joint_dim)
    noise_level = torch.rand(B)
    noisy_target = torch.randn(B, 72)  # Root (9D) + Joint RIC (63D)
    prev_frame_features = torch.randn(B, 261)  # Root (9D) + Joint features (252D)
    temporal_progress = torch.rand(B)

    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target=noisy_target,
        prev_frame_features=prev_frame_features,
        temporal_progress=temporal_progress,
    )

    expected_shape = (B, 72)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"
    print(f"  Output shape: {output.shape} - PASS")

    # Verify output is finite
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    print("  Output values are finite - PASS")


def test_predictor_zero_shot():
    """Test predictor without previous frame (zero-shot generation)."""
    print("\n=== Test 2: Zero-shot (no prev frame) ===")

    B = 2
    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    history_features = torch.randn(B, joint_count, 64)
    noise_level = torch.rand(B)
    noisy_target = torch.randn(B, 72)  # Correct shape

    # No previous frame, no temporal progress
    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target=noisy_target,
        prev_frame_features=None,
        temporal_progress=None,
    )

    expected_shape = (B, 72)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"
    print(f"  Zero-shot output shape: {output.shape} - PASS")


def test_predictor_no_noisy_target():
    """Test predictor with None noisy_target (random generation)."""
    print("\n=== Test 3: No noisy target (random generation) ===")

    B = 4
    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    history_features = torch.randn(B, joint_count, 64)
    noise_level = torch.rand(B)
    prev_frame_features = torch.randn(B, 261)

    # No noisy target - should generate random internally
    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target=None,
        prev_frame_features=prev_frame_features,
    )

    expected_shape = (B, 72)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"
    print(f"  Random generation output shape: {output.shape} - PASS")


def test_predictor_different_batch_sizes():
    """Test predictor with various batch sizes."""
    print("\n=== Test 4: Different batch sizes ===")

    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    for B in [1, 4, 8, 16]:
        history_features = torch.randn(B, joint_count, 64)
        noise_level = torch.rand(B)
        noisy_target = torch.randn(B, 72)
        prev_frame_features = torch.randn(B, 261)

        output = predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target=noisy_target,
            prev_frame_features=prev_frame_features,
        )

        assert output.shape == (
            B,
            72,
        ), f"Batch {B}: expected {(B, 72)}, got {output.shape}"
        print(f"  Batch size {B}: {output.shape} - PASS")


def test_predictor_noise_level_range():
    """Test predictor with different noise levels (flow time t)."""
    print("\n=== Test 5: Noise level range [0, 1] ===")

    B = 2
    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    history_features = torch.randn(B, joint_count, 64)
    noisy_target = torch.randn(B, 72)
    prev_frame_features = torch.randn(B, 261)

    # Test boundary values
    for t_val in [0.0, 0.5, 1.0]:
        noise_level = torch.full((B,), t_val)
        output = predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target=noisy_target,
            prev_frame_features=prev_frame_features,
        )
        assert output.shape == (B, 72)
        assert torch.isfinite(output).all()
        print(f"  noise_level={t_val}: {output.shape} - PASS")


def test_predictor_output_structure():
    """Test that output has correct structure (root + joints)."""
    print("\n=== Test 6: Output structure validation ===")

    B = 2
    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    history_features = torch.randn(B, joint_count, 64)
    noise_level = torch.rand(B)
    noisy_target = torch.randn(B, 72)
    prev_frame_features = torch.randn(B, 261)

    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target=noisy_target,
        prev_frame_features=prev_frame_features,
    )

    # Check root portion (first 9D)
    root_output = output[:, :9]
    assert root_output.shape == (
        B,
        9,
    ), f"Root shape: expected {(B, 9)}, got {root_output.shape}"
    print(f"  Root output shape: {root_output.shape} - PASS")

    # Check joint portion (next 63D)
    joint_output = output[:, 9:]
    assert joint_output.shape == (
        B,
        63,
    ), f"Joint shape: expected {(B, 63)}, got {joint_output.shape}"
    print(f"  Joint output shape: {joint_output.shape} - PASS")


def test_predictor_gradient_flow():
    """Test that gradients flow through the predictor."""
    print("\n=== Test 7: Gradient flow ===")

    B = 2
    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )

    # Requires grad for input
    history_features = torch.randn(B, joint_count, 64, requires_grad=True)
    noisy_target = torch.randn(B, 72, requires_grad=True)
    prev_frame_features = torch.randn(B, 261, requires_grad=True)
    noise_level = torch.rand(B)

    output = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target=noisy_target,
        prev_frame_features=prev_frame_features,
    )

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert history_features.grad is not None, "No gradient for history_features"
    assert noisy_target.grad is not None, "No gradient for noisy_target"
    assert prev_frame_features.grad is not None, "No gradient for prev_frame_features"
    print("  Gradients computed for all inputs - PASS")


def test_predictor_deterministic():
    """Test that same inputs produce same outputs in eval mode."""
    print("\n=== Test 8: Deterministic output in eval mode ===")

    B = 2
    joint_count = 22
    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=128,
        num_layers=2,
        joint_count=joint_count,
    )
    predictor.eval()

    # Fixed inputs
    torch.manual_seed(42)
    history_features = torch.randn(B, joint_count, 64)
    noise_level = torch.rand(B)
    noisy_target = torch.randn(B, 72)
    prev_frame_features = torch.randn(B, 261)

    with torch.no_grad():
        output1 = predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target=noisy_target,
            prev_frame_features=prev_frame_features,
        )
        output2 = predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target=noisy_target,
            prev_frame_features=prev_frame_features,
        )

    assert torch.allclose(
        output1, output2
    ), "Outputs differ for same inputs in eval mode"
    print("  Deterministic output in eval mode - PASS")


if __name__ == "__main__":
    test_predictor_basic()
    test_predictor_zero_shot()
    test_predictor_no_noisy_target()
    test_predictor_different_batch_sizes()
    test_predictor_noise_level_range()
    test_predictor_output_structure()
    test_predictor_gradient_flow()
    test_predictor_deterministic()
    print("\n" + "=" * 50)
    print("All FlowMatchingPredictor tests passed!")
    print("=" * 50)
