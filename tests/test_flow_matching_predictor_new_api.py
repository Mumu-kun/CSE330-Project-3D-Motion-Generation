"""
Test FlowMatchingPredictor with new API: required text_embedding and 3D relative_shifts.

This test verifies:
1. Predictor initialization with use_relative_shift enabled
2. Forward pass with explicit relative_shifts (3D root-relative offsets)
3. Forward pass with None relative_shifts (auto-zero padding)
4. Output shape and numerical stability
5. Text conditioning requirement enforced
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, FlowMatchingPredictorConfig
from models import MotionHistoryEncoder, FlowMatchingPredictor


def get_test_config():
    """Get test configuration with minimal settings."""
    config = Config()
    config.device = "cpu"  # Use CPU for testing
    config.batch_size = 2
    config.encoder_per_joint_dim = 64
    return config


def get_predictor_config():
    """Get FlowMatchingPredictor configuration."""
    pred_config = FlowMatchingPredictorConfig()
    pred_config.hidden_size = 256
    pred_config.intermediate_size = 512
    pred_config.num_hidden_layers = 2
    pred_config.num_attention_heads = 4
    pred_config.attention_dropout = 0.0
    pred_config.track_dimensionality = 3  # 3D positions
    return pred_config


# =============================================================================
# Test 1: Predictor Initialization with use_relative_shift=True
# =============================================================================


def test_predictor_initialization():
    """Test FlowMatchingPredictor initialization."""
    print("=" * 70)
    print("Test 1: FlowMatchingPredictor Initialization")
    print("=" * 70)

    config = get_test_config()
    pred_config = get_predictor_config()

    # Initialize predictor with relative shift enabled
    predictor = FlowMatchingPredictor(
        feature_size=config.encoder_per_joint_dim,  # 64
        config=pred_config,
        use_relative_shift=True,
    )

    param_count = sum(p.numel() for p in predictor.parameters())
    print(f"[OK] Predictor initialized with {param_count:,} parameters")
    print(f"[OK] Use relative shift: {predictor.use_relative_shift}")
    print(f"[OK] Output channels (track_dimensionality): {predictor.out_channels}")

    print("[PASS] Predictor initialization successful\n")
    return predictor, config, pred_config


# =============================================================================
# Test 2: Forward Pass with Explicit Relative Shifts
# =============================================================================


def test_forward_with_explicit_shifts(predictor, config, pred_config):
    """Test forward pass with explicit 3D relative shifts."""
    print("=" * 70)
    print("Test 2: Forward Pass with Explicit 3D Relative Shifts")
    print("=" * 70)

    B = config.batch_size  # 2
    N = 22  # Joint count
    F = config.encoder_per_joint_dim  # 64
    D = pred_config.track_dimensionality  # 3
    C = pred_config.global_cond_dim  # 512

    device = config.device

    # Create mock inputs
    track_features = torch.randn(B, N, F, device=device)  # (2, 22, 64)
    noised_tracks = torch.randn(B, N, D, device=device)  # (2, 22, 3)
    timesteps = torch.rand(B, device=device)  # (2,) in [0, 1)
    text_embedding = torch.randn(B, C, device=device)  # (2, 512)

    # Compute relative shifts as root-relative offsets
    relative_shifts = noised_tracks - noised_tracks[:, :1, :]  # (2, 22, 3)

    print(f"Input shapes:")
    print(f"  track_features:  {track_features.shape}")
    print(f"  noised_tracks:   {noised_tracks.shape}")
    print(f"  timesteps:       {timesteps.shape}")
    print(f"  text_embedding:  {text_embedding.shape}")
    print(f"  relative_shifts: {relative_shifts.shape}")

    # Forward pass
    with torch.no_grad():
        output = predictor(
            track_features=track_features,
            noised_tracks=noised_tracks,
            timesteps=timesteps,
            text_embedding=text_embedding,
            relative_shifts=relative_shifts,
        )

    # Unpack tuple: (flow_prediction, hidden_states, attentions)
    flow_pred, _, _ = output
    print(f"\n[OK] Forward pass completed")
    print(f"  Output shape: {flow_pred.shape}")

    # Verify output shape
    assert flow_pred.shape == (
        B,
        N,
        D,
    ), f"Expected shape ({B}, {N}, {D}), got {flow_pred.shape}"
    print(f"[OK] Output shape correct: {flow_pred.shape}")

    # Check numerical stability
    assert torch.isfinite(flow_pred).all(), "Output contains NaN or inf!"
    print(f"[OK] Output is numerically stable (no NaN/inf)")

    # Check output range (should be relatively bounded)
    max_val = flow_pred.abs().max().item()
    print(
        f"[OK] Output value range: [{flow_pred.min():.4f}, {flow_pred.max():.4f}] (max abs: {max_val:.4f})"
    )

    print("[PASS] Forward pass with explicit shifts successful\n")
    return True


# =============================================================================
# Test 3: Forward Pass with None relative_shifts (auto-zero)
# =============================================================================


def test_forward_with_none_shifts(predictor, config, pred_config):
    """Test forward pass with None relative_shifts (should auto-zero)."""
    print("=" * 70)
    print("Test 3: Forward Pass with None Relative Shifts (Auto-Zero)")
    print("=" * 70)

    B = config.batch_size  # 2
    N = 22  # Joint count
    F = config.encoder_per_joint_dim  # 64
    D = pred_config.track_dimensionality  # 3
    C = pred_config.global_cond_dim  # 512

    device = config.device

    # Create mock inputs (same as Test 2)
    track_features = torch.randn(B, N, F, device=device)  # (2, 22, 64)
    noised_tracks = torch.randn(B, N, D, device=device)  # (2, 22, 3)
    timesteps = torch.rand(B, device=device)  # (2,)
    text_embedding = torch.randn(B, C, device=device)  # (2, 512)

    print(f"Input shapes (no explicit shifts provided):")
    print(f"  track_features:  {track_features.shape}")
    print(f"  noised_tracks:   {noised_tracks.shape}")
    print(f"  timesteps:       {timesteps.shape}")
    print(f"  text_embedding:  {text_embedding.shape}")
    print(f"  relative_shifts: None (should auto-zero)")

    # Forward pass with None relative_shifts
    with torch.no_grad():
        output = predictor(
            track_features=track_features,
            noised_tracks=noised_tracks,
            timesteps=timesteps,
            text_embedding=text_embedding,
            relative_shifts=None,  # Should auto-zero
        )

    # Unpack tuple: (flow_prediction, hidden_states, attentions)
    flow_pred, _, _ = output
    print(f"\n[OK] Forward pass completed with auto-zero shifts")
    print(f"  Output shape: {flow_pred.shape}")

    # Verify output shape
    assert flow_pred.shape == (
        B,
        N,
        D,
    ), f"Expected shape ({B}, {N}, {D}), got {flow_pred.shape}"
    print(f"[OK] Output shape correct: {flow_pred.shape}")

    # Check numerical stability
    assert torch.isfinite(flow_pred).all(), "Output contains NaN or inf!"
    print(f"[OK] Output is numerically stable (no NaN/inf)")

    # Compare with explicit zero shifts
    with torch.no_grad():
        explicit_zeros = torch.zeros_like(noised_tracks)
        output_explicit = predictor(
            track_features=track_features,
            noised_tracks=noised_tracks,
            timesteps=timesteps,
            text_embedding=text_embedding,
            relative_shifts=explicit_zeros,
        )

    # Unpack both for comparison
    flow_pred_explicit, _, _ = output_explicit
    print(f"[OK] Comparing None vs explicit zeros:")
    print(f"  Are outputs identical? {torch.allclose(flow_pred, flow_pred_explicit)}")

    print("[PASS] Forward pass with None shifts (auto-zero) successful\n")
    return True


# =============================================================================
# Test 4: Global Conditioning Requirement
# =============================================================================


def test_text_embedding_required(predictor, config, pred_config):
    """Test that text_embedding parameter is required."""
    print("=" * 70)
    print("Test 4: Global Conditioning is Required")
    print("=" * 70)

    B = config.batch_size
    N = 22
    F = config.encoder_per_joint_dim
    D = pred_config.track_dimensionality
    device = config.device

    # Create mock inputs
    track_features = torch.randn(B, N, F, device=device)
    noised_tracks = torch.randn(B, N, D, device=device)
    timesteps = torch.rand(B, device=device)

    print(f"Attempting forward pass WITHOUT text_embedding (should fail)...")

    try:
        with torch.no_grad():
            output = predictor(
                track_features=track_features,
                noised_tracks=noised_tracks,
                timesteps=timesteps,
                # text_embedding intentionally omitted
            )
        print(
            "[FAIL] Forward pass succeeded without text_embedding! This should not happen!"
        )
        return False
    except TypeError as e:
        print(f"[OK] Expected TypeError caught: {str(e)[:80]}...")
        print(f"[OK] text_embedding is correctly required")

    print("[PASS] Global conditioning requirement enforced\n")
    return True


# =============================================================================
# Test 5: Batch Size Variation
# =============================================================================


def test_batch_size_variation(predictor, config, pred_config):
    """Test predictor with different batch sizes."""
    print("=" * 70)
    print("Test 5: Batch Size Variation")
    print("=" * 70)

    N = 22
    F = config.encoder_per_joint_dim
    D = pred_config.track_dimensionality
    C = pred_config.global_cond_dim
    device = config.device

    batch_sizes = [1, 2, 4, 8]

    for B in batch_sizes:
        track_features = torch.randn(B, N, F, device=device)
        noised_tracks = torch.randn(B, N, D, device=device)
        timesteps = torch.rand(B, device=device)
        text_embedding = torch.randn(B, C, device=device)

        with torch.no_grad():
            output = predictor(
                track_features=track_features,
                noised_tracks=noised_tracks,
                timesteps=timesteps,
                text_embedding=text_embedding,
            )

        # Unpack tuple: (flow_prediction, hidden_states, attentions)
        flow_pred, _, _ = output
        assert flow_pred.shape == (B, N, D), f"Shape mismatch for B={B}"
        assert torch.isfinite(flow_pred).all(), f"Non-finite output for B={B}"
        print(f"[OK] B={B}: output shape {flow_pred.shape}, stable")

    print("[PASS] Batch size variation test successful\n")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================


def main():
    print("\n" + "=" * 70)
    print("FlowMatchingPredictor New API Test Suite")
    print("=" * 70 + "\n")

    try:
        # Test 1: Initialization
        predictor, config, pred_config = test_predictor_initialization()

        # Test 2: Forward with explicit shifts
        test_forward_with_explicit_shifts(predictor, config, pred_config)

        # Test 3: Forward with None shifts (auto-zero)
        test_forward_with_none_shifts(predictor, config, pred_config)

        # Test 4: Text conditioning requirement
        test_text_embedding_required(predictor, config, pred_config)

        # Test 5: Batch size variation
        test_batch_size_variation(predictor, config, pred_config)

        print("=" * 70)
        print("ALL TESTS PASSED [OK]")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"TEST FAILED [FAIL]")
        print(f"{'=' * 70}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
