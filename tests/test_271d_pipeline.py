"""
Test script for 271D feature pipeline.
Tests MotionHistoryEncoder, FlowMatchingPredictor, and HumanMotionGenerator.
"""

import sys
import os

# Add project root to path for 'src' module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator
from src.config import Config


def test_motion_history_encoder():
    """Test MotionHistoryEncoder with 271D features."""
    print("\n" + "=" * 60)
    print("TEST 1: MotionHistoryEncoder with 271D features")
    print("=" * 60)

    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        joint_feature_projection_dim=64,
        text_projection_dim=64,
        per_joint_out_dim=64,
        model_dim=256,
        joint_count=22,
    )

    B, T_hist = 2, 15
    text = torch.randn(B, 77, 512)  # CLIP sequence
    motion_history = torch.randn(B, T_hist, 271)  # 271D features

    output = encoder(text=text, input_features=motion_history)
    print(f"Input motion history shape: {motion_history.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, 22, 64), f"Expected (2, 22, 64), got {output.shape}"
    print("[PASS] MotionHistoryEncoder test PASSED!")
    return encoder, output


def test_flow_matching_predictor(encoder_output):
    """Test FlowMatchingPredictor."""
    print("\n" + "=" * 60)
    print("TEST 2: FlowMatchingPredictor")
    print("=" * 60)

    B = encoder_output.shape[0]

    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=256,
        num_layers=4,
        joint_count=22,
    )

    history_features = encoder_output
    noise_level = torch.rand(B)
    noisy_target = torch.randn(B, 22, 3)
    prev_frame_features = torch.randn(B, 22, 12)  # pos(3) + rot(6) + vel(3)

    pred = predictor(
        history_features=history_features,
        noise_level=noise_level,
        noisy_target=noisy_target,
        prev_frame_features=prev_frame_features,
    )
    print(f"Predictor output shape: {pred.shape}")
    assert pred.shape == (B, 22, 3), f"Expected (2, 22, 3), got {pred.shape}"
    print("[PASS] FlowMatchingPredictor test PASSED!")
    return predictor


def test_human_motion_generator(encoder, predictor):
    """Test HumanMotionGenerator full pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: HumanMotionGenerator (short generation)")
    print("=" * 60)

    generator = HumanMotionGenerator(encoder, predictor)

    B = 2
    text = torch.randn(B, 77, 512)

    # Generate a short sequence
    joint_positions = generator.generate_sequence(
        text=text,
        num_frames=5,
        num_steps=3,
        guidance_scale=1.0,
    )
    print(f"Generated joint positions shape: {joint_positions.shape}")
    assert joint_positions.shape == (
        B,
        5,
        22,
        3,
    ), f"Expected (2, 5, 22, 3), got {joint_positions.shape}"
    print("[PASS] HumanMotionGenerator test PASSED!")


def test_config():
    """Test Config has correct motion_dim."""
    print("\n" + "=" * 60)
    print("TEST 4: Config compatibility")
    print("=" * 60)

    config = Config()
    print(f"Config motion_dim: {config.motion_dim}")
    assert config.motion_dim == 271, f"Expected 271, got {config.motion_dim}"
    print("[PASS] Config test PASSED!")


def test_incremental_extractor_integration():
    """Test integration with IncrementalFeatureExtractor."""
    print("\n" + "=" * 60)
    print("TEST 5: IncrementalFeatureExtractor integration")
    print("=" * 60)

    from src.utils.motion_utils import (
        IncrementalFeatureExtractor,
        preprocess_sequence,
        features_to_positions,
    )

    # Create sample joint positions
    B, N, J, D = 1, 30, 22, 3
    positions = torch.randn(B, N, J, D)

    # Preprocess to 271D features
    features = preprocess_sequence(positions)
    print(f"Preprocessed features shape: {features.shape}")
    assert features.shape == (B, N, 271), f"Expected (1, 30, 271), got {features.shape}"

    # Reconstruct positions
    reconstructed = features_to_positions(features)
    print(f"Reconstructed positions shape: {reconstructed.shape}")
    assert reconstructed.shape == (
        B,
        N,
        22,
        3,
    ), f"Expected (1, 30, 22, 3), got {reconstructed.shape}"

    # Test incremental extractor
    extractor = IncrementalFeatureExtractor(device=torch.device("cpu"))
    init_positions = positions[0, 0]  # First frame (22, 3)
    init_features = extractor.initialize(init_positions.unsqueeze(0))
    print(f"Initial features shape: {init_features.shape}")
    assert init_features.shape == (
        1,
        271,
    ), f"Expected (1, 271), got {init_features.shape}"

    # Process a few frames
    for i in range(1, 5):
        frame_positions = positions[0, i]  # (22, 3)
        frame_features = extractor.process_frame(frame_positions.unsqueeze(0))
        assert frame_features.shape == (
            1,
            271,
        ), f"Expected (1, 271), got {frame_features.shape}"

    print("[PASS] IncrementalFeatureExtractor integration test PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("271D FEATURE PIPELINE TESTS")
    print("=" * 60)

    encoder, encoder_output = test_motion_history_encoder()
    predictor = test_flow_matching_predictor(encoder_output)
    test_human_motion_generator(encoder, predictor)
    test_config()
    test_incremental_extractor_integration()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
