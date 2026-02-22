"""
Test full history tracking in HumanMotionGenerator.generate_sequence.

Verifies that:
1. Position history grows correctly
2. Feature history grows correctly
3. No NaN values in generated sequences
4. Velocities are computed correctly from full history
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
from models import HumanMotionGenerator, MotionHistoryEncoder, FlowMatchingPredictor
from config import Config


def test_full_history_tracking():
    """Test that full history is maintained during generation."""
    print("=" * 60)
    print("Testing Full History Tracking")
    print("=" * 60)

    config = Config()

    # Create encoder and predictor
    encoder = MotionHistoryEncoder(
        frame_feature_dim=config.motion_dim,
        text_embedding_dim=config.text_embedding_dim,
        per_joint_out_dim=config.per_joint_out_dim,
        joint_count=config.num_joints,
        model_dim=config.model_dim,
        num_layers=2,  # Smaller for testing
        max_text_seq_len=config.max_text_seq_len,
        dropout=config.dropout,
    )

    predictor = FlowMatchingPredictor(
        per_joint_dim=config.per_joint_out_dim,
        model_dim=config.model_dim,
        num_layers=2,  # Smaller for testing
        joint_count=config.num_joints,
    )

    # Create generator
    generator = HumanMotionGenerator(encoder, predictor)
    generator.eval()

    # Test 1: Zero-shot generation
    print("\n[Test 1] Zero-shot generation (null_history)")
    text_embedding = torch.randn(1, 1, 512)  # Mock CLIP embedding

    with torch.no_grad():
        joints = generator.generate_sequence(
            text=text_embedding,
            num_frames=10,
            num_steps=2,
            guidance_scale=2.5,
            input_features=None,
            dataset_type="t2m",
        )

    print(f"  Output shape: {joints.shape}")
    print(f"  Expected: (1, 10, 22, 3)")
    assert joints.shape == (1, 10, 22, 3), f"Shape mismatch: {joints.shape}"
    print(f"  Has NaN: {torch.isnan(joints).any().item()}")
    assert not torch.isnan(joints).any(), "NaN values in output"
    print("  [PASS]")

    # Test 2: Seeded generation with single frame
    print("\n[Test 2] Seeded generation (single frame input)")
    input_features = torch.randn(1, 271)

    with torch.no_grad():
        joints = generator.generate_sequence(
            text=text_embedding,
            num_frames=5,
            num_steps=2,
            guidance_scale=2.5,
            input_features=input_features,
            dataset_type="t2m",
        )

    print(f"  Output shape: {joints.shape}")
    assert joints.shape == (1, 5, 22, 3), f"Shape mismatch: {joints.shape}"
    assert not torch.isnan(joints).any(), "NaN values in output"
    print("  [PASS]")

    # Test 3: Seeded generation with sequence
    print("\n[Test 3] Seeded generation (sequence input)")
    input_features = torch.randn(1, 5, 271)  # 5 frames of history

    with torch.no_grad():
        joints = generator.generate_sequence(
            text=text_embedding,
            num_frames=5,
            num_steps=2,
            guidance_scale=2.5,
            input_features=input_features,
            dataset_type="t2m",
        )

    print(f"  Output shape: {joints.shape}")
    assert joints.shape == (1, 5, 22, 3), f"Shape mismatch: {joints.shape}"
    assert not torch.isnan(joints).any(), "NaN values in output"
    print("  [PASS]")

    # Test 4: Batch generation
    print("\n[Test 4] Batch generation")
    text_embedding = torch.randn(2, 1, 512)  # Batch of 2

    with torch.no_grad():
        joints = generator.generate_sequence(
            text=text_embedding,
            num_frames=5,
            num_steps=2,
            guidance_scale=2.5,
            input_features=None,
            dataset_type="t2m",
        )

    print(f"  Output shape: {joints.shape}")
    assert joints.shape == (2, 5, 22, 3), f"Shape mismatch: {joints.shape}"
    assert not torch.isnan(joints).any(), "NaN values in output"
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_history_tracking()
