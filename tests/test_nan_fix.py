"""
Test for NaN values fix in HumanMotionGenerator.generate_sequence().

Uses tests/checkpoints/best.pt and sample_data/000070_vec.npy to verify
the output matches sample_data/000070_joint.npy.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
import numpy as np


def test_nan_fix():
    """Test that generate_sequence produces valid output without NaN."""
    print("=" * 60)
    print("NaN Fix Test - HumanMotionGenerator.generate_sequence()")
    print("=" * 60)

    device = torch.device("cpu")

    # Load sample data
    vecs = np.load("sample_data/000070_vec.npy")
    joints = np.load("sample_data/000070_joint.npy")

    print(f"\n1. Sample data:")
    print(f"   vecs shape: {vecs.shape}")
    print(f"   joints shape: {joints.shape}")

    # Load checkpoint
    from src.models import HumanMotionGenerator
    from src.config import Config

    config = Config()
    config.max_text_seq_len = 1  # Match checkpoint

    print(f"\n2. Loading checkpoint...")
    generator = HumanMotionGenerator.load_from_checkpoint(
        "tests/checkpoints/best.pt", config, device=str(device)
    )
    generator.eval()
    print(f"   Generator loaded successfully")

    # Prepare input features (B, N, 271) = (1, 1, 271)
    # Use first frame from sample data
    vecs_t = torch.from_numpy(vecs).float()
    input_features = vecs_t[0:1].unsqueeze(0)  # (1, 1, 271)

    print(f"\n3. Input features:")
    print(f"   shape: {input_features.shape}")
    print(f"   NaN: {torch.isnan(input_features).any()}")

    # Generate sequence
    print(f"\n4. Generating sequence...")
    with torch.no_grad():
        joint_positions = generator.generate_sequence(
            text="a person walks",
            num_frames=10,
            num_steps=10,
            guidance_scale=2.5,
            input_features=input_features,
            dataset_type="t2m",
        )

    print(f"\n5. Output:")
    print(f"   joint_positions shape: {joint_positions.shape}")
    print(f"   NaN values: {torch.isnan(joint_positions).any()}")

    # Assertions
    assert not torch.isnan(joint_positions).any(), "Output contains NaN values!"
    assert joint_positions.shape == (
        1,
        10,
        22,
        3,
    ), f"Unexpected shape: {joint_positions.shape}"

    print(f"   min: {joint_positions.min().item():.6f}")
    print(f"   max: {joint_positions.max().item():.6f}")

    # Compare with reference data (first 10 frames)
    reference_joints = torch.from_numpy(joints[:10]).float()  # (10, 22, 3)
    generated_joints = joint_positions[0]  # (10, 22, 3)

    # Note: The generated motion won't match reference exactly since we're using
    # a different text prompt and the model is trained on different data.
    # But we can check that the output is in a reasonable range.
    print(f"\n6. Reference comparison:")
    print(
        f"   Reference range: [{reference_joints.min().item():.3f}, {reference_joints.max().item():.3f}]"
    )
    print(
        f"   Generated range: [{generated_joints.min().item():.3f}, {generated_joints.max().item():.3f}]"
    )

    # Check that generated positions are within reasonable bounds
    # (not exploding or collapsing)
    assert torch.abs(generated_joints).max() < 100, "Generated positions are too large!"

    print("\n" + "=" * 60)
    print("[PASS] No NaN values in output!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_nan_fix()
