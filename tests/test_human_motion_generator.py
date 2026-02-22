"""
Test HumanMotionGenerator with updated MotionHistoryEncoder output shape.

Tests that the generate_sequence method correctly handles the encoder's
(B, T_hist, 22, out_dim) output by extracting the last frame context.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
from src.models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator


def test_human_motion_generator():
    """Test HumanMotionGenerator with the updated encoder output shape."""
    print("=" * 60)
    print("Testing HumanMotionGenerator")
    print("=" * 60)

    device = torch.device("cpu")

    # Configuration
    frame_feature_dim = 271
    text_embedding_dim = 512
    per_joint_dim = 32
    model_dim = 256

    print("\n1. Creating MotionHistoryEncoder...")
    encoder = MotionHistoryEncoder(
        frame_feature_dim=frame_feature_dim,
        text_embedding_dim=text_embedding_dim,
        per_joint_out_dim=per_joint_dim,
        joint_count=22,
        model_dim=model_dim,
        num_layers=2,  # Fewer layers for testing
    ).to(device)
    print(
        f"   Encoder created with model_dim={model_dim}, per_joint_out_dim={per_joint_dim}"
    )

    print("\n2. Creating FlowMatchingPredictor...")
    predictor = FlowMatchingPredictor(
        per_joint_dim=per_joint_dim,
        model_dim=64,
        num_layers=1,  # Fewer layers for testing
        joint_count=22,
    ).to(device)
    print(f"   Predictor created with per_joint_dim={per_joint_dim}")

    print("\n3. Creating HumanMotionGenerator...")
    generator = HumanMotionGenerator(
        encoder=encoder,
        predictor=predictor,
    ).to(device)
    print("   Generator created successfully")

    print("\n4. Testing encoder output shape with various T_hist values...")
    B = 2
    text_emb = torch.randn(B, 1, text_embedding_dim, device=device)

    for T_hist in [1, 4, 8, 16]:
        history = torch.randn(B, T_hist, frame_feature_dim, device=device)

        encoder_out = encoder(
            batch_size=B,
            text=text_emb,
            input_features=history,
        )
        print(f"   T_hist={T_hist}: Encoder output shape: {encoder_out.shape}")
        assert encoder_out.shape == (
            B,
            T_hist,
            22,
            per_joint_dim,
        ), f"Encoder output shape mismatch for T_hist={T_hist}: {encoder_out.shape}"

        # Test last frame extraction
        context = encoder_out[:, -1, :, :]
        assert context.shape == (
            B,
            22,
            per_joint_dim,
        ), f"Context shape mismatch for T_hist={T_hist}: {context.shape}"

    print("   All T_hist values - PASS")

    print("\n6. Testing generate_sequence with small number of frames...")
    num_frames = 3
    num_steps = 2

    joint_positions = generator.generate_sequence(
        text=text_emb,
        num_frames=num_frames,
        num_steps=num_steps,
        guidance_scale=2.5,
        input_features=None,  # Use null history
        dataset_type="t2m",
    )

    print(f"   Generated joint positions shape: {joint_positions.shape}")
    print(f"   Expected: (B={B}, num_frames={num_frames}, 22, 3)")
    assert joint_positions.shape == (
        B,
        num_frames,
        22,
        3,
    ), f"Generated shape mismatch: {joint_positions.shape}"
    print("   generate_sequence - PASS")

    print("\n7. Testing with initial input features...")
    init_features = torch.randn(B, frame_feature_dim, device=device)

    joint_positions_with_init = generator.generate_sequence(
        text=text_emb,
        num_frames=num_frames,
        num_steps=num_steps,
        guidance_scale=2.5,
        input_features=init_features,
        dataset_type="t2m",
    )

    print(f"   Generated with init features shape: {joint_positions_with_init.shape}")
    assert joint_positions_with_init.shape == (
        B,
        num_frames,
        22,
        3,
    ), f"Generated shape mismatch: {joint_positions_with_init.shape}"
    print("   generate_sequence with init features - PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_human_motion_generator()
