import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator


def test_cfg_logic():
    print("Testing CFG Logic...")

    # Initialize models
    encoder = MotionHistoryEncoder(
        frame_feature_dim=263,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=32,
        per_joint_out_dim=32,
        model_dim=64,
    )

    predictor = FlowMatchingPredictor(per_joint_dim=32, model_dim=64)

    generator = HumanMotionGenerator(encoder, predictor)

    B = 2
    text_emb = torch.randn(B, 512)
    hist = torch.randn(B, 15, 263)

    # Test 1: Generate with scale 1.0 (no CFG effect basically)
    out_1 = generator.generate(
        text=text_emb, input_features=hist, guidance_scale=1.0, num_steps=5
    )
    print(f"Output shape (scale 1.0): {out_1.shape}")
    assert out_1.shape == (B, 22, 3)

    # Test 2: Generate with scale 2.5
    out_2 = generator.generate(
        text=text_emb, input_features=hist, guidance_scale=2.5, num_steps=5
    )
    print(f"Output shape (scale 2.5): {out_2.shape}")
    assert out_2.shape == (B, 22, 3)

    # Test 3: Check that outputs differ
    diff = torch.abs(out_1 - out_2).mean().item()
    print(f"Mean Difference between scale 1.0 and 2.5: {diff:.6f}")
    assert diff > 0, "CFG scale should affect the output!"

    # Test 4: Zero-shot (no history)
    out_zs = generator.generate(
        text=text_emb, input_features=None, guidance_scale=2.5, num_steps=5
    )
    print(f"Output shape (Zero-shot): {out_zs.shape}")
    assert out_zs.shape == (B, 22, 3)

    # Test 5: Unconditional (no text)
    # This specifically tests if encoder handles text=None
    out_uncond = generator.generate(
        text=None, input_features=hist, guidance_scale=1.0, num_steps=5
    )
    print(f"Output shape (Unconditional): {out_uncond.shape}")
    assert out_uncond.shape == (B, 22, 3)

    print("CFG Logic Test Passed!")


if __name__ == "__main__":
    test_cfg_logic()
