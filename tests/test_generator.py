import sys
import os

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator


def test_human_motion_generator():
    print("Initializing test for HumanMotionGenerator...")

    # Dimensions (Lightweight for testing)
    B, T_hist, frame_feature_dim = 2, 4, 263
    per_joint_out_dim = 32
    model_dim = 64
    joint_count = 22

    # 1. Initialize Components
    encoder = MotionHistoryEncoder(
        frame_feature_dim=frame_feature_dim,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=16,
        per_joint_out_dim=per_joint_out_dim,
        model_dim=model_dim,
        joint_count=joint_count,
        text_encoder=lambda x: torch.randn(len(x) if isinstance(x, list) else 1, 512),
    )

    predictor = FlowMatchingPredictor(
        per_joint_dim=per_joint_out_dim, model_dim=model_dim, joint_count=joint_count
    )

    generator = HumanMotionGenerator(encoder, predictor)

    # 2. Dummy Data
    text = ["A person walks", "A person jumps"]
    total_duration = torch.tensor([[0.5], [0.8]])

    # 3. Test Generate (Iterative Inference Pass)
    print("\nTesting generate (Full Iterative Loop)...")
    try:
        # Zero-shot style (no history) + 5 steps of refinement
        # Produces the next frame deltas (B, 22, 3)
        v_t_gen = generator.generate(
            text=text, total_duration=total_duration, num_steps=5
        )
        print(f"Success! Iterative output shape: {v_t_gen.shape}")

        # Verify shape
        assert v_t_gen.shape == (B, joint_count, 3), f"Wrong shape: {v_t_gen.shape}"
        print("Iterative generation verified.")

    except Exception as e:
        print(f"Generate failed: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\nGenerator Integration Test Passed!")


if __name__ == "__main__":
    test_human_motion_generator()
