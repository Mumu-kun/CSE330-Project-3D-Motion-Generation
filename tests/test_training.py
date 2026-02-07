import sys
import os

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.models import MotionHistoryEncoder, FlowMatchingPredictor
from src.train_utils import train, build_prev_and_clean_diffs


def test_feature_extraction():
    print("\n--- Testing Feature Extraction ---")
    B, T_hist, C = 2, 10, 263
    hist = torch.randn(B, T_hist, C)
    future = torch.randn(B, 1, C)

    prev_pos, prev_rot6d, prev_v, clean_v = build_prev_and_clean_diffs(hist, future)

    print(f"prev_pos shape: {prev_pos.shape}")
    print(f"prev_rot6d shape: {prev_rot6d.shape}")
    print(f"prev_v shape: {prev_v.shape}")
    print(f"clean_v shape: {clean_v.shape}")

    assert prev_pos.shape == (B, 22, 3)
    assert prev_rot6d.shape == (B, 22, 6)
    assert clean_v.shape == (B, 22, 3)

    # Check root identity for rotation (Index 0)
    # [1, 0, 0, 0, 1, 0]
    expected_root_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert torch.allclose(prev_rot6d[:, 0], expected_root_rot.expand(B, -1), atol=1e-5)
    print("Feature extraction verification PASSED.")


def test_training_loop():
    print("\n--- Testing Training Loop ---")

    # 1. Dimensions
    B, T, C = 4, 100, 263
    per_joint_out_dim = 32
    model_dim = 64
    joint_count = 22

    # 2. Dummy Data (Mimicking dataloader output)
    batch = {
        "motion": torch.randn(B, T, C),
        "text_clip": torch.randn(B, 512),
        "lengths": torch.tensor([80, 90, 100, 70]),
        "duration": torch.rand(B, 1),
    }

    dataloader = [batch]  # Single batch iterable

    # 3. Models
    encoder = MotionHistoryEncoder(
        frame_feature_dim=C,
        text_embedding_dim=512,
        joint_feature_projection_dim=32,
        text_projection_dim=16,
        per_joint_out_dim=per_joint_out_dim,
        model_dim=model_dim,
        joint_count=joint_count,
    )

    predictor = FlowMatchingPredictor(
        per_joint_dim=per_joint_out_dim, model_dim=model_dim, joint_count=joint_count
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Run 5 steps
    print(f"Running 5 steps of training on {device}...")
    try:
        ema_mhe, ema_fmp = train(
            encoder, predictor, dataloader, num_steps=5, device=device, lr=1e-4
        )
        print("Training steps completed successfully!")

        # Verify EMA
        assert not next(ema_mhe.parameters()).requires_grad
        assert not next(ema_fmp.parameters()).requires_grad
        print("EMA validation PASSED.")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_feature_extraction()
    test_training_loop()
    print("\nAll Training Utilities verified!")
