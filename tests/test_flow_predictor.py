import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from src.models import FlowMatchingPredictor


def test_flow_matching_predictor():
    print("Initializing test for FlowMatchingPredictor...")

    # Parameters
    B = 2
    N_joints = 22
    per_joint_dim = 32
    model_dim = 64
    num_layers = 2

    # Initialize Model
    model = FlowMatchingPredictor(
        per_joint_dim=per_joint_dim, model_dim=model_dim, num_layers=num_layers
    )

    # Dummy Inputs
    history_features = torch.randn(B, N_joints, per_joint_dim)
    prev_frame_features = torch.randn(B, N_joints, 12)  # pos(3) + rot(6) + diffs(3)

    noise_level = torch.rand(B)
    noisy_target_diffs = torch.randn(B, N_joints, 3)
    temporal_progress = torch.tensor([0.1, 0.9])  # Progress through generation

    # 1. Forward Pass (Zero-shot Inference style)
    # Only provides history and noise_level. Relies on null tokens and internal sampling.
    print(f"Running forward pass (Zero-shot / All defaults)...")
    try:
        pred_noise = model(
            history_features=history_features,
            noise_level=noise_level,
        )
        print(f"Success! Output shape: {pred_noise.shape}")
    except Exception as e:
        print(f"Zero-shot forward pass failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # 2. Forward Pass (Full Conditioning)
    print(f"Running forward pass with full conditioning...")
    try:
        pred_noise = model(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target_diffs=noisy_target_diffs,
            prev_frame_features=prev_frame_features,
            temporal_progress=temporal_progress,
        )
        print(f"Success! Output shape: {pred_noise.shape}")

        # Verify output shape
        expected_shape = (B, N_joints, 3)
        if pred_noise.shape == expected_shape:
            print("Output shape is correct.")
        else:
            print(f"Error: Expected shape {expected_shape}, got {pred_noise.shape}")

        # Total parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}")

    except Exception as e:
        print(f"Full conditioning forward pass failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_flow_matching_predictor()
