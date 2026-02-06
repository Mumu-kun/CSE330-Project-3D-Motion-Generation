import sys
import os

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.models import MotionHistoryEncoder


def test_motion_history_encoder():
    print("Initializing test for MotionHistoryEncoder...")

    # Parameters (Reduced for testing)
    B = 2
    T_hist = 5
    frame_feature_dim = 263
    text_embedding_dim = 512
    joint_feature_projection_dim = 32
    text_projection_dim = 16
    per_joint_out_dim = 32
    joint_count = 22
    model_dim = 64

    # Mock Text Encoder
    class MockTextEncoder(nn.Module):
        def forward(self, text):
            # Return random embeddings for text
            if isinstance(text, str):
                return torch.randn(1, 512)
            else:
                return torch.randn(len(text), 512)

    text_encoder = MockTextEncoder()

    # Initialize Model
    model = MotionHistoryEncoder(
        frame_feature_dim=frame_feature_dim,
        text_embedding_dim=text_embedding_dim,
        joint_feature_projection_dim=joint_feature_projection_dim,
        text_projection_dim=text_projection_dim,
        per_joint_out_dim=per_joint_out_dim,
        joint_count=joint_count,
        model_dim=model_dim,
        text_encoder=text_encoder,
    )

    # Dummy Input
    input_features = torch.randn(B, T_hist, frame_feature_dim)
    text = ["A person is walking", "A person is jumping"]

    # Forward Pass
    print(f"Running forward pass with input shape: {input_features.shape}...")
    try:
        output = model(input_features, text)
        print(f"Success! Output shape: {output.shape}")

        # Verify output shape
        expected_shape = (B, joint_count, per_joint_out_dim)
        if output.shape == expected_shape:
            print("Output shape is correct.")
        else:
            print(f"Error: Expected shape {expected_shape}, got {output.shape}")

        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal Trainable Parameters: {total_params:,}")

    except Exception as e:
        print(f"Forward pass failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_motion_history_encoder()
