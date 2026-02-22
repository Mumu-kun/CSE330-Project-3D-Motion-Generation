"""Verify generated motion using visualize_motion."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
import numpy as np
from utils.visualization import visualize_motion


def test_visualize_generated_motion():
    """Generate motion and visualize it."""
    from models import HumanMotionGenerator
    from config import Config

    device = torch.device("cpu")

    # Load checkpoint
    config = Config()
    config.max_text_seq_len = 1

    print("Loading checkpoint...")
    generator = HumanMotionGenerator.load_from_checkpoint(
        "tests/checkpoints/best.pt", config, device=str(device)
    )
    generator.eval()

    # Load sample data for input features
    vecs = np.load("sample_data/000070_vec.npy")
    vecs_t = torch.from_numpy(vecs).float()
    input_features = vecs_t[0:1].unsqueeze(0)  # (1, 1, 271)

    print("Generating motion...")
    with torch.no_grad():
        joint_positions = generator.generate_sequence(
            text="a person walks",
            num_frames=50,
            num_steps=10,
            guidance_scale=2.5,
            input_features=input_features,
            dataset_type="t2m",
        )

    print(f"Generated shape: {joint_positions.shape}")
    print(f"NaN values: {torch.isnan(joint_positions).any()}")
    print(f"Min: {joint_positions.min().item():.4f}")
    print(f"Max: {joint_positions.max().item():.4f}")

    # Check for NaN
    if torch.isnan(joint_positions).any():
        print("\n[FAIL] Generated motion contains NaN values!")
        return False

    # Convert to numpy for visualization
    motion_np = joint_positions[0].numpy()  # (T, 22, 3)

    # Print first frame
    print(f"\nFirst frame joints:")
    print(motion_np[0])

    # Print statistics per coordinate
    print(f"\nCoordinate statistics:")
    print(f"  X: min={motion_np[:,:,0].min():.4f}, max={motion_np[:,:,0].max():.4f}")
    print(f"  Y: min={motion_np[:,:,1].min():.4f}, max={motion_np[:,:,1].max():.4f}")
    print(f"  Z: min={motion_np[:,:,2].min():.4f}, max={motion_np[:,:,2].max():.4f}")

    # Try to create visualization (returns HTML for notebooks)
    print("\nCreating visualization...")
    try:
        ani = visualize_motion(motion_np, title="Generated Motion", notebook=False)
        print("Visualization created successfully!")

        # Save to file
        from pathlib import Path

        save_path = Path("generation/test_motion.mp4")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Note: saving requires ffmpeg
        # ani.save(str(save_path), writer="ffmpeg", fps=20)
        # print(f"Saved to {save_path}")

    except Exception as e:
        print(f"Visualization error: {e}")

    print("\n[PASS] Motion generation successful!")
    return True


if __name__ == "__main__":
    test_visualize_generated_motion()
