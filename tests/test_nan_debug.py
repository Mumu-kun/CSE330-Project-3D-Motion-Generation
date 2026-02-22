"""
Debug test for NaN values in HumanMotionGenerator.generate_sequence().

This test loads the checkpoint and sample data to trace where NaN values originate.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
import numpy as np


def test_nan_debug():
    """Debug NaN values in generate_sequence."""
    print("=" * 60)
    print("NaN Debug Test")
    print("=" * 60)

    device = torch.device("cpu")

    # Load sample data
    vec_path = "sample_data/000070_vec.npy"
    joint_path = "sample_data/000070_joint.npy"

    vecs = np.load(vec_path)
    joints = np.load(joint_path)

    print(f"\n1. Loaded sample data:")
    print(f"   vecs shape: {vecs.shape}")
    print(f"   joints shape: {joints.shape}")
    print(f"   vecs NaN: {np.isnan(vecs).any()}")
    print(f"   joints NaN: {np.isnan(joints).any()}")

    # Convert to torch
    vecs_t = torch.from_numpy(vecs).float()
    joints_t = torch.from_numpy(joints).float()

    # Test features_to_positions
    print("\n2. Testing features_to_positions...")
    from src.utils.motion_utils import features_to_positions

    positions = features_to_positions(vecs_t)
    print(f"   positions shape: {positions.shape}")
    print(f"   positions NaN: {torch.isnan(positions).any()}")
    print(f"   positions min: {positions.min().item():.6f}")
    print(f"   positions max: {positions.max().item():.6f}")

    # Test extract_prev_frame_features
    print("\n3. Testing extract_prev_frame_features...")
    from src.utils.train_utils import extract_prev_frame_features

    prev_features = extract_prev_frame_features(vecs_t)
    print(f"   prev_features shape: {prev_features.shape}")
    print(f"   prev_features NaN: {torch.isnan(prev_features).any()}")

    # Test extract_clean_target
    print("\n4. Testing extract_clean_target...")
    from src.utils.train_utils import extract_clean_target

    clean_target = extract_clean_target(vecs_t)
    print(f"   clean_target shape: {clean_target.shape}")
    print(f"   clean_target NaN: {torch.isnan(clean_target).any()}")

    # Test flow_output_to_positions
    print("\n5. Testing flow_output_to_positions...")
    from src.utils.motion_utils import flow_output_to_positions

    # Use first frame as test
    flow_output = clean_target[0:1]  # (1, 72)
    prev_root_pos = joints_t[0, 0:1, :]  # (1, 3) - first frame root position
    prev_root_rot_6d = vecs_t[0, 69:75]  # (6,) - first frame root rotation

    print(f"   flow_output shape: {flow_output.shape}")
    print(f"   prev_root_pos shape: {prev_root_pos.shape}")
    print(f"   prev_root_rot_6d shape: {prev_root_rot_6d.shape}")
    print(f"   flow_output NaN: {torch.isnan(flow_output).any()}")
    print(f"   prev_root_pos NaN: {torch.isnan(prev_root_pos).any()}")
    print(f"   prev_root_rot_6d NaN: {torch.isnan(prev_root_rot_6d).any()}")

    new_positions = flow_output_to_positions(
        flow_output, prev_root_pos, prev_root_rot_6d.unsqueeze(0)
    )
    print(f"   new_positions shape: {new_positions.shape}")
    print(f"   new_positions NaN: {torch.isnan(new_positions).any()}")

    # Test IncrementalFeatureExtractor
    print("\n6. Testing IncrementalFeatureExtractor...")
    from src.utils.motion_utils import IncrementalFeatureExtractor

    extractor = IncrementalFeatureExtractor(device=device)

    # Initialize with first frame
    init_positions = joints_t[0:1]  # (1, 22, 3)
    print(f"   init_positions shape: {init_positions.shape}")
    print(f"   init_positions NaN: {torch.isnan(init_positions).any()}")

    init_features = extractor.initialize(init_positions)
    print(f"   init_features shape: {init_features.shape}")
    print(f"   init_features NaN: {torch.isnan(init_features).any()}")

    # Process second frame
    second_positions = joints_t[1:2]
    second_features = extractor.process_frame(second_positions)
    print(f"   second_features shape: {second_features.shape}")
    print(f"   second_features NaN: {torch.isnan(second_features).any()}")

    # Test process_flow_output
    print("\n7. Testing process_flow_output...")
    extractor.reset()
    _ = extractor.initialize(init_positions)

    prev_frame_271d = vecs_t[0:1]  # First frame 271D
    flow_out = clean_target[1:2]  # Second frame's clean target as flow output

    new_frame_271d, new_positions = extractor.process_flow_output(
        flow_out, prev_frame_271d
    )
    print(f"   new_frame_271d shape: {new_frame_271d.shape}")
    print(f"   new_frame_271d NaN: {torch.isnan(new_frame_271d).any()}")
    print(f"   new_positions shape: {new_positions.shape}")
    print(f"   new_positions NaN: {torch.isnan(new_positions).any()}")

    # Test with checkpoint
    print("\n8. Testing with checkpoint...")
    checkpoint_path = "tests/checkpoints/best.pt"

    if os.path.exists(checkpoint_path):
        from src.models import HumanMotionGenerator
        from src.config import Config

        config = Config()
        # Override max_text_seq_len to match checkpoint
        config.max_text_seq_len = 1
        generator = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path, config, device=str(device)
        )
        generator.eval()

        print(f"   Generator loaded successfully")

        # Test generate_sequence with input_features
        print("\n9. Testing generate_sequence with input_features...")

        # Use first frame as input (B, N, 271) = (1, 1, 271)
        input_features = vecs_t[0:1].unsqueeze(0)  # (1, 1, 271)
        print(f"   input_features shape: {input_features.shape}")

        with torch.no_grad():
            joint_positions = generator.generate_sequence(
                text="a person walks",
                num_frames=5,
                num_steps=10,
                guidance_scale=2.5,
                input_features=input_features,
                dataset_type="t2m",
            )

        print(f"   joint_positions shape: {joint_positions.shape}")
        print(f"   joint_positions NaN: {torch.isnan(joint_positions).any()}")

        if torch.isnan(joint_positions).any():
            # Find which frames have NaN
            for i in range(joint_positions.shape[1]):
                frame = joint_positions[0, i]
                if torch.isnan(frame).any():
                    print(f"   Frame {i} has NaN values")
                    # Find which joints have NaN
                    for j in range(frame.shape[0]):
                        if torch.isnan(frame[j]).any():
                            print(f"     Joint {j} has NaN: {frame[j]}")
        else:
            print(f"   No NaN values found!")
            print(f"   joint_positions min: {joint_positions.min().item():.6f}")
            print(f"   joint_positions max: {joint_positions.max().item():.6f}")
    else:
        print(f"   Checkpoint not found at {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Debug test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_nan_debug()
