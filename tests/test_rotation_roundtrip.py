"""
Test rotation roundtrip for flow_output_to_271d.

This test verifies that:
1. features_to_positions correctly reconstructs joint positions from 271D features
2. The new IK-based rotation computation in flow_output_to_271d produces consistent rotations
3. Roundtrip: positions -> features -> positions preserves the original positions
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import (
    features_to_positions,
    flow_output_to_271d,
    get_dataset_config,
    _compute_ik,
)
from utils.quaternion import quaternion_to_cont6d, cont6d_to_quaternion


def test_features_to_positions_roundtrip():
    """Test that features_to_positions reconstructs positions from sample data."""
    print("=" * 60)
    print("Test 1: features_to_positions reconstruction")
    print("=" * 60)

    # Load sample data
    vec_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_vec.npy"
    )
    joint_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_joint.npy"
    )

    features = np.load(vec_path)  # (N, 271)
    joints = np.load(joint_path)  # (N, 22, 3)

    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded joints shape: {joints.shape}")

    # Convert to torch
    features_t = torch.from_numpy(features).float()
    joints_t = torch.from_numpy(joints).float()

    # Reconstruct positions
    reconstructed = features_to_positions(features_t)  # (N, 22, 3)

    # Compare
    diff = reconstructed - joints_t
    mse = torch.mean(diff**2).item()
    max_error = torch.max(torch.abs(diff)).item()

    print(f"\nReconstruction MSE: {mse:.6e}")
    print(f"Max absolute error: {max_error:.6e}")

    if mse < 1e-6:
        print("[PASS] features_to_positions reconstruction is accurate")
    else:
        print("[FAIL] features_to_positions has significant error")

    return mse, max_error


def test_ik_rotation_consistency():
    """Test that IK produces consistent rotations from positions."""
    print("\n" + "=" * 60)
    print("Test 2: IK rotation consistency")
    print("=" * 60)

    # Load sample data
    joint_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_joint.npy"
    )
    joints = np.load(joint_path)  # (N, 22, 3)
    joints_t = torch.from_numpy(joints).float()

    # Get skeleton config
    config = get_dataset_config("t2m")
    raw_offsets = config["raw_offsets"].float()
    kinematic_chain = config["kinematic_chain"]
    face_joint_indx = config["face_joint_indx"]

    # Compute IK for all frames
    quaternions = _compute_ik(joints_t, raw_offsets, kinematic_chain, face_joint_indx)
    rotations_6d = quaternion_to_cont6d(quaternions)

    print(f"Input positions shape: {joints_t.shape}")
    print(f"Output quaternions shape: {quaternions.shape}")
    print(f"Output rotations_6d shape: {rotations_6d.shape}")

    # Check quaternion norms (should be ~1)
    quat_norms = torch.norm(quaternions, dim=-1)
    mean_norm = torch.mean(quat_norms).item()
    norm_std = torch.std(quat_norms).item()

    print(f"\nQuaternion norms - Mean: {mean_norm:.6f}, Std: {norm_std:.6e}")

    if abs(mean_norm - 1.0) < 0.01 and norm_std < 0.01:
        print("[PASS] Quaternions are properly normalized")
    else:
        print("[FAIL] Quaternion normalization issue")

    return quaternions, rotations_6d


def test_flow_output_to_271d_rotation():
    """Test that flow_output_to_271d produces consistent rotations."""
    print("\n" + "=" * 60)
    print("Test 3: flow_output_to_271d rotation computation")
    print("=" * 60)

    # Load sample data
    vec_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_vec.npy"
    )
    joint_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_joint.npy"
    )

    features = np.load(vec_path)  # (N, 271)
    joints = np.load(joint_path)  # (N, 22, 3)

    features_t = torch.from_numpy(features).float()
    joints_t = torch.from_numpy(joints).float()

    # Extract flow output (72D) from frame 1
    # flow_output format: [0:9] root (height, vel, rot_6d), [9:72] joint_ric_21
    frame_0 = features_t[0:1]  # (1, 271)
    frame_1 = features_t[1:2]  # (1, 271)

    # Build flow_output from frame_1
    root_height = frame_1[:, 0:1]
    root_vel = frame_1[:, 1:3]
    root_rot_6d = frame_1[:, 69:75]

    # RIC layout: [3:69] is 22x3=66D, so joints 1-21 are at [6:69] (63D)
    joint_ric_21 = frame_1[:, 6:69].reshape(1, 21, 3)

    flow_output = torch.cat(
        [root_height, root_vel, root_rot_6d, joint_ric_21.reshape(1, -1)], dim=-1
    )

    print(f"flow_output shape: {flow_output.shape}")

    # Get prev_root_pos from frame 0
    # Root position needs to be computed from velocity form
    root_pos_x = torch.cumsum(features_t[:1, 1:2], dim=0)  # Just first frame
    root_pos_z = torch.cumsum(features_t[:1, 2:3], dim=0)
    root_height_0 = features_t[:1, 0:1]
    prev_root_pos = torch.cat([root_pos_x, root_height_0, root_pos_z], dim=-1)  # (1, 3)

    # Call flow_output_to_271d
    new_frame, new_root_pos = flow_output_to_271d(
        flow_output=flow_output,
        prev_frame=frame_0,
        prev_root_pos=prev_root_pos,
    )

    print(f"new_frame shape: {new_frame.shape}")
    print(f"new_root_pos shape: {new_root_pos.shape}")

    # Extract rotations from new_frame
    new_rotations_6d = new_frame[:, 69:201].reshape(1, 22, 6)
    expected_rotations_6d = frame_1[:, 69:201].reshape(1, 22, 6)

    # Compare rotations
    rot_diff = new_rotations_6d - expected_rotations_6d
    rot_mse = torch.mean(rot_diff**2).item()
    rot_max_error = torch.max(torch.abs(rot_diff)).item()

    print(f"\nRotation comparison:")
    print(f"  MSE: {rot_mse:.6e}")
    print(f"  Max error: {rot_max_error:.6e}")

    # Also compare full frames
    frame_diff = new_frame - frame_1
    frame_mse = torch.mean(frame_diff**2).item()
    frame_max_error = torch.max(torch.abs(frame_diff)).item()

    print(f"\nFull frame comparison:")
    print(f"  MSE: {frame_mse:.6e}")
    print(f"  Max error: {frame_max_error:.6e}")

    # Check if rotations are close
    # Note: IK may produce different but equivalent rotations, so we check position consistency
    # Reconstruct positions from new_frame and compare
    new_positions = features_to_positions(new_frame)
    expected_positions = joints_t[1:2]

    pos_diff = new_positions - expected_positions
    pos_mse = torch.mean(pos_diff**2).item()
    pos_max_error = torch.max(torch.abs(pos_diff)).item()

    print(f"\nPosition consistency (from new_frame):")
    print(f"  MSE: {pos_mse:.6e}")
    print(f"  Max error: {pos_max_error:.6e}")

    if pos_mse < 1e-4:
        print("[PASS] flow_output_to_271d produces position-consistent rotations")
    else:
        print("[FAIL] Position inconsistency detected")

    return new_frame, new_rotations_6d


def test_full_sequence_roundtrip():
    """Test roundtrip over full sequence using flow_output_to_271d incrementally."""
    print("\n" + "=" * 60)
    print("Test 4: Full sequence roundtrip (autoregressive simulation)")
    print("=" * 60)

    # Load sample data
    vec_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_vec.npy"
    )
    joint_path = os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "000070_joint.npy"
    )

    features = np.load(vec_path)  # (N, 271)
    joints = np.load(joint_path)  # (N, 22, 3)

    features_t = torch.from_numpy(features).float()
    joints_t = torch.from_numpy(joints).float()

    N = features_t.shape[0]
    print(f"Sequence length: {N}")

    # Initialize tracking
    # Root position from first frame (velocity form, so cumsum)
    root_pos_x = torch.cumsum(features_t[:, 1:2], dim=0)
    root_pos_z = torch.cumsum(features_t[:, 2:3], dim=0)
    root_height = features_t[:, 0:1]
    root_positions = torch.cat([root_pos_x, root_height, root_pos_z], dim=-1)  # (N, 3)

    # Process each frame
    all_new_frames = []
    current_root_pos = root_positions[0:1]  # (1, 3)

    for i in range(1, N):
        # Build flow_output from features[i]
        frame = features_t[i : i + 1]
        root_height_i = frame[:, 0:1]
        root_vel_i = frame[:, 1:3]
        root_rot_6d_i = frame[:, 69:75]
        joint_ric_21_i = frame[:, 6:69].reshape(1, 21, 3)

        flow_output = torch.cat(
            [root_height_i, root_vel_i, root_rot_6d_i, joint_ric_21_i.reshape(1, -1)],
            dim=-1,
        )

        # Get previous frame
        prev_frame = features_t[i - 1 : i]

        # Call flow_output_to_271d
        new_frame, current_root_pos = flow_output_to_271d(
            flow_output=flow_output,
            prev_frame=prev_frame,
            prev_root_pos=current_root_pos,
        )
        all_new_frames.append(new_frame)

    # Stack all frames
    new_features = torch.cat([features_t[0:1]] + all_new_frames, dim=0)  # (N, 271)

    # Reconstruct positions from new features
    reconstructed_positions = features_to_positions(new_features)

    # Compare with original
    pos_diff = reconstructed_positions - joints_t
    pos_mse = torch.mean(pos_diff**2).item()
    pos_max_error = torch.max(torch.abs(pos_diff)).item()

    print(f"\nFull sequence reconstruction:")
    print(f"  Position MSE: {pos_mse:.6e}")
    print(f"  Max position error: {pos_max_error:.6e}")

    # Also check feature differences
    feat_diff = new_features - features_t
    feat_mse = torch.mean(feat_diff**2).item()
    feat_max_error = torch.max(torch.abs(feat_diff)).item()

    print(f"  Feature MSE: {feat_mse:.6e}")
    print(f"  Max feature error: {feat_max_error:.6e}")

    if pos_mse < 1e-4:
        print("[PASS] Full sequence roundtrip is accurate")
    else:
        print("[FAIL] Full sequence roundtrip has significant error")

    return new_features, reconstructed_positions


if __name__ == "__main__":
    print("Testing rotation roundtrip for flow_output_to_271d\n")

    test_features_to_positions_roundtrip()
    test_ik_rotation_consistency()
    test_flow_output_to_271d_rotation()
    test_full_sequence_roundtrip()

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
