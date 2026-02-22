"""
Verification script for flow_output_to_positions and flow_output_to_displacements.

This script verifies the reconstruction functions using sample data:
- sample_data/000070_joint.npy: Global joint positions (N, 22, 3)
- sample_data/000070_vec.npy: 271D feature vectors (N, 271)

The verification process:
1. Load the 271D vectors and joint positions
2. Extract 72D flow output from 271D vectors for each frame
3. Use flow_output_to_positions to reconstruct positions frame by frame
4. Compare reconstructed positions with original joint positions
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from src.utils.motion_utils import (
    flow_output_to_positions,
    flow_output_to_displacements,
    preprocess_sequence,
    features_to_positions,
    cont6d_to_quaternion,
    qrot,
    qinv,
)


def extract_flow_output_from_271d(features_271d: torch.Tensor) -> torch.Tensor:
    """
    Extract 72D flow output from 271D features.

    271D Format:
        [0:3] Root height Y, velocity X, velocity Z
        [3:69] 22 RIC positions (66D)
        [69:201] 22 6D rotations (132D)
        [201:267] 22 local velocities (66D)
        [267:271] Foot contacts (4D)

    72D Flow Output Format:
        [0:9] Root features: height (1D) + velocity (2D) + rotation_6d (6D)
        [9:72] Joint RIC positions: 21 non-root joints × 3D = 63D

    Args:
        features_271d: (N, 271) feature vectors

    Returns:
        flow_output: (N, 72) flow output vectors
    """
    N = features_271d.shape[0]
    flow_output = torch.zeros(N, 72, dtype=features_271d.dtype)

    # Root height (1D)
    flow_output[:, 0:1] = features_271d[:, 0:1]

    # Root velocity X, Z (2D)
    flow_output[:, 1:3] = features_271d[:, 1:3]

    # Root rotation 6D (6D) - from rotations_6d[0] (root joint)
    flow_output[:, 3:9] = features_271d[:, 69:75]

    # Joint RIC positions (63D) - RIC for joints 1-21 (excluding root)
    # RIC positions are at [3:69], root is at [3:6], joints 1-21 are at [6:69]
    flow_output[:, 9:72] = features_271d[:, 6:69]

    return flow_output


def verify_reconstruction(joint_path: str, vec_path: str):
    """
    Verify reconstruction functions using sample data.

    Args:
        joint_path: Path to joint positions .npy file
        vec_path: Path to 271D vectors .npy file
    """
    print("=" * 60)
    print("Reconstruction Verification")
    print("=" * 60)

    # Load data
    joints = np.load(joint_path)
    vecs = np.load(vec_path)

    print(f"\nLoaded data:")
    print(f"  Joints shape: {joints.shape}")
    print(f"  Vectors shape: {vecs.shape}")

    # Convert to torch tensors
    joints_t = torch.from_numpy(joints).float()
    vecs_t = torch.from_numpy(vecs).float()

    N = joints.shape[0]

    # =========================================================================
    # Test 1: Verify features_to_positions (271D -> positions)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 1: features_to_positions (271D -> positions)")
    print("-" * 60)

    reconstructed_positions = features_to_positions(vecs_t)
    print(f"  Reconstructed positions shape: {reconstructed_positions.shape}")

    # Compare with original joints
    position_error = torch.mean((reconstructed_positions - joints_t) ** 2).item()
    position_mae = torch.mean(torch.abs(reconstructed_positions - joints_t)).item()
    max_error = torch.max(torch.abs(reconstructed_positions - joints_t)).item()

    print(f"  MSE: {position_error:.6f}")
    print(f"  MAE: {position_mae:.6f}")
    print(f"  Max error: {max_error:.6f}")

    # Per-joint error analysis
    per_joint_error = torch.mean(
        (reconstructed_positions - joints_t) ** 2, dim=(0, 2)
    )  # (22,)
    print(f"\n  Per-joint MSE (top 5 worst):")
    sorted_idx = torch.argsort(per_joint_error, descending=True)
    for i in sorted_idx[:5]:
        print(f"    Joint {i:2d}: {per_joint_error[i].item():.6f}")

    # =========================================================================
    # Test 2: Verify flow_output_to_positions (frame-by-frame reconstruction)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 2: flow_output_to_positions (frame-by-frame)")
    print("-" * 60)

    # Extract flow outputs from 271D vectors
    flow_outputs = extract_flow_output_from_271d(vecs_t)
    print(f"  Flow outputs shape: {flow_outputs.shape}")

    # Frame-by-frame reconstruction
    reconstructed_frames = []

    # Initial state: use first frame's root position and rotation
    prev_root_pos = joints_t[0, 0, :].unsqueeze(0)  # (1, 3)
    prev_root_rot_6d = vecs_t[0, 69:75].unsqueeze(0)  # (1, 6)

    for i in range(N):
        flow_out = flow_outputs[i : i + 1]  # (1, 72)

        # Reconstruct positions
        positions = flow_output_to_positions(flow_out, prev_root_pos, prev_root_rot_6d)
        reconstructed_frames.append(positions)

        # Update state for next frame
        prev_root_pos = positions[0, 0, :].unsqueeze(0)  # New root position
        prev_root_rot_6d = flow_out[0, 3:9].unsqueeze(0)  # New root rotation

    reconstructed_frames = torch.cat(reconstructed_frames, dim=0)  # (N, 22, 3)

    # Compare with original joints
    flow_error = torch.mean((reconstructed_frames - joints_t) ** 2).item()
    flow_mae = torch.mean(torch.abs(reconstructed_frames - joints_t)).item()
    flow_max_error = torch.max(torch.abs(reconstructed_frames - joints_t)).item()

    print(f"  MSE: {flow_error:.6f}")
    print(f"  MAE: {flow_mae:.6f}")
    print(f"  Max error: {flow_max_error:.6f}")

    # Per-joint error analysis
    flow_per_joint_error = torch.mean(
        (reconstructed_frames - joints_t) ** 2, dim=(0, 2)
    )
    print(f"\n  Per-joint MSE (top 5 worst):")
    sorted_idx = torch.argsort(flow_per_joint_error, descending=True)
    for i in sorted_idx[:5]:
        print(f"    Joint {i:2d}: {flow_per_joint_error[i].item():.6f}")

    # =========================================================================
    # Test 3: Verify flow_output_to_displacements
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 3: flow_output_to_displacements")
    print("-" * 60)

    displacements = flow_output_to_displacements(flow_outputs)
    print(f"  Displacements shape: {displacements.shape}")

    # Check displacement statistics
    print(f"\n  Displacement statistics:")
    print(f"    Mean: {torch.mean(displacements).item():.6f}")
    print(f"    Std: {torch.std(displacements).item():.6f}")
    print(f"    Min: {torch.min(displacements).item():.6f}")
    print(f"    Max: {torch.max(displacements).item():.6f}")

    # Root displacement should match velocity in 271D
    root_disp_x = displacements[:, 0, 0]
    root_disp_z = displacements[:, 0, 2]
    expected_vx = vecs_t[:, 1]
    expected_vz = vecs_t[:, 2]

    vx_error = torch.mean((root_disp_x - expected_vx) ** 2).item()
    vz_error = torch.mean((root_disp_z - expected_vz) ** 2).item()

    print(f"\n  Root velocity verification:")
    print(f"    VX MSE: {vx_error:.6f}")
    print(f"    VZ MSE: {vz_error:.6f}")

    # =========================================================================
    # Test 4: Cumulative root position tracking
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 4: Cumulative root position tracking")
    print("-" * 60)

    # Reconstruct root trajectory from velocities
    cumulative_x = torch.cumsum(vecs_t[:, 1], dim=0)
    cumulative_z = torch.cumsum(vecs_t[:, 2], dim=0)

    # Original root trajectory
    original_x = joints_t[:, 0, 0]
    original_z = joints_t[:, 0, 2]

    # Compare (need to account for initial position)
    cumulative_x = cumulative_x + joints_t[0, 0, 0]  # Add initial X
    cumulative_z = cumulative_z + joints_t[0, 0, 2]  # Add initial Z

    x_error = torch.mean((cumulative_x - original_x) ** 2).item()
    z_error = torch.mean((cumulative_z - original_z) ** 2).item()

    print(f"  Root X trajectory MSE: {x_error:.6f}")
    print(f"  Root Z trajectory MSE: {z_error:.6f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\n  features_to_positions reconstruction:")
    print(f"    MSE: {position_error:.6f}, MAE: {position_mae:.6f}")

    print(f"\n  flow_output_to_positions reconstruction:")
    print(f"    MSE: {flow_error:.6f}, MAE: {flow_mae:.6f}")

    if position_error < 1e-4:
        print("\n  [PASS] features_to_positions: EXCELLENT reconstruction quality")
    elif position_error < 1e-2:
        print("\n  [PASS] features_to_positions: GOOD reconstruction quality")
    else:
        print(
            "\n  [FAIL] features_to_positions: POOR reconstruction quality - needs investigation"
        )

    if flow_error < 1e-4:
        print("  [PASS] flow_output_to_positions: EXCELLENT reconstruction quality")
    elif flow_error < 1e-2:
        print("  [PASS] flow_output_to_positions: GOOD reconstruction quality")
    else:
        print(
            "  [FAIL] flow_output_to_positions: POOR reconstruction quality - needs investigation"
        )

    return {
        "features_to_positions_mse": position_error,
        "flow_output_to_positions_mse": flow_error,
    }


if __name__ == "__main__":
    joint_path = "sample_data/000070_joint.npy"
    vec_path = "sample_data/000070_vec.npy"

    results = verify_reconstruction(joint_path, vec_path)
