"""
Test script for flow_output_to_271d and RootPositionTracker reconstruction.

This script verifies the incremental reconstruction approach using sample data:
- sample_data/000000_joint.npy: Global joint positions (N, 22, 3)
- sample_data/000000_vec.npy: 271D feature vectors (N, 271)

The verification process:
1. Load the 271D vectors and joint positions
2. Extract 72D flow output from 271D vectors for each frame
3. Use flow_output_to_271d with RootPositionTracker to reconstruct 271D frames
4. Compare reconstructed positions with original joint positions
"""

import sys
import os

# Add both project root and src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import torch
import numpy as np
from utils.motion_utils import (
    flow_output_to_271d,
    RootPositionTracker,
    features_to_positions,
    preprocess_sequence,
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


def convert_263d_to_271d(features_263d: torch.Tensor) -> torch.Tensor:
    """
    Convert 263D features to 271D format by adding foot contacts.

    263D Format:
        [0:3] Root height Y, velocity X, velocity Z
        [3:69] 22 RIC positions (66D)
        [69:201] 22 6D rotations (132D)
        [201:263] Local velocities (62D)

    271D Format:
        [0:3] Root height Y, velocity X, velocity Z
        [3:69] 22 RIC positions (66D)
        [69:201] 22 6D rotations (132D)
        [201:267] 22 local velocities (66D)
        [267:271] Foot contacts (4D)
    """
    N = features_263d.shape[0]
    features_271d = torch.zeros(N, 271, dtype=features_263d.dtype)

    # Copy common parts
    features_271d[:, 0:201] = features_263d[:, 0:201]  # Root + RIC + rotations

    # Local velocities: 263D has 62D, 271D has 66D
    # Pad with zeros for the missing 4D
    features_271d[:, 201:263] = features_263d[:, 201:263]  # Copy existing velocities
    # features_271d[:, 263:267] = 0  # Missing 4D (already zeros)

    # Foot contacts (not in 263D, set to zeros)
    # features_271d[:, 267:271] = 0  # Already zeros

    return features_271d


def verify_incremental_reconstruction(joint_path: str, vec_path: str):
    """
    Verify incremental reconstruction using flow_output_to_271d and RootPositionTracker.

    Args:
        joint_path: Path to joint positions .npy file
        vec_path: Path to 271D vectors .npy file
    """
    print("=" * 60)
    print("Incremental Reconstruction Verification")
    print("Using flow_output_to_271d + RootPositionTracker")
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

    # Handle 263D format by converting to 271D
    if vecs_t.shape[1] == 263:
        print("  Converting 263D features to 271D format...")
        vecs_t = convert_263d_to_271d(vecs_t)
        print(f"  Converted vectors shape: {vecs_t.shape}")

    N = joints.shape[0]

    # =========================================================================
    # Test 1: Baseline - features_to_positions (271D -> positions)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 1: Baseline - features_to_positions (271D -> positions)")
    print("-" * 60)

    baseline_positions = features_to_positions(vecs_t)
    baseline_mse = torch.mean((baseline_positions - joints_t) ** 2).item()
    baseline_mae = torch.mean(torch.abs(baseline_positions - joints_t)).item()

    print(f"  MSE: {baseline_mse:.6f}")
    print(f"  MAE: {baseline_mae:.6f}")

    # =========================================================================
    # Test 2: Incremental reconstruction with flow_output_to_271d
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 2: Incremental reconstruction with flow_output_to_271d")
    print("-" * 60)

    # Extract flow outputs from 271D vectors
    flow_outputs = extract_flow_output_from_271d(vecs_t)
    print(f"  Flow outputs shape: {flow_outputs.shape}")

    # Initialize with first frame
    first_frame = vecs_t[0:1]  # (1, 271)

    # Initialize RootPositionTracker from first frame
    # First, get the initial root position from the joints
    initial_root_pos = joints_t[0, 0, :].unsqueeze(0)  # (1, 3)
    root_tracker = RootPositionTracker(initial_root_pos)

    # Reconstructed 271D frames
    reconstructed_frames = [first_frame]

    # Incremental reconstruction
    for i in range(1, N):
        prev_frame = reconstructed_frames[-1]  # (1, 271)
        prev_root_pos = root_tracker.get()  # (1, 3)
        flow_out = flow_outputs[i : i + 1]  # (1, 72)

        # Use flow_output_to_271d for incremental reconstruction
        new_frame, new_root_pos = flow_output_to_271d(
            flow_output=flow_out,
            prev_frame=prev_frame,
            prev_root_pos=prev_root_pos,
        )

        # Update tracker
        root_tracker.update(new_frame)

        reconstructed_frames.append(new_frame)

    # Stack all frames
    reconstructed_271d = torch.cat(reconstructed_frames, dim=0)  # (N, 271)
    print(f"  Reconstructed 271D shape: {reconstructed_271d.shape}")

    # Convert to positions
    reconstructed_positions = features_to_positions(reconstructed_271d)
    print(f"  Reconstructed positions shape: {reconstructed_positions.shape}")

    # Compare with original joints
    incremental_mse = torch.mean((reconstructed_positions - joints_t) ** 2).item()
    incremental_mae = torch.mean(torch.abs(reconstructed_positions - joints_t)).item()
    incremental_max = torch.max(torch.abs(reconstructed_positions - joints_t)).item()

    print(f"  MSE: {incremental_mse:.6f}")
    print(f"  MAE: {incremental_mae:.6f}")
    print(f"  Max error: {incremental_max:.6f}")

    # =========================================================================
    # Test 3: RootPositionTracker accuracy
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 3: RootPositionTracker accuracy")
    print("-" * 60)

    # Reinitialize tracker and track through all frames
    root_tracker2 = RootPositionTracker(initial_root_pos)
    tracked_positions = [initial_root_pos]

    for i in range(1, N):
        root_tracker2.update(vecs_t[i : i + 1])
        tracked_positions.append(root_tracker2.get().clone())

    tracked_positions = torch.cat(tracked_positions, dim=0)  # (N, 3)

    # Compare with original root trajectory
    original_root = joints_t[:, 0, :]  # (N, 3)

    root_x_error = torch.mean(
        (tracked_positions[:, 0] - original_root[:, 0]) ** 2
    ).item()
    root_y_error = torch.mean(
        (tracked_positions[:, 1] - original_root[:, 1]) ** 2
    ).item()
    root_z_error = torch.mean(
        (tracked_positions[:, 2] - original_root[:, 2]) ** 2
    ).item()

    print(f"  Root X MSE: {root_x_error:.6f}")
    print(f"  Root Y MSE: {root_y_error:.6f}")
    print(f"  Root Z MSE: {root_z_error:.6f}")

    # =========================================================================
    # Test 4: Feature comparison (original vs reconstructed)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 4: Feature comparison (original vs reconstructed)")
    print("-" * 60)

    # Compare feature components
    feature_errors = {
        "root_height": torch.mean(
            (reconstructed_271d[:, 0] - vecs_t[:, 0]) ** 2
        ).item(),
        "root_vel_x": torch.mean((reconstructed_271d[:, 1] - vecs_t[:, 1]) ** 2).item(),
        "root_vel_z": torch.mean((reconstructed_271d[:, 2] - vecs_t[:, 2]) ** 2).item(),
        "ric_positions": torch.mean(
            (reconstructed_271d[:, 3:69] - vecs_t[:, 3:69]) ** 2
        ).item(),
        "rotations_6d": torch.mean(
            (reconstructed_271d[:, 69:201] - vecs_t[:, 69:201]) ** 2
        ).item(),
        "local_velocities": torch.mean(
            (reconstructed_271d[:, 201:267] - vecs_t[:, 201:267]) ** 2
        ).item(),
        "foot_contacts": torch.mean(
            (reconstructed_271d[:, 267:271] - vecs_t[:, 267:271]) ** 2
        ).item(),
    }

    print("  Feature component MSE:")
    for name, error in feature_errors.items():
        print(f"    {name}: {error:.6f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\n  Baseline (features_to_positions):")
    print(f"    MSE: {baseline_mse:.6f}, MAE: {baseline_mae:.6f}")

    print(f"\n  Incremental (flow_output_to_271d + RootPositionTracker):")
    print(f"    MSE: {incremental_mse:.6f}, MAE: {incremental_mae:.6f}")

    # Determine pass/fail
    if incremental_mse < 1e-4:
        print("\n  [PASS] EXCELLENT reconstruction quality")
    elif incremental_mse < 1e-2:
        print("\n  [PASS] GOOD reconstruction quality")
    elif incremental_mse < 1e-1:
        print("\n  [WARN] ACCEPTABLE reconstruction quality")
    else:
        print("\n  [FAIL] POOR reconstruction quality - needs investigation")

    # Compare with baseline
    if incremental_mse <= baseline_mse * 1.1:
        print("  [PASS] Incremental approach matches baseline quality")
    else:
        print(
            f"  [WARN] Incremental approach has {incremental_mse/baseline_mse:.2f}x higher error than baseline"
        )

    return {
        "baseline_mse": baseline_mse,
        "incremental_mse": incremental_mse,
        "root_x_error": root_x_error,
        "root_y_error": root_y_error,
        "root_z_error": root_z_error,
        "feature_errors": feature_errors,
    }


def test_all_samples():
    """Test all sample data files."""
    sample_files = [
        ("sample_data/000000_joint.npy", "sample_data/000000_vec.npy"),
        ("sample_data/000070_joint.npy", "sample_data/000070_vec.npy"),
    ]

    all_results = {}

    for joint_path, vec_path in sample_files:
        if os.path.exists(joint_path) and os.path.exists(vec_path):
            print(f"\n{'#' * 60}")
            print(f"Testing: {joint_path}")
            print(f"{'#' * 60}")
            all_results[joint_path] = verify_incremental_reconstruction(
                joint_path, vec_path
            )
        else:
            print(f"\nSkipping {joint_path} - files not found")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test incremental reconstruction")
    parser.add_argument("--all", action="store_true", help="Test all sample files")
    parser.add_argument(
        "--joint",
        type=str,
        default="sample_data/000000_joint.npy",
        help="Path to joint file",
    )
    parser.add_argument(
        "--vec", type=str, default="sample_data/000000_vec.npy", help="Path to vec file"
    )

    args = parser.parse_args()

    if args.all:
        test_all_samples()
    else:
        verify_incremental_reconstruction(args.joint, args.vec)
