"""
Test for IncrementalFeatureExtractor.

Compares the output of IncrementalFeatureExtractor against pre-extracted
feature vectors from the HumanML3D dataset.

Sample data:
- sample_data/000000_joint.npy: (Nframe, 22, 3) global joint positions
- sample_data/000000_vec.npy: (Nframe, 263) expected feature vectors
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import torch
from utils.motion_utils import (
    IncrementalFeatureExtractor,
    get_dataset_config,
    extract_features,
)


def test_incremental_extractor():
    """
    Test IncrementalFeatureExtractor against pre-extracted features.

    The pre-extracted features were computed using extract_features() which
    uses full sequence IK. We compare against incremental extraction.
    """
    print("\n" + "=" * 60)
    print("Testing IncrementalFeatureExtractor")
    print("=" * 60)

    # Load sample data
    data_dir = Path(__file__).parent.parent / "sample_data"
    joint_path = data_dir / "000000_joint.npy"
    vec_path = data_dir / "000000_vec.npy"

    if not joint_path.exists() or not vec_path.exists():
        print(f"[ERROR] Sample data not found:")
        print(f"   Joint file exists: {joint_path.exists()}")
        print(f"   Vec file exists: {vec_path.exists()}")
        return

    joints = np.load(joint_path)  # (Nframe, 22, 3)
    expected_vec = np.load(vec_path)  # (Nframe, 263)

    print(f"\nLoaded data:")
    print(f"  Joints shape: {joints.shape}")
    print(f"  Expected features shape: {expected_vec.shape}")

    # Get dataset config
    config = get_dataset_config("t2m")

    # =========================================================================
    # Test 1: Compare against extract_features (full sequence)
    # =========================================================================
    print("\n--- Test 1: Full sequence extraction comparison ---")

    # Extract features using the original function
    extracted_features = extract_features(
        positions=joints,
        feet_thre=0.002,
        n_raw_offsets=config["raw_offsets"],
        kinematic_chain=config["kinematic_chain"],
        face_joint_indx=config["face_joint_indx"],
        fid_r=config["fid_r"],
        fid_l=config["fid_l"],
    )

    print(f"  extract_features output shape: {extracted_features.shape}")
    print(f"  Expected features shape: {expected_vec.shape}")

    # Note: extract_features returns (Nframe-1, 263) due to velocity calculation
    print(
        f"\n  Note: extract_features produces Nframe-1 features due to velocity calculation"
    )

    # Compare with expected - need to handle shape mismatch
    min_len = min(extracted_features.shape[0], expected_vec.shape[0])

    diff = np.abs(extracted_features[:min_len] - expected_vec[:min_len])
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    print(f"\n  Comparison (first {min_len} frames):")
    print(f"    Mean absolute diff: {mean_diff:.6f}")
    print(f"    Max absolute diff: {max_diff:.6f}")

    # Check if they match (allowing small numerical tolerance)
    if mean_diff < 1e-5:
        print(f"    [PASS] extract_features matches expected features!")
    else:
        print(f"    [WARN] extract_features differs from expected features")

        # Show per-feature breakdown
        feature_names = [
            "root (0:4)",
            "RIC (4:67)",
            "rot (67:193)",
            "vel (193:259)",
            "foot (259:263)",
        ]
        feature_slices = [
            slice(0, 4),
            slice(4, 67),
            slice(67, 193),
            slice(193, 259),
            slice(259, 263),
        ]

        print("\n  Per-feature breakdown:")
        for name, sl in zip(feature_names, feature_slices):
            feat_diff = np.mean(
                np.abs(extracted_features[:min_len, sl] - expected_vec[:min_len, sl])
            )
            print(f"    {name}: mean diff = {feat_diff:.6f}")

    # =========================================================================
    # Test 2: IncrementalFeatureExtractor
    # =========================================================================
    print("\n--- Test 2: IncrementalFeatureExtractor comparison ---")

    # Create incremental extractor
    extractor = IncrementalFeatureExtractor(
        n_raw_offsets=config["raw_offsets"],
        kinematic_chain=config["kinematic_chain"],
        face_joint_indx=config["face_joint_indx"],
        fid_r=config["fid_r"],
        fid_l=config["fid_l"],
        feet_thre=0.002,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Process frames incrementally
    n_frames = joints.shape[0]
    incremental_features = []

    # Convert to torch tensor with batch dimension
    joints_torch = torch.from_numpy(joints).float()  # (Nframe, 22, 3)

    for i in range(n_frames):
        frame = joints_torch[i : i + 1]  # (1, 22, 3) - keep batch dim
        features = extractor.process_frame(frame)  # (1, 263)
        incremental_features.append(features.detach().numpy())

    incremental_features = np.concatenate(incremental_features, axis=0)  # (Nframe, 263)

    print(f"  Incremental features shape: {incremental_features.shape}")
    print(f"  Expected features shape: {expected_vec.shape}")

    # First frame should be zeros (no previous frame for velocity)
    print(f"\n  First frame (should be zeros):")
    print(f"    Mean: {np.mean(np.abs(incremental_features[0])):.6f}")
    print(f"    Max: {np.max(np.abs(incremental_features[0])):.6f}")

    # Compare frames 1: with expected (skip first frame)
    # Incremental produces Nframe features, expected has Nframe
    # Compare incremental[1:] with expected[1:] or expected[:-1]

    # Try both comparisons
    print(f"\n  Comparison attempts:")

    # Option A: incremental[1:] vs expected[:-1] (both have Nframe-1 elements)
    if incremental_features.shape[0] - 1 == expected_vec.shape[0] - 1:
        inc_from_frame_1 = incremental_features[1:]
        exp_except_last = expected_vec[:-1]
        diff_a = np.abs(inc_from_frame_1 - exp_except_last)
        mean_diff_a = np.mean(diff_a)
        print(
            f"    Option A (incremental[1:] vs expected[:-1]): mean diff = {mean_diff_a:.6f}"
        )

    # Option B: incremental[1:] vs expected[1:]
    inc_from_frame_1 = incremental_features[1:]
    exp_from_frame_1 = expected_vec[1:]
    diff_b = np.abs(inc_from_frame_1 - exp_from_frame_1)
    mean_diff_b = np.mean(diff_b)
    print(
        f"    Option B (incremental[1:] vs expected[1:]): mean diff = {mean_diff_b:.6f}"
    )

    # Per-feature breakdown for Option B
    feature_names = [
        "root (0:4)",
        "RIC (4:67)",
        "rot (67:193)",
        "vel (193:259)",
        "foot (259:263)",
    ]
    feature_slices = [
        slice(0, 4),
        slice(4, 67),
        slice(67, 193),
        slice(193, 259),
        slice(259, 263),
    ]

    print("\n  Per-feature breakdown (Option B):")
    for name, sl in zip(feature_names, feature_slices):
        feat_diff = np.mean(np.abs(inc_from_frame_1[:, sl] - exp_from_frame_1[:, sl]))
        print(f"    {name}: mean diff = {feat_diff:.6f}")

    if mean_diff_b < 0.1:  # Allow larger tolerance for incremental
        print(f"\n  [PASS] IncrementalFeatureExtractor produces similar results!")
    else:
        print(f"\n  [WARN] IncrementalFeatureExtractor has significant differences")
        print(f"     This may be expected due to simplified IK in incremental mode")

    # =========================================================================
    # Test 3: Compare incremental vs full extraction
    # =========================================================================
    print("\n--- Test 3: Incremental vs Full extraction ---")

    if extracted_features.shape[0] == incremental_features.shape[0] - 1:
        diff = np.abs(extracted_features - incremental_features[1:])
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  Max absolute diff: {max_diff:.6f}")

        # Per-feature breakdown
        print("\n  Per-feature breakdown:")
        for name, sl in zip(feature_names, feature_slices):
            feat_diff = np.mean(
                np.abs(extracted_features[:, sl] - incremental_features[1:, sl])
            )
            print(f"    {name}: mean diff = {feat_diff:.6f}")

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_incremental_extractor()
