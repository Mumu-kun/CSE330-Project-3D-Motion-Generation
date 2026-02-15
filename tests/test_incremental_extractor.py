"""
Test for new 271D feature extraction and IncrementalFeatureExtractor.

Tests:
1. preprocess_sequence() - full sequence extraction
2. features_to_positions() - reconstruction
3. IncrementalFeatureExtractor - frame-by-frame extraction
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
from utils.motion_utils import (
    IncrementalFeatureExtractor,
    get_dataset_config,
    preprocess_sequence,
    features_to_positions,
    FEATURE_SLICES,
)


def test_preprocess_sequence():
    """Test preprocess_sequence() for dataset preprocessing."""
    print("\n" + "=" * 60)
    print("Testing preprocess_sequence()")
    print("=" * 60)

    # Load sample data
    data_dir = Path(__file__).parent.parent / "sample_data"
    joint_path = data_dir / "000000_joint.npy"

    if not joint_path.exists():
        print(f"[ERROR] Sample data not found: {joint_path}")
        return

    import numpy as np

    joints_np = np.load(joint_path)  # (Nframe, 22, 3)
    joints = torch.from_numpy(joints_np).float()
    print(f"\nLoaded joints shape: {joints.shape}")

    # Extract features using new API
    features = preprocess_sequence(
        positions=joints,
        dataset_type="t2m",
    )

    print(f"Extracted features shape: {features.shape}")
    print(f"Expected: ({joints.shape[0]}, 271)")

    # Verify feature slices
    print("\nFeature slice verification:")
    for name, sl in FEATURE_SLICES.items():
        feat = features[:, sl]
        print(
            f"  {name}: shape {feat.shape}, mean={feat.mean():.6f}, std={feat.std():.6f}"
        )

    # Test reconstruction
    print("\n--- Testing Reconstruction ---")

    # Direct RIC transform
    positions_direct = features_to_positions(features, dataset_type="t2m")
    print(f"Reconstructed positions: {positions_direct.shape}")

    # Compare with original
    diff = torch.abs(positions_direct - joints)
    mean_diff = torch.mean(diff).item()
    max_diff = torch.max(diff).item()
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Max absolute diff: {max_diff:.6f}")

    if mean_diff < 1e-5:
        print("  [PASS] Reconstruction matches original!")
    else:
        print("  [WARN] Reconstruction has differences")

    print("\n" + "=" * 60)


def test_incremental_extractor():
    """Test IncrementalFeatureExtractor for frame-by-frame extraction."""
    print("\n" + "=" * 60)
    print("Testing IncrementalFeatureExtractor")
    print("=" * 60)

    # Load sample data
    data_dir = Path(__file__).parent.parent / "sample_data"
    joint_path = data_dir / "000000_joint.npy"

    if not joint_path.exists():
        print(f"[ERROR] Sample data not found: {joint_path}")
        return

    import numpy as np

    joints_np = np.load(joint_path)  # (Nframe, 22, 3)
    joints = torch.from_numpy(joints_np).float()
    print(f"\nLoaded joints shape: {joints.shape}")

    # Create incremental extractor with new API (dataset_type instead of individual params)
    extractor = IncrementalFeatureExtractor(
        dataset_type="t2m",
        feet_thre=0.002,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Process frames incrementally
    n_frames = joints.shape[0]
    incremental_features = []

    for i in range(n_frames):
        frame = joints[i : i + 1]  # (1, 22, 3)
        features = extractor.process_frame(frame)
        incremental_features.append(features)

    incremental_features = torch.cat(incremental_features, dim=0)

    print(f"Incremental features shape: {incremental_features.shape}")

    # First frame should be zeros
    print(f"\nFirst frame (should be zeros):")
    print(f"  Mean: {torch.mean(torch.abs(incremental_features[0])).item():.6f}")
    print(f"  Max: {torch.max(torch.abs(incremental_features[0])).item():.6f}")

    # Compare with full sequence extraction
    full_features = preprocess_sequence(joints, dataset_type="t2m")

    # Skip first frame for comparison
    diff = torch.abs(incremental_features[1:] - full_features[1:])
    mean_diff = torch.mean(diff).item()
    max_diff = torch.max(diff).item()

    print(f"\nComparison (incremental vs full, skipping first frame):")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Max absolute diff: {max_diff:.6f}")

    if mean_diff < 1e-5:
        print("  [PASS] Incremental matches full extraction!")
    else:
        print("  [WARN] Incremental has differences from full extraction")

        # Per-feature breakdown
        print("\n  Per-feature breakdown:")
        for name, sl in FEATURE_SLICES.items():
            feat_diff = torch.mean(
                torch.abs(incremental_features[1:, sl] - full_features[1:, sl])
            ).item()
            print(f"    {name}: mean diff = {feat_diff:.6f}")

    print("\n" + "=" * 60)


def test_round_trip():
    """Test round-trip: positions -> features -> positions."""
    print("\n" + "=" * 60)
    print("Testing Round-Trip Conversion")
    print("=" * 60)

    # Load sample data
    data_dir = Path(__file__).parent.parent / "sample_data"
    joint_path = data_dir / "000000_joint.npy"

    if not joint_path.exists():
        print(f"[ERROR] Sample data not found: {joint_path}")
        return

    import numpy as np

    joints_np = np.load(joint_path)
    joints = torch.from_numpy(joints_np).float()
    print(f"\nOriginal joints shape: {joints.shape}")

    # Extract features
    features = preprocess_sequence(joints, dataset_type="t2m")
    print(f"Features shape: {features.shape}")

    # Reconstruct positions
    reconstructed = features_to_positions(features, dataset_type="t2m")

    print(f"Reconstructed shape: {reconstructed.shape}")

    # Compare
    diff = torch.abs(reconstructed - joints)
    print(f"\nRound-trip error:")
    print(f"  Mean: {torch.mean(diff).item():.6f}")
    print(f"  Max: {torch.max(diff).item():.6f}")
    print(f"  Per-joint mean error: {torch.mean(diff, dim=(0, 2))}")

    if torch.mean(diff).item() < 1e-5:
        print("\n  [PASS] Round-trip successful!")
    else:
        print("\n  [WARN] Round-trip has errors")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_preprocess_sequence()
    test_incremental_extractor()
    test_round_trip()
    print("\nAll tests complete!")
