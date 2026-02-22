"""
Concrete test script for pose validation analysis.

This script runs validation on sample data and provides detailed output
for analysis of the validation results.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from utils.pose_validation import (
    validate_pose,
    validate_pose_sequence,
    validate_bone_length_ratios,
    validate_ric_bounds,
    validate_kinematic_chain,
    compute_scaling_metrics,
    get_joint_names,
)
from utils.motion_utils import features_to_positions, T2M_KINEMATIC_CHAIN


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def analyze_sample_data():
    """Analyze sample data with detailed output."""

    # Load sample data
    print_separator("Loading Sample Data")

    joint_path = "sample_data/000070_joint.npy"
    vec_path = "sample_data/000070_vec.npy"

    joints = np.load(joint_path)
    vecs = np.load(vec_path)

    print(f"Joints shape: {joints.shape}")
    print(f"Features shape: {vecs.shape}")

    joints_t = torch.from_numpy(joints).float()
    vecs_t = torch.from_numpy(vecs).float()

    # Extract RIC positions from features
    ric = vecs_t[:, 3:69].reshape(-1, 22, 3)

    print(f"RIC positions shape: {ric.shape}")

    # =========================================================================
    # Test 1: Validate single frame from RIC
    # =========================================================================
    print_separator("Test 1: Single Frame RIC Validation")

    frame_idx = 0
    single_frame_ric = ric[frame_idx : frame_idx + 1]  # (1, 22, 3)

    result = validate_pose(
        single_frame_ric,
        check_bone_ratios=True,
        check_ric_bounds=True,
        check_kinematic_chain=False,  # RIC is in local coords, not global
        report_scaling=True,
    )

    print(result.summary())

    # =========================================================================
    # Test 2: Validate sequence of RIC frames
    # =========================================================================
    print_separator("Test 2: RIC Sequence Validation (10 frames)")

    sequence_ric = ric[:10]  # (10, 22, 3)

    overall, per_frame = validate_pose_sequence(
        sequence_ric,
        check_kinematic_chain=False,
    )

    print(f"Overall valid: {overall.is_valid}")
    print(f"Valid frames: {sum(1 for r in per_frame if r.is_valid)}/{len(per_frame)}")

    if overall.scaling_metrics:
        print(f"\nScaling Metrics:")
        print(f"  Mean scale: {overall.scaling_metrics.mean_scale:.4f}")
        print(f"  Std scale: {overall.scaling_metrics.std_scale:.4f}")
        print(
            f"  Range: [{overall.scaling_metrics.min_scale:.4f}, {overall.scaling_metrics.max_scale:.4f}]"
        )

    # =========================================================================
    # Test 3: Validate global joint positions
    # =========================================================================
    print_separator("Test 3: Global Joint Positions Validation")

    # Reconstruct global positions from features
    global_positions = features_to_positions(vecs_t)
    print(f"Global positions shape: {global_positions.shape}")

    # Validate first frame
    first_frame = global_positions[0:1]  # (1, 22, 3)

    result_global = validate_pose(
        first_frame,
        check_bone_ratios=True,
        check_ric_bounds=False,  # Global positions don't use RIC bounds
        check_kinematic_chain=True,
        report_scaling=True,
    )

    print(result_global.summary())

    # =========================================================================
    # Test 4: Detailed bone length analysis
    # =========================================================================
    print_separator("Test 4: Bone Length Analysis")

    metrics = compute_scaling_metrics(global_positions[:10])

    print("\nBone Lengths (mean across 10 frames):")
    joint_names = get_joint_names()

    for bone_name, length in sorted(metrics.bone_lengths.items()):
        scale = metrics.scale_factors.get(bone_name, 1.0)
        print(f"  {bone_name}: length={length:.4f}, scale={scale:.4f}")

    # =========================================================================
    # Test 5: Kinematic chain analysis
    # =========================================================================
    print_separator("Test 5: Kinematic Chain Analysis")

    print("\nKinematic Chains:")
    for i, chain in enumerate(T2M_KINEMATIC_CHAIN):
        chain_name = ["left_leg", "right_leg", "spine", "right_arm", "left_arm"][i]
        joint_names_chain = [joint_names[j] for j in chain]
        print(f"  {chain_name}: {' -> '.join(joint_names_chain)}")

    issues = validate_kinematic_chain(global_positions[:10])

    if issues:
        print(f"\nKinematic Issues Found: {len(issues)}")
        for issue in issues[:10]:  # Show first 10
            print(
                f"  [{issue.severity}] {issue.issue_type}: joints {issue.parent_joint}->{issue.child_joint}"
            )
            if "actual_distance" in issue.details:
                print(f"    Actual distance: {issue.details['actual_distance']:.4f}")
    else:
        print("\nNo kinematic issues found.")

    # =========================================================================
    # Test 6: RIC bounds analysis
    # =========================================================================
    print_separator("Test 6: RIC Bounds Analysis")

    violations = validate_ric_bounds(ric[:10])

    if violations:
        print(f"\nRIC Bound Violations: {len(violations)} joints affected")
        for joint_key, frames in list(violations.items())[:10]:
            print(f"  {joint_key}: {len(frames)} violations")
    else:
        print("\nNo RIC bound violations found.")

    # =========================================================================
    # Summary
    # =========================================================================
    print_separator("Summary")

    print(
        f"""
Validation Results Summary:
---------------------------
Sample: 000070 (HumanML3D)
Frames analyzed: {min(10, joints.shape[0])}

RIC Validation:
  - Bone ratio issues: {len(overall.bone_ratio_issues)}
  - RIC bound violations: {sum(len(v) for v in violations.values())}

Global Position Validation:
  - Kinematic issues: {len(issues)}
  - Mean scale factor: {metrics.mean_scale:.4f}
  - Scale std deviation: {metrics.std_scale:.4f}

Interpretation:
  - The bone ratio issues indicate that bone lengths in the data
    don't match the expected 1:1 ratios from T2M_RAW_OFFSETS
  - This is expected since T2M_RAW_OFFSETS use unit vectors
  - The scale factor (~{metrics.mean_scale:.2f}) shows the actual bone lengths
    relative to the unit vectors
  - For generated motion, check if scale factors are consistent
    and bone ratios are similar to training data
"""
    )


if __name__ == "__main__":
    analyze_sample_data()
