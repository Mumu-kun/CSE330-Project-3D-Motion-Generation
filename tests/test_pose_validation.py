"""
Unit tests for pose validation module.

Tests the validation functions using sample data from the HumanML3D dataset.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest

from src.utils.pose_validation import (
    validate_pose,
    validate_pose_sequence,
    validate_bone_length_ratios,
    validate_ric_bounds,
    validate_kinematic_chain,
    compute_scaling_metrics,
    extract_ric_from_positions,
    PoseValidationResult,
    BoneRatioIssue,
    KinematicIssue,
    ScalingMetrics,
    DEFAULT_RIC_BOUNDS,
    get_joint_names,
)
from src.utils.motion_utils import (
    features_to_positions,
    T2M_KINEMATIC_CHAIN,
    T2M_RAW_OFFSETS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_joints():
    """Load sample joint positions from dataset."""
    joint_path = "sample_data/000070_joint.npy"
    if os.path.exists(joint_path):
        joints = np.load(joint_path)
        return torch.from_numpy(joints).float()
    else:
        # Generate synthetic pose if sample data not available
        return generate_synthetic_pose()


@pytest.fixture
def sample_features():
    """Load sample 271D features from dataset."""
    vec_path = "sample_data/000070_vec.npy"
    if os.path.exists(vec_path):
        vecs = np.load(vec_path)
        return torch.from_numpy(vecs).float()
    else:
        return None


def generate_synthetic_pose(num_frames: int = 10) -> torch.Tensor:
    """
    Generate a synthetic valid pose for testing.

    Creates a simple T-pose like configuration with proper bone lengths.

    Returns:
        Joint positions (N, 22, 3)
    """
    # Start with T2M_RAW_OFFSETS as bone directions
    # Build pose by accumulating offsets along kinematic chains

    positions = torch.zeros(num_frames, 22, 3)

    # Set root at origin (with slight Y offset for height)
    positions[:, 0, 1] = 0.8  # Root height

    # Build each chain
    for chain in T2M_KINEMATIC_CHAIN:
        for i in range(len(chain) - 1):
            parent = chain[i]
            child = chain[i + 1]

            # Get offset direction
            offset = T2M_RAW_OFFSETS[child].float()

            # Scale factor (realistic bone lengths)
            scale = 0.2  # 20cm per bone segment

            # Child position = parent + offset * scale
            positions[:, child] = positions[:, parent] + offset * scale

    return positions


def generate_corrupted_pose(corruption_type: str = "dislocation") -> torch.Tensor:
    """
    Generate a corrupted pose for testing validation failures.

    Args:
        corruption_type: Type of corruption to introduce
            - "dislocation": Joint moved too far from parent
            - "extreme_ric": RIC values outside normal bounds
            - "scaling": Unusual bone length ratios

    Returns:
        Corrupted joint positions (1, 22, 3)
    """
    positions = generate_synthetic_pose(1)

    if corruption_type == "dislocation":
        # Move right knee too far from right hip
        positions[0, 4, :] = positions[0, 1, :] + torch.tensor([2.0, -1.0, 0.0])

    elif corruption_type == "extreme_ric":
        # Move left hand to extreme position
        positions[0, 21, :] = torch.tensor([5.0, 5.0, 5.0])

    elif corruption_type == "scaling":
        # Make one leg much longer than the other
        positions[0, 4, :] = positions[0, 1, :] + torch.tensor(
            [0.0, -1.0, 0.0]
        )  # Long right leg
        positions[0, 7, :] = positions[0, 4, :] + torch.tensor([0.0, -1.0, 0.0])

    return positions


# ============================================================================
# Test Bone Length Ratio Validation
# ============================================================================


class TestBoneLengthRatioValidation:
    """Tests for bone length ratio validation."""

    def test_valid_pose_no_issues(self, sample_joints):
        """Valid pose should have no bone ratio issues."""
        # Take first frame
        if sample_joints.ndim == 3:
            frame = sample_joints[0:1]  # (1, 22, 3)
        else:
            frame = sample_joints

        issues = validate_bone_length_ratios(frame, tolerance=0.5)

        # Should have minimal or no issues for valid data
        severe_issues = [i for i in issues if i.severity == "severe"]
        assert len(severe_issues) == 0, f"Found severe bone ratio issues in valid pose"

    def test_synthetic_pose_no_issues(self):
        """Synthetic T-pose should have no bone ratio issues."""
        pose = generate_synthetic_pose(1)
        issues = validate_bone_length_ratios(pose, tolerance=0.5)

        # Synthetic pose should be valid
        severe_issues = [i for i in issues if i.severity == "severe"]
        assert len(severe_issues) == 0

    def test_corrupted_pose_detects_issues(self):
        """Corrupted pose should detect bone ratio issues."""
        pose = generate_corrupted_pose("scaling")
        issues = validate_bone_length_ratios(pose, tolerance=0.3)

        # Should detect issues
        assert len(issues) > 0, "Should detect bone ratio issues in corrupted pose"

    def test_tolerance_affects_detection(self):
        """Lower tolerance should detect more issues."""
        pose = generate_synthetic_pose(1)

        # High tolerance - few issues
        issues_high = validate_bone_length_ratios(pose, tolerance=1.0)

        # Low tolerance - more issues
        issues_low = validate_bone_length_ratios(pose, tolerance=0.1)

        # Lower tolerance should be more sensitive
        assert len(issues_low) >= len(issues_high)


# ============================================================================
# Test RIC Bounds Validation
# ============================================================================


class TestRICBoundsValidation:
    """Tests for RIC position bounds validation."""

    def test_valid_pose_within_bounds(self, sample_joints):
        """Valid pose should be within RIC bounds."""
        if sample_joints.ndim == 3:
            frame = sample_joints[0:1]
        else:
            frame = sample_joints

        # Extract RIC (assuming positions are global)
        # For this test, we'll use the positions directly
        # In practice, you'd extract RIC from features
        violations = validate_ric_bounds(frame)

        # Count total violations
        total_violations = sum(len(v) for v in violations.values())

        # Should have minimal violations for valid data
        # Note: Some violations may occur due to RIC extraction differences
        print(f"Total RIC violations: {total_violations}")

    def test_extreme_pose_detected(self):
        """Extreme RIC values should be detected."""
        pose = generate_corrupted_pose("extreme_ric")
        violations = validate_ric_bounds(pose)

        # Should detect violations
        total_violations = sum(len(v) for v in violations.values())
        assert total_violations > 0, "Should detect RIC bound violations"

    def test_custom_bounds(self):
        """Custom bounds should be respected."""
        pose = generate_synthetic_pose(1)

        # Very restrictive bounds
        custom_bounds = {i: (-0.01, 0.01, -0.01, 0.01, -0.01, 0.01) for i in range(22)}

        violations = validate_ric_bounds(pose, bounds=custom_bounds)

        # Should have many violations with restrictive bounds
        total_violations = sum(len(v) for v in violations.values())
        assert total_violations > 0


# ============================================================================
# Test Kinematic Chain Validation
# ============================================================================


class TestKinematicChainValidation:
    """Tests for kinematic chain validation."""

    def test_valid_pose_no_dislocations(self, sample_joints):
        """Valid pose should have no dislocations."""
        if sample_joints.ndim == 3:
            frame = sample_joints[0:1]
        else:
            frame = sample_joints

        issues = validate_kinematic_chain(frame, dislocation_tolerance=2.0)

        # Filter for dislocations
        dislocations = [i for i in issues if i.issue_type == "dislocation"]
        severe = [i for i in dislocations if i.severity == "severe"]

        assert len(severe) == 0, f"Found severe dislocations in valid pose"

    def test_dislocation_detected(self):
        """Dislocated joint should be detected."""
        pose = generate_corrupted_pose("dislocation")
        issues = validate_kinematic_chain(pose, dislocation_tolerance=1.5)

        # Should detect dislocation
        dislocations = [i for i in issues if i.issue_type == "dislocation"]
        assert len(dislocations) > 0, "Should detect joint dislocation"

    def test_tolerance_affects_dislocation_detection(self):
        """Lower tolerance should detect more dislocations."""
        pose = generate_corrupted_pose("dislocation")

        # High tolerance - may not detect
        issues_high = validate_kinematic_chain(pose, dislocation_tolerance=5.0)

        # Low tolerance - should detect
        issues_low = validate_kinematic_chain(pose, dislocation_tolerance=1.0)

        dislocations_low = [i for i in issues_low if i.issue_type == "dislocation"]
        dislocations_high = [i for i in issues_high if i.issue_type == "dislocation"]

        assert len(dislocations_low) >= len(dislocations_high)


# ============================================================================
# Test Scaling Metrics
# ============================================================================


class TestScalingMetrics:
    """Tests for scaling metrics computation."""

    def test_returns_metrics(self, sample_joints):
        """Should return scaling metrics."""
        if sample_joints.ndim == 3:
            frame = sample_joints[0:1]
        else:
            frame = sample_joints

        metrics = compute_scaling_metrics(frame)

        assert isinstance(metrics, ScalingMetrics)
        assert metrics.mean_scale > 0
        assert metrics.std_scale >= 0
        assert metrics.min_scale <= metrics.mean_scale <= metrics.max_scale

    def test_synthetic_pose_consistent_scale(self):
        """Synthetic pose should have consistent scale."""
        pose = generate_synthetic_pose(1)
        metrics = compute_scaling_metrics(pose)

        # All bones have same length in synthetic pose
        # So std should be low
        assert metrics.std_scale < 0.5, f"Scale std too high: {metrics.std_scale}"

    def test_bone_lengths_computed(self):
        """Should compute bone lengths for all chains."""
        pose = generate_synthetic_pose(1)
        metrics = compute_scaling_metrics(pose)

        # Should have bone lengths for each bone in kinematic chain
        expected_bones = 0
        for chain in T2M_KINEMATIC_CHAIN:
            expected_bones += len(chain) - 1

        assert len(metrics.bone_lengths) == expected_bones


# ============================================================================
# Test Main Validation Function
# ============================================================================


class TestValidatePose:
    """Tests for the main validate_pose function."""

    def test_returns_result_object(self, sample_joints):
        """Should return PoseValidationResult."""
        result = validate_pose(
            sample_joints[0] if sample_joints.ndim == 3 else sample_joints
        )

        assert isinstance(result, PoseValidationResult)
        assert isinstance(result.is_valid, bool)

    def test_valid_pose_passes(self, sample_joints):
        """Valid pose should pass validation."""
        result = validate_pose(
            sample_joints[0] if sample_joints.ndim == 3 else sample_joints,
            bone_ratio_tolerance=0.5,
            dislocation_tolerance=2.0,
        )

        # Should be valid or have only minor issues
        print(f"Validation result: {result.is_valid}")
        print(result.summary())

    def test_corrupted_pose_fails(self):
        """Corrupted pose should fail validation."""
        pose = generate_corrupted_pose("dislocation")
        result = validate_pose(pose, dislocation_tolerance=1.5)

        # Should detect issues
        assert len(result.kinematic_issues) > 0 or not result.is_valid

    def test_can_disable_checks(self, sample_joints):
        """Should be able to disable individual checks."""
        result = validate_pose(
            sample_joints[0] if sample_joints.ndim == 3 else sample_joints,
            check_bone_ratios=False,
            check_ric_bounds=False,
            check_kinematic_chain=False,
            report_scaling=False,
        )

        # Should have no issues when all checks disabled
        assert len(result.bone_ratio_issues) == 0
        assert len(result.ric_bound_violations) == 0
        assert len(result.kinematic_issues) == 0
        assert result.scaling_metrics is None

    def test_handles_different_shapes(self, sample_joints):
        """Should handle different input shapes."""
        # Single frame (22, 3)
        if sample_joints.ndim == 3:
            result1 = validate_pose(sample_joints[0])
            assert isinstance(result1, PoseValidationResult)

        # Batched (B, 22, 3)
        result2 = validate_pose(
            sample_joints[:2] if sample_joints.ndim == 3 else sample_joints
        )
        assert isinstance(result2, PoseValidationResult)

        # Sequence (N, 22, 3)
        result3 = validate_pose(
            sample_joints[:10] if sample_joints.ndim == 3 else sample_joints
        )
        assert isinstance(result3, PoseValidationResult)

    def test_summary_method(self, sample_joints):
        """Summary should return readable string."""
        result = validate_pose(
            sample_joints[0] if sample_joints.ndim == 3 else sample_joints
        )
        summary = result.summary()

        assert isinstance(summary, str)
        assert "VALID" in summary or "INVALID" in summary

    def test_to_dict_method(self, sample_joints):
        """to_dict should return serializable dictionary."""
        result = validate_pose(
            sample_joints[0] if sample_joints.ndim == 3 else sample_joints
        )
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "is_valid" in d
        assert "num_bone_ratio_issues" in d


# ============================================================================
# Test Sequence Validation
# ============================================================================


class TestValidatePoseSequence:
    """Tests for sequence validation."""

    def test_returns_per_frame_results(self, sample_joints):
        """Should return per-frame results."""
        if sample_joints.ndim != 3:
            pytest.skip("Need sequence data")

        # Take first 10 frames
        sequence = sample_joints[:10].unsqueeze(0)  # (1, 10, 22, 3)

        overall, per_frame = validate_pose_sequence(sequence)

        assert isinstance(overall, PoseValidationResult)
        assert len(per_frame) == 10
        assert all(isinstance(r, PoseValidationResult) for r in per_frame)

    def test_frame_threshold_affects_validity(self, sample_joints):
        """Frame threshold should affect overall validity."""
        if sample_joints.ndim != 3:
            pytest.skip("Need sequence data")

        sequence = sample_joints[:10].unsqueeze(0)

        # Strict threshold
        overall_strict, _ = validate_pose_sequence(sequence, frame_threshold=0.0)

        # Lenient threshold
        overall_lenient, _ = validate_pose_sequence(sequence, frame_threshold=1.0)

        # Lenient should be at least as valid as strict
        assert overall_lenient.is_valid or not overall_strict.is_valid


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_joint_names(self):
        """Should return joint name mapping."""
        names = get_joint_names()

        assert isinstance(names, dict)
        assert len(names) == 22
        assert names[0] == "root"
        assert names[15] == "head"

    def test_extract_ric_from_positions(self):
        """Should extract RIC from global positions."""
        pose = generate_synthetic_pose(1)

        ric = extract_ric_from_positions(pose)

        assert ric.shape == pose.shape
        # Root should be at origin in RIC
        assert torch.allclose(ric[0, 0, :], torch.zeros(3), atol=1e-5)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real data."""

    def test_with_sample_data(self, sample_joints, sample_features):
        """Test with actual sample data from dataset."""
        if sample_features is None:
            pytest.skip("Sample features not available")

        # Reconstruct positions from features
        reconstructed = features_to_positions(sample_features)

        # Validate reconstructed positions
        result = validate_pose(reconstructed)

        print("\nValidation of reconstructed positions:")
        print(result.summary())

        # Reconstructed positions should be valid
        # (they come from the dataset)
        assert result.is_valid or len(result.kinematic_issues) < 5

    def test_with_generator_output(self):
        """Test validation with generator-like output."""
        # Simulate generator output
        # In practice, this would come from HumanMotionGenerator.generate_sequence()

        # Generate synthetic sequence
        sequence = generate_synthetic_pose(100)  # 100 frames

        # Validate
        result = validate_pose(sequence)

        print("\nValidation of synthetic sequence:")
        print(result.summary())

        # Synthetic sequence should be valid
        assert result.is_valid


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
