"""
Pose Validation Module for Human Motion Generation.

This module provides comprehensive validation for generated motion poses,
checking if they conform to the expected skeleton structure using RIC positions.

Validation Components:
1. Bone Length Ratio Validation - Compares ratios within kinematic chains
2. RIC Position Bounds Checking - Validates positions are within plausible ranges
3. Kinematic Chain Validity - Detects dislocations and chain discontinuities
4. Scaling Deviation Reporting - Reports how much poses deviate from expected proportions

Usage:
    from utils.pose_validation import validate_pose, validate_pose_sequence

    # Validate single frame
    result = validate_pose(positions)  # (B, 22, 3)

    # Validate sequence
    results = validate_pose_sequence(joints)  # (B, T, 22, 3)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

# Import from same package - use relative imports
from .motion_utils import (
    T2M_RAW_OFFSETS,
    T2M_KINEMATIC_CHAIN,
    get_dataset_config,
)


# ============================================================================
# Data Classes for Validation Results
# ============================================================================


@dataclass
class BoneRatioIssue:
    """Represents a bone length ratio issue."""

    chain_name: str
    bone_pair: str
    actual_ratio: float
    expected_ratio: float
    deviation: float
    severity: str  # "minor", "moderate", "severe"


@dataclass
class KinematicIssue:
    """Represents a kinematic chain issue."""

    issue_type: str  # "dislocation", "unusual_ratio", "chain_break"
    chain_name: str
    parent_joint: int
    child_joint: int
    details: Dict[str, Any]
    severity: str  # "minor", "moderate", "severe"


@dataclass
class ScalingMetrics:
    """Scaling deviation metrics for a pose."""

    mean_scale: float
    std_scale: float
    min_scale: float
    max_scale: float
    bone_lengths: Dict[str, float]
    scale_factors: Dict[str, float]


@dataclass
class PoseValidationResult:
    """Comprehensive result of pose validation."""

    is_valid: bool
    bone_ratio_issues: List[BoneRatioIssue] = field(default_factory=list)
    ric_bound_violations: Dict[str, List[int]] = field(default_factory=dict)
    kinematic_issues: List[KinematicIssue] = field(default_factory=list)
    scaling_metrics: Optional[ScalingMetrics] = None

    def summary(self) -> str:
        """Return human-readable summary of validation results."""
        lines = []
        lines.append(f"Pose Validation: {'VALID' if self.is_valid else 'INVALID'}")

        if self.bone_ratio_issues:
            lines.append(f"\nBone Ratio Issues ({len(self.bone_ratio_issues)}):")
            for issue in self.bone_ratio_issues[:5]:  # Show first 5
                lines.append(
                    f"  - {issue.chain_name}: {issue.bone_pair} "
                    f"ratio={issue.actual_ratio:.2f} (expected ~{issue.expected_ratio:.2f}) "
                    f"[{issue.severity}]"
                )
            if len(self.bone_ratio_issues) > 5:
                lines.append(f"  ... and {len(self.bone_ratio_issues) - 5} more")

        if self.ric_bound_violations:
            lines.append(f"\nRIC Bound Violations:")
            for joint_key, frame_indices in self.ric_bound_violations.items():
                lines.append(f"  - {joint_key}: {len(frame_indices)} violations")

        if self.kinematic_issues:
            lines.append(f"\nKinematic Issues ({len(self.kinematic_issues)}):")
            for issue in self.kinematic_issues[:5]:
                lines.append(
                    f"  - {issue.issue_type}: joints {issue.parent_joint}->{issue.child_joint} "
                    f"[{issue.severity}]"
                )
            if len(self.kinematic_issues) > 5:
                lines.append(f"  ... and {len(self.kinematic_issues) - 5} more")

        if self.scaling_metrics:
            lines.append(f"\nScaling Metrics:")
            lines.append(f"  - Mean scale: {self.scaling_metrics.mean_scale:.3f}")
            lines.append(f"  - Std scale: {self.scaling_metrics.std_scale:.3f}")
            lines.append(
                f"  - Range: [{self.scaling_metrics.min_scale:.3f}, {self.scaling_metrics.max_scale:.3f}]"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "is_valid": self.is_valid,
            "num_bone_ratio_issues": len(self.bone_ratio_issues),
            "num_ric_violations": sum(
                len(v) for v in self.ric_bound_violations.values()
            ),
            "num_kinematic_issues": len(self.kinematic_issues),
            "scaling_metrics": {
                "mean_scale": (
                    self.scaling_metrics.mean_scale if self.scaling_metrics else None
                ),
                "std_scale": (
                    self.scaling_metrics.std_scale if self.scaling_metrics else None
                ),
            },
        }


# ============================================================================
# Default Configuration
# ============================================================================

# Chain names for readable output
CHAIN_NAMES = {
    0: "left_leg",
    1: "right_leg",
    2: "spine",
    3: "right_arm",
    4: "left_arm",
}

# Default RIC position bounds per joint (based on HumanML3D dataset statistics)
# Format: (min_x, max_x, min_y, max_y, min_z, max_z)
# These are relative to root joint in root-local coordinates
# Computed from sample_data/000070_vec.npy with 20% margin
DEFAULT_RIC_BOUNDS = {
    0: (-0.01, 0.01, -0.01, 0.01, -0.01, 0.01),  # Root (should be near origin)
    1: (0.02, 0.08, -0.10, -0.06, -0.07, 0.02),  # Right Hip
    2: (-0.08, -0.04, -0.11, -0.07, -0.04, 0.04),  # Left Hip
    3: (-0.03, 0.03, 0.11, 0.14, -0.07, -0.02),  # Spine
    4: (0.01, 0.24, -0.51, -0.40, -0.25, 0.27),  # Right Knee
    5: (-0.23, -0.02, -0.51, -0.40, -0.22, 0.28),  # Left Knee
    6: (-0.03, 0.05, 0.24, 0.28, -0.04, 0.05),  # Spine1
    7: (-0.10, 0.30, -0.98, -0.66, -0.58, 0.46),  # Right Ankle
    8: (-0.32, 0.06, -0.97, -0.67, -0.58, 0.51),  # Left Ankle
    9: (-0.04, 0.06, 0.30, 0.33, -0.03, 0.07),  # Spine2 (chest)
    10: (-0.05, 0.40, -1.04, -0.80, -0.59, 0.62),  # Right Foot
    11: (-0.38, 0.03, -1.02, -0.79, -0.56, 0.69),  # Left Foot
    12: (-0.06, 0.10, 0.51, 0.55, -0.05, 0.13),  # Neck
    13: (0.03, 0.16, 0.40, 0.45, -0.05, 0.10),  # Right Shoulder
    14: (-0.14, -0.01, 0.40, 0.46, -0.05, 0.10),  # Left Shoulder
    15: (-0.08, 0.16, 0.56, 0.65, -0.01, 0.22),  # Head
    16: (0.16, 0.29, 0.37, 0.46, -0.08, 0.09),  # Right Elbow
    17: (-0.26, -0.12, 0.40, 0.50, -0.07, 0.12),  # Left Elbow
    18: (0.20, 0.30, 0.11, 0.23, -0.14, 0.09),  # Right Wrist
    19: (-0.32, -0.21, 0.15, 0.27, -0.14, 0.00),  # Left Wrist
    20: (0.10, 0.40, -0.15, 0.05, -0.24, 0.37),  # Right Hand
    21: (-0.38, -0.23, -0.11, 0.03, -0.12, 0.17),  # Left Hand
}


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_bone_lengths(
    positions: torch.Tensor,
    kinematic_chain: List[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute bone lengths for each parent-child pair in the kinematic chain.

    Args:
        positions: Joint positions (..., 22, 3)
        kinematic_chain: Skeleton kinematic chain (default: T2M_KINEMATIC_CHAIN)

    Returns:
        Dictionary mapping "parent_child" to bone length tensors
    """
    if kinematic_chain is None:
        kinematic_chain = T2M_KINEMATIC_CHAIN

    bone_lengths = {}

    for chain_idx, chain in enumerate(kinematic_chain):
        chain_name = CHAIN_NAMES.get(chain_idx, f"chain_{chain_idx}")

        for i in range(len(chain) - 1):
            parent = chain[i]
            child = chain[i + 1]
            bone_name = f"{chain_name}_{parent}_{child}"

            # Compute distance between parent and child
            parent_pos = positions[..., parent, :]
            child_pos = positions[..., child, :]
            length = torch.norm(child_pos - parent_pos, dim=-1)

            bone_lengths[bone_name] = length

    return bone_lengths


def _compute_expected_bone_lengths() -> Dict[str, float]:
    """
    Compute expected bone lengths from T2M_RAW_OFFSETS.

    Returns:
        Dictionary mapping "parent_child" to expected length
    """
    expected_lengths = {}

    for chain_idx, chain in enumerate(T2M_KINEMATIC_CHAIN):
        chain_name = CHAIN_NAMES.get(chain_idx, f"chain_{chain_idx}")

        for i in range(len(chain) - 1):
            parent = chain[i]
            child = chain[i + 1]
            bone_name = f"{chain_name}_{parent}_{child}"

            # Expected length from raw offsets
            offset = T2M_RAW_OFFSETS[child]
            expected_length = torch.norm(offset).item()
            expected_lengths[bone_name] = (
                expected_length if expected_length > 0 else 1.0
            )

    return expected_lengths


def _get_severity(
    deviation: float, thresholds: Tuple[float, float] = (0.3, 0.5)
) -> str:
    """
    Determine severity based on deviation from expected.

    Args:
        deviation: Absolute deviation from expected
        thresholds: (moderate_threshold, severe_threshold)

    Returns:
        Severity string: "minor", "moderate", or "severe"
    """
    if deviation < thresholds[0]:
        return "minor"
    elif deviation < thresholds[1]:
        return "moderate"
    else:
        return "severe"


# ============================================================================
# Validation Functions
# ============================================================================


def validate_bone_length_ratios(
    positions: torch.Tensor,
    tolerance: float = 0.5,
) -> List[BoneRatioIssue]:
    """
    Validate bone length ratios within each kinematic chain.

    This checks if the ratios between consecutive bones in a chain are
    consistent, which handles scaling issues since we compare relative
    proportions within the same pose.

    Args:
        positions: Joint positions (..., 22, 3)
        tolerance: Maximum allowed deviation from expected ratio (default 0.5)

    Returns:
        List of BoneRatioIssue objects for any problematic ratios
    """
    issues = []

    # Compute bone lengths
    bone_lengths = _compute_bone_lengths(positions)

    # Check ratios within each chain
    for chain_idx, chain in enumerate(T2M_KINEMATIC_CHAIN):
        chain_name = CHAIN_NAMES.get(chain_idx, f"chain_{chain_idx}")

        # Get bone lengths for this chain
        chain_bones = []
        for i in range(len(chain) - 1):
            parent = chain[i]
            child = chain[i + 1]
            bone_name = f"{chain_name}_{parent}_{child}"
            chain_bones.append((bone_name, bone_lengths[bone_name]))

        # Check ratios between consecutive bones
        for i in range(len(chain_bones) - 1):
            bone1_name, bone1_length = chain_bones[i]
            bone2_name, bone2_length = chain_bones[i + 1]

            # Compute ratio (avoid division by zero)
            ratio = bone1_length / (bone2_length + 1e-8)

            # Expected ratio from T2M_RAW_OFFSETS
            # For T2M skeleton, most bones have equal length (ratio ~1.0)
            # except for specific cases
            expected_ratio = 1.0  # Default expectation

            # Get mean ratio across batch/time dimensions
            if ratio.ndim > 0:
                mean_ratio = ratio.mean().item()
                std_ratio = ratio.std().item() if ratio.numel() > 1 else 0.0
            else:
                mean_ratio = ratio.item()
                std_ratio = 0.0

            # Check deviation from expected
            deviation = abs(mean_ratio - expected_ratio)

            if deviation > tolerance:
                # Extract bone pair name for readability
                bone_pair = (
                    f"{bone1_name.split('_')[-2]}->{bone1_name.split('_')[-1]} vs "
                    f"{bone2_name.split('_')[-2]}->{bone2_name.split('_')[-1]}"
                )

                issues.append(
                    BoneRatioIssue(
                        chain_name=chain_name,
                        bone_pair=bone_pair,
                        actual_ratio=mean_ratio,
                        expected_ratio=expected_ratio,
                        deviation=deviation,
                        severity=_get_severity(deviation, (tolerance, tolerance * 2)),
                    )
                )

    return issues


def validate_ric_bounds(
    ric_positions: torch.Tensor,
    bounds: Dict[int, Tuple[float, float, float, float, float, float]] = None,
) -> Dict[str, List[int]]:
    """
    Validate that RIC positions fall within anatomically plausible bounds.

    Args:
        ric_positions: RIC positions (..., 22, 3)
        bounds: Per-joint bounds dictionary (default: DEFAULT_RIC_BOUNDS)

    Returns:
        Dictionary mapping "joint_dim" to list of violating frame indices
    """
    if bounds is None:
        bounds = DEFAULT_RIC_BOUNDS

    violations = {}

    # Handle different input shapes
    if ric_positions.ndim == 3:
        # Shape: (N, 22, 3) - single sequence
        N = ric_positions.shape[0]
        for joint_idx in range(22):
            if joint_idx not in bounds:
                continue

            min_x, max_x, min_y, max_y, min_z, max_z = bounds[joint_idx]

            for dim, (min_val, max_val) in enumerate(
                [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
            ):
                joint_dim = f"joint_{joint_idx}_dim_{dim}"
                values = ric_positions[:, joint_idx, dim]

                # Find violating frames
                violating = torch.where((values < min_val) | (values > max_val))[0]

                if len(violating) > 0:
                    violations[joint_dim] = violating.tolist()

    elif ric_positions.ndim == 4:
        # Shape: (B, N, 22, 3) - batched sequences
        B, N = ric_positions.shape[:2]

        for joint_idx in range(22):
            if joint_idx not in bounds:
                continue

            min_x, max_x, min_y, max_y, min_z, max_z = bounds[joint_idx]

            for dim, (min_val, max_val) in enumerate(
                [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
            ):
                joint_dim = f"joint_{joint_idx}_dim_{dim}"
                values = ric_positions[:, :, joint_idx, dim]

                # Find violating frames
                violating = torch.where((values < min_val) | (values > max_val))

                if len(violating[0]) > 0:
                    # Store as list of (batch, frame) tuples
                    violations[joint_dim] = list(
                        zip(violating[0].tolist(), violating[1].tolist())
                    )

    return violations


def validate_kinematic_chain(
    positions: torch.Tensor,
    dislocation_tolerance: float = 2.0,
    continuity_tolerance: float = 3.0,
) -> List[KinematicIssue]:
    """
    Validate kinematic chain integrity.

    Checks for:
    1. Joint dislocations (child too far from parent)
    2. Unusual bone length ratios within chains
    3. Chain continuity issues

    Args:
        positions: Joint positions (..., 22, 3)
        dislocation_tolerance: Max ratio of actual/expected bone length
        continuity_tolerance: Max ratio change between consecutive bones

    Returns:
        List of KinematicIssue objects
    """
    issues = []
    expected_lengths = _compute_expected_bone_lengths()

    for chain_idx, chain in enumerate(T2M_KINEMATIC_CHAIN):
        chain_name = CHAIN_NAMES.get(chain_idx, f"chain_{chain_idx}")

        chain_bone_lengths = []

        for i in range(len(chain) - 1):
            parent = chain[i]
            child = chain[i + 1]
            bone_name = f"{chain_name}_{parent}_{child}"

            # Compute actual distance
            parent_pos = positions[..., parent, :]
            child_pos = positions[..., child, :]
            actual_dist = torch.norm(child_pos - parent_pos, dim=-1)

            chain_bone_lengths.append((bone_name, actual_dist))

            # Get expected length
            expected_length = expected_lengths.get(bone_name, 1.0)

            if expected_length > 0:
                # Check for dislocation
                max_allowed = expected_length * dislocation_tolerance

                if actual_dist.ndim > 0:
                    # Batched - check each frame
                    dislocated = actual_dist > max_allowed
                    if dislocated.any():
                        # Find which frames are dislocated
                        dislocated_frames = torch.where(dislocated)
                        mean_dist = (
                            actual_dist[dislocated].mean().item()
                            if dislocated.any()
                            else actual_dist.mean().item()
                        )

                        issues.append(
                            KinematicIssue(
                                issue_type="dislocation",
                                chain_name=chain_name,
                                parent_joint=parent,
                                child_joint=child,
                                details={
                                    "actual_distance": mean_dist,
                                    "max_allowed": max_allowed,
                                    "num_affected_frames": (
                                        len(dislocated_frames[0])
                                        if len(dislocated_frames) > 0
                                        else 1
                                    ),
                                },
                                severity=_get_severity(
                                    mean_dist / max_allowed - 1, (0.2, 0.5)
                                ),
                            )
                        )
                else:
                    # Single frame
                    if actual_dist.item() > max_allowed:
                        issues.append(
                            KinematicIssue(
                                issue_type="dislocation",
                                chain_name=chain_name,
                                parent_joint=parent,
                                child_joint=child,
                                details={
                                    "actual_distance": actual_dist.item(),
                                    "max_allowed": max_allowed,
                                },
                                severity=_get_severity(
                                    actual_dist.item() / max_allowed - 1, (0.2, 0.5)
                                ),
                            )
                        )

        # Check chain continuity (unusual ratio changes)
        for i in range(len(chain_bone_lengths) - 1):
            bone1_name, bone1_length = chain_bone_lengths[i]
            bone2_name, bone2_length = chain_bone_lengths[i + 1]

            ratio = bone1_length / (bone2_length + 1e-8)

            if ratio.ndim > 0:
                unusual = (ratio < 1.0 / continuity_tolerance) | (
                    ratio > continuity_tolerance
                )
                if unusual.any():
                    unusual_frames = torch.where(unusual)
                    mean_ratio = (
                        ratio[unusual].mean().item()
                        if unusual.any()
                        else ratio.mean().item()
                    )

                    issues.append(
                        KinematicIssue(
                            issue_type="unusual_ratio",
                            chain_name=chain_name,
                            parent_joint=chain[i],
                            child_joint=chain[i + 2],
                            details={
                                "ratio": mean_ratio,
                                "bone1": bone1_name,
                                "bone2": bone2_name,
                                "num_affected_frames": (
                                    len(unusual_frames[0])
                                    if len(unusual_frames) > 0
                                    else 1
                                ),
                            },
                            severity=_get_severity(abs(mean_ratio - 1.0), (0.5, 1.0)),
                        )
                    )
            else:
                if (
                    ratio.item() < 1.0 / continuity_tolerance
                    or ratio.item() > continuity_tolerance
                ):
                    issues.append(
                        KinematicIssue(
                            issue_type="unusual_ratio",
                            chain_name=chain_name,
                            parent_joint=chain[i],
                            child_joint=chain[i + 2],
                            details={
                                "ratio": ratio.item(),
                                "bone1": bone1_name,
                                "bone2": bone2_name,
                            },
                            severity=_get_severity(abs(ratio.item() - 1.0), (0.5, 1.0)),
                        )
                    )

    return issues


def compute_scaling_metrics(
    positions: torch.Tensor,
) -> ScalingMetrics:
    """
    Compute scaling deviation metrics for a pose.

    Args:
        positions: Joint positions (..., 22, 3)

    Returns:
        ScalingMetrics with scale factors and statistics
    """
    bone_lengths = _compute_bone_lengths(positions)
    expected_lengths = _compute_expected_bone_lengths()

    # Compute scale factors
    scale_factors = {}
    scale_values = []

    for bone_name, actual_length in bone_lengths.items():
        expected = expected_lengths.get(bone_name, 1.0)
        if expected > 0:
            if actual_length.ndim > 0:
                scale = actual_length.mean().item() / expected
            else:
                scale = actual_length.item() / expected

            scale_factors[bone_name] = scale
            scale_values.append(scale)

    # Compute statistics
    if scale_values:
        mean_scale = float(np.mean(scale_values))
        std_scale = float(np.std(scale_values))
        min_scale = float(np.min(scale_values))
        max_scale = float(np.max(scale_values))
    else:
        mean_scale = std_scale = min_scale = max_scale = 1.0

    # Convert bone lengths to scalars
    bone_lengths_scalar = {}
    for bone_name, length in bone_lengths.items():
        if length.ndim > 0:
            bone_lengths_scalar[bone_name] = length.mean().item()
        else:
            bone_lengths_scalar[bone_name] = length.item()

    return ScalingMetrics(
        mean_scale=mean_scale,
        std_scale=std_scale,
        min_scale=min_scale,
        max_scale=max_scale,
        bone_lengths=bone_lengths_scalar,
        scale_factors=scale_factors,
    )


# ============================================================================
# Main Validation Function
# ============================================================================


def validate_pose(
    positions: torch.Tensor,
    check_bone_ratios: bool = True,
    check_ric_bounds: bool = True,
    check_kinematic_chain: bool = True,
    report_scaling: bool = True,
    bone_ratio_tolerance: float = 0.5,
    dislocation_tolerance: float = 2.0,
    ric_bounds: Dict[int, Tuple[float, float, float, float, float, float]] = None,
) -> PoseValidationResult:
    """
    Comprehensive pose validation for generated motion.

    Validates that joint positions conform to the expected skeleton structure
    using multiple checks: bone length ratios, RIC position bounds, and
    kinematic chain integrity.

    Args:
        positions: Joint positions with shape:
            - (22, 3) - single frame
            - (B, 22, 3) - batched single frames
            - (N, 22, 3) - sequence of frames
            - (B, N, 22, 3) - batched sequences
        check_bone_ratios: Whether to validate bone length ratios
        check_ric_bounds: Whether to validate RIC position bounds
        check_kinematic_chain: Whether to validate kinematic chain
        report_scaling: Whether to compute scaling metrics
        bone_ratio_tolerance: Maximum allowed deviation from expected ratio
        dislocation_tolerance: Max ratio of actual/expected bone length
        ric_bounds: Custom RIC bounds (default: DEFAULT_RIC_BOUNDS)

    Returns:
        PoseValidationResult with validation status and details

    Example:
        >>> positions = generator.generate_sequence("walk", num_frames=100)
        >>> result = validate_pose(positions)
        >>> print(result.summary())
        >>> if not result.is_valid:
        ...     print(f"Found {len(result.kinematic_issues)} kinematic issues")
    """
    # Ensure tensor
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).float()

    # Handle different input shapes
    original_shape = positions.shape

    if positions.ndim == 2:
        # Single frame (22, 3) -> add batch dim
        positions = positions.unsqueeze(0)
    elif positions.ndim == 3:
        # Could be (B, 22, 3) or (N, 22, 3)
        # Treat as batched frames for validation
        pass
    elif positions.ndim == 4:
        # (B, N, 22, 3) - batched sequences
        # Flatten to (B*N, 22, 3) for validation
        B, N = positions.shape[:2]
        positions = positions.reshape(B * N, 22, 3)

    # Initialize result components
    bone_ratio_issues = []
    ric_bound_violations = {}
    kinematic_issues = []
    scaling_metrics = None

    # Run validations
    if check_bone_ratios:
        bone_ratio_issues = validate_bone_length_ratios(positions, bone_ratio_tolerance)

    if check_ric_bounds:
        ric_bound_violations = validate_ric_bounds(positions, ric_bounds)

    if check_kinematic_chain:
        kinematic_issues = validate_kinematic_chain(positions, dislocation_tolerance)

    if report_scaling:
        scaling_metrics = compute_scaling_metrics(positions)

    # Determine overall validity
    # A pose is invalid if there are severe issues
    has_severe_issues = (
        any(issue.severity == "severe" for issue in bone_ratio_issues)
        or any(issue.severity == "severe" for issue in kinematic_issues)
        or len(ric_bound_violations) > 10  # Many bound violations indicate a problem
    )

    is_valid = not has_severe_issues

    return PoseValidationResult(
        is_valid=is_valid,
        bone_ratio_issues=bone_ratio_issues,
        ric_bound_violations=ric_bound_violations,
        kinematic_issues=kinematic_issues,
        scaling_metrics=scaling_metrics,
    )


def validate_pose_sequence(
    joints: torch.Tensor,
    frame_threshold: float = 0.1,
    **kwargs,
) -> Tuple[PoseValidationResult, List[PoseValidationResult]]:
    """
    Validate a sequence of poses frame by frame.

    Args:
        joints: Joint positions (B, T, 22, 3) or (T, 22, 3)
        frame_threshold: Fraction of frames that can fail before sequence is invalid
        **kwargs: Additional arguments passed to validate_pose

    Returns:
        Tuple of (overall_result, per_frame_results)
    """
    # Handle input shape
    if joints.ndim == 3:
        # (T, 22, 3) -> add batch dim
        joints = joints.unsqueeze(0)

    B, T = joints.shape[:2]

    # Validate each frame
    per_frame_results = []
    invalid_count = 0

    for t in range(T):
        frame = joints[:, t, :, :]  # (B, 22, 3)
        result = validate_pose(frame, **kwargs)
        per_frame_results.append(result)

        if not result.is_valid:
            invalid_count += 1

    # Create overall result
    invalid_fraction = invalid_count / T
    is_valid = invalid_fraction <= frame_threshold

    # Aggregate issues
    all_bone_issues = []
    all_kinematic_issues = []
    for result in per_frame_results:
        all_bone_issues.extend(result.bone_ratio_issues)
        all_kinematic_issues.extend(result.kinematic_issues)

    # Compute average scaling metrics
    scaling_metrics = None
    if kwargs.get("report_scaling", True):
        scaling_metrics = compute_scaling_metrics(joints.reshape(B * T, 22, 3))

    overall_result = PoseValidationResult(
        is_valid=is_valid,
        bone_ratio_issues=all_bone_issues,
        ric_bound_violations={},  # Aggregated violations not computed
        kinematic_issues=all_kinematic_issues,
        scaling_metrics=scaling_metrics,
    )

    return overall_result, per_frame_results


# ============================================================================
# Utility Functions
# ============================================================================


def extract_ric_from_positions(
    positions: torch.Tensor,
    root_rotation: torch.Tensor = None,
) -> torch.Tensor:
    """
    Extract RIC (Root-Invariant Coordinates) from global joint positions.

    RIC positions are local to the root joint and rotated to the root's
    local coordinate frame.

    Args:
        positions: Global joint positions (..., 22, 3)
        root_rotation: Root rotation as quaternion (..., 4) or 6D (..., 6)
                      If None, identity rotation is used

    Returns:
        RIC positions (..., 22, 3)
    """
    from utils.motion_utils import qrot, qinv, cont6d_to_quaternion

    # Center on root
    root_pos = positions[..., 0:1, :]  # (..., 1, 3)
    local_pos = positions - root_pos  # (..., 22, 3)

    # Rotate to root-local frame
    if root_rotation is not None:
        # Handle 6D rotation
        if root_rotation.shape[-1] == 6:
            root_quat = cont6d_to_quaternion(root_rotation)
        else:
            root_quat = root_rotation

        # Expand for all joints
        root_quat_expanded = root_quat.unsqueeze(-2).expand(
            root_quat.shape[:-1] + (22, 4)
        )

        # Rotate local positions
        ric = qrot(root_quat_expanded, local_pos)
    else:
        ric = local_pos

    return ric


def get_joint_names() -> Dict[int, str]:
    """Get human-readable joint names for the T2M skeleton."""
    return {
        0: "root",
        1: "right_hip",
        2: "left_hip",
        3: "spine",
        4: "right_knee",
        5: "left_knee",
        6: "spine1",
        7: "right_ankle",
        8: "left_ankle",
        9: "spine2_chest",
        10: "right_foot",
        11: "left_foot",
        12: "neck",
        13: "right_shoulder",
        14: "left_shoulder",
        15: "head",
        16: "right_elbow",
        17: "left_elbow",
        18: "right_wrist",
        19: "left_wrist",
        20: "right_hand",
        21: "left_hand",
    }
