"""
Generate motion using HumanMotionGenerator and validate poses.

This script:
1. Loads a trained model from checkpoint
2. Generates motion using the text prompt from sample data
3. Validates the generated poses
4. Compares with ground truth
5. Identifies where the model is going wrong
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from utils.pose_validation import (
    validate_pose,
    validate_pose_sequence,
    compute_scaling_metrics,
    get_joint_names,
)
from utils.motion_utils import features_to_positions, T2M_KINEMATIC_CHAIN
from models import HumanMotionGenerator
from config import Config


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def compare_generated_vs_ground_truth(
    generated_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    generated_features: torch.Tensor = None,
    gt_features: torch.Tensor = None,
):
    """Compare generated motion with ground truth in detail."""

    print_separator("Position Comparison")

    # Compute position errors
    if generated_joints.shape[1] > gt_joints.shape[0]:
        # Generated more frames than GT
        generated_joints = generated_joints[:, : gt_joints.shape[0]]
    elif generated_joints.shape[1] < gt_joints.shape[0]:
        # Generated fewer frames than GT
        gt_joints = gt_joints[: generated_joints.shape[1]]

    position_error = torch.mean(
        (generated_joints.squeeze(0) - gt_joints) ** 2, dim=-1
    )  # (N, 22)

    print(f"\nPosition Error Statistics:")
    print(f"  Mean MSE: {position_error.mean().item():.6f}")
    print(f"  Max MSE: {position_error.max().item():.6f}")
    print(f"  Min MSE: {position_error.min().item():.6f}")

    # Per-joint error
    print(f"\nPer-Joint Position Error (sorted by severity):")
    joint_names = get_joint_names()
    per_joint_error = position_error.mean(dim=0)  # (22,)
    sorted_idx = torch.argsort(per_joint_error, descending=True)

    for idx_tensor in sorted_idx[:10]:
        idx = idx_tensor.item() if isinstance(idx_tensor, torch.Tensor) else idx_tensor
        print(
            f"  Joint {idx:2d} ({joint_names[idx]:15s}): MSE = {per_joint_error[idx].item():.6f}"
        )

    # Frame-by-frame error
    print(f"\nFrame-by-Frame Position Error:")
    frame_error = position_error.mean(dim=1)  # (N,)

    # Find worst frames
    worst_frames = torch.argsort(frame_error, descending=True)[:5]
    print(f"  Worst frames: {worst_frames.tolist()}")
    for frame_idx in worst_frames:
        print(f"    Frame {frame_idx}: MSE = {frame_error[frame_idx].item():.6f}")

    return position_error


def analyze_generated_motion(
    checkpoint_path: str,
    text_prompt: str,
    gt_joint_path: str,
    gt_vec_path: str,
    num_frames: int = 60,
    device: str = "cpu",
):
    """Full analysis of generated motion vs ground truth."""

    # =========================================================================
    # Load Ground Truth
    # =========================================================================
    print_separator("Loading Ground Truth Data")

    gt_joints = np.load(gt_joint_path)
    gt_vecs = np.load(gt_vec_path)

    print(f"Ground truth joints shape: {gt_joints.shape}")
    print(f"Ground truth features shape: {gt_vecs.shape}")
    print(f"Text prompt: {text_prompt}")

    gt_joints_t = torch.from_numpy(gt_joints).float()
    gt_vecs_t = torch.from_numpy(gt_vecs).float()

    # =========================================================================
    # Load Model
    # =========================================================================
    print_separator("Loading Model")

    config = Config()

    try:
        generator = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path, config, device=device
        )
        generator.to(device)
        generator.eval()
        print(f"Model loaded from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # =========================================================================
    # Generate Motion
    # =========================================================================
    print_separator("Generating Motion")

    # Use ground truth initial features as input
    # Take first few frames as history
    num_history_frames = 1
    input_features = (
        gt_vecs_t[:num_history_frames].unsqueeze(0).to(device)
    )  # (1, 1, 271)

    print(f"Input features shape: {input_features.shape}")
    print(f"Generating {num_frames} frames...")

    try:
        with torch.no_grad():
            generated_joints = generator.generate_sequence(
                text=text_prompt,
                num_frames=num_frames,
                num_steps=10,
                guidance_scale=2.5,
                input_features=input_features,
            )
        print(f"Generated joints shape: {generated_joints.shape}")
    except Exception as e:
        print(f"Error generating motion: {e}")
        import traceback

        traceback.print_exc()
        return None

    # =========================================================================
    # Validate Generated Motion
    # =========================================================================
    print_separator("Validating Generated Motion")

    # Validate each frame
    overall_result, per_frame_results = validate_pose_sequence(
        generated_joints,
        check_bone_ratios=True,
        check_ric_bounds=False,  # Global positions
        check_kinematic_chain=True,
        report_scaling=True,
    )

    print(f"\nOverall valid: {overall_result.is_valid}")
    print(
        f"Valid frames: {sum(1 for r in per_frame_results if r.is_valid)}/{len(per_frame_results)}"
    )

    if overall_result.scaling_metrics:
        print(f"\nGenerated Motion Scaling Metrics:")
        print(f"  Mean scale: {overall_result.scaling_metrics.mean_scale:.4f}")
        print(f"  Std scale: {overall_result.scaling_metrics.std_scale:.4f}")
        print(
            f"  Range: [{overall_result.scaling_metrics.min_scale:.4f}, {overall_result.scaling_metrics.max_scale:.4f}]"
        )

    # =========================================================================
    # Validate Ground Truth
    # =========================================================================
    print_separator("Validating Ground Truth")

    gt_overall, gt_per_frame = validate_pose_sequence(
        gt_joints_t[:num_frames].unsqueeze(0),
        check_bone_ratios=True,
        check_ric_bounds=False,
        check_kinematic_chain=True,
        report_scaling=True,
    )

    print(f"\nGround truth valid: {gt_overall.is_valid}")
    print(
        f"Valid frames: {sum(1 for r in gt_per_frame if r.is_valid)}/{len(gt_per_frame)}"
    )

    if gt_overall.scaling_metrics:
        print(f"\nGround Truth Scaling Metrics:")
        print(f"  Mean scale: {gt_overall.scaling_metrics.mean_scale:.4f}")
        print(f"  Std scale: {gt_overall.scaling_metrics.std_scale:.4f}")
        print(
            f"  Range: [{gt_overall.scaling_metrics.min_scale:.4f}, {gt_overall.scaling_metrics.max_scale:.4f}]"
        )

    # =========================================================================
    # Compare Generated vs Ground Truth
    # =========================================================================
    position_error = compare_generated_vs_ground_truth(generated_joints, gt_joints_t)

    # =========================================================================
    # Detailed Issue Analysis
    # =========================================================================
    print_separator("Detailed Issue Analysis")

    # Find frames with most issues
    frames_with_issues = []
    for i, result in enumerate(per_frame_results):
        num_issues = len(result.bone_ratio_issues) + len(result.kinematic_issues)
        if num_issues > 0:
            frames_with_issues.append((i, num_issues, result))

    if frames_with_issues:
        frames_with_issues.sort(key=lambda x: x[1], reverse=True)
        print(f"\nFrames with most issues:")
        for frame_idx, num_issues, result in frames_with_issues[:5]:
            print(f"\n  Frame {frame_idx}: {num_issues} issues")
            for issue in result.kinematic_issues[:3]:
                print(
                    f"    [{issue.severity}] {issue.issue_type}: joints {issue.parent_joint}->{issue.child_joint}"
                )
    else:
        print("\nNo frames with issues found.")

    # =========================================================================
    # Scaling Comparison
    # =========================================================================
    print_separator("Scaling Comparison")

    gen_metrics = compute_scaling_metrics(generated_joints.squeeze(0))
    gt_metrics = compute_scaling_metrics(gt_joints_t[:num_frames])

    print("\nBone Length Comparison (Generated vs Ground Truth):")
    print(f"{'Bone':<25} {'Generated':>12} {'Ground Truth':>12} {'Ratio':>10}")
    print("-" * 60)

    for bone_name in sorted(gen_metrics.bone_lengths.keys()):
        gen_len = gen_metrics.bone_lengths[bone_name]
        gt_len = gt_metrics.bone_lengths.get(bone_name, 0)
        ratio = gen_len / (gt_len + 1e-8)
        print(f"{bone_name:<25} {gen_len:>12.4f} {gt_len:>12.4f} {ratio:>10.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_separator("Summary")

    print(
        f"""
Analysis Summary:
-----------------
Text prompt: "{text_prompt}"

Generated Motion:
  - Frames: {generated_joints.shape[1]}
  - Valid: {overall_result.is_valid}
  - Mean scale: {gen_metrics.mean_scale:.4f}
  - Scale std: {gen_metrics.std_scale:.4f}

Ground Truth:
  - Frames: {min(num_frames, gt_joints.shape[0])}
  - Valid: {gt_overall.is_valid}
  - Mean scale: {gt_metrics.mean_scale:.4f}
  - Scale std: {gt_metrics.std_scale:.4f}

Position Error:
  - Mean MSE: {position_error.mean().item():.6f}
  - Max MSE: {position_error.max().item():.6f}

Potential Issues:
"""
    )

    # Identify potential issues
    issues_found = []

    if gen_metrics.mean_scale < gt_metrics.mean_scale * 0.5:
        issues_found.append("Generated motion has much smaller scale than ground truth")
    elif gen_metrics.mean_scale > gt_metrics.mean_scale * 2.0:
        issues_found.append("Generated motion has much larger scale than ground truth")

    if gen_metrics.std_scale > gt_metrics.std_scale * 2.0:
        issues_found.append("Generated motion has inconsistent bone scaling")

    if position_error.mean().item() > 0.1:
        issues_found.append("Large position error compared to ground truth")

    severe_kinematic = sum(
        1
        for r in per_frame_results
        for i in r.kinematic_issues
        if i.severity == "severe"
    )
    if severe_kinematic > 0:
        issues_found.append(f"{severe_kinematic} severe kinematic issues found")

    if issues_found:
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("  No major issues identified.")

    return {
        "generated_joints": generated_joints,
        "gt_joints": gt_joints_t,
        "position_error": position_error,
        "overall_result": overall_result,
        "per_frame_results": per_frame_results,
    }


if __name__ == "__main__":
    # Configuration
    checkpoint_path = "tests/checkpoints/best.pt"
    text_prompt = "a person walks one way then backtracks"
    gt_joint_path = "sample_data/000070_joint.npy"
    gt_vec_path = "sample_data/000070_vec.npy"
    num_frames = 60
    device = "cpu"  # Use "cuda" if available

    # Run analysis
    results = analyze_generated_motion(
        checkpoint_path=checkpoint_path,
        text_prompt=text_prompt,
        gt_joint_path=gt_joint_path,
        gt_vec_path=gt_vec_path,
        num_frames=num_frames,
        device=device,
    )
