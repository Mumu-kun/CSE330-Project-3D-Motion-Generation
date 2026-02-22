"""
Detailed motion validation for different generation scenarios.

Tests:
1. Single frame prediction (1 seed, predict 1 frame)
2. Short horizon with 20 seeds (20 seed frames, predict 10 frames)
3. Long horizon prediction (1 seed, predict 60 frames)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from utils.pose_validation import (
    validate_pose_sequence,
    compute_scaling_metrics,
    get_joint_names,
)
from models import HumanMotionGenerator
from config import Config


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def analyze_scenario(
    generator: HumanMotionGenerator,
    gt_joints: torch.Tensor,
    gt_vecs: torch.Tensor,
    text_prompt: str,
    num_seed_frames: int,
    num_predict_frames: int,
    scenario_name: str,
    device: str = "cpu",
):
    """Analyze a specific generation scenario."""

    print_separator(f"SCENARIO: {scenario_name}")

    # Setup seed frames
    input_features = gt_vecs[:num_seed_frames].unsqueeze(0).to(device)  # (1, N, 271)

    print(f"\nConfiguration:")
    print(f"  Seed frames: {num_seed_frames}")
    print(f"  Predict frames: {num_predict_frames}")
    print(f"  Input features shape: {input_features.shape}")

    # Generate
    with torch.no_grad():
        generated_joints = generator.generate_sequence(
            text=text_prompt,
            num_frames=num_predict_frames,
            num_steps=10,
            guidance_scale=2.5,
            input_features=input_features,
        )

    print(f"  Generated joints shape: {generated_joints.shape}")

    # Compare with ground truth
    gt_slice = gt_joints[num_seed_frames : num_seed_frames + num_predict_frames]
    gen_slice = generated_joints.squeeze(0)

    # Align lengths
    min_len = min(gt_slice.shape[0], gen_slice.shape[0])
    gt_slice = gt_slice[:min_len]
    gen_slice = gen_slice[:min_len]

    # Position error
    position_error = torch.mean((gen_slice - gt_slice) ** 2, dim=-1)  # (N, 22)

    print(f"\nPosition Error Statistics:")
    print(f"  Mean MSE: {position_error.mean().item():.6f}")
    print(f"  Max MSE: {position_error.max().item():.6f}")
    print(f"  Min MSE: {position_error.min().item():.6f}")

    # Per-frame error
    frame_error = position_error.mean(dim=1)  # (N,)
    print(f"\nPer-Frame Error (first 10 frames):")
    for i in range(min(10, len(frame_error))):
        print(f"  Frame {i}: MSE = {frame_error[i].item():.6f}")

    # Scaling analysis
    gen_metrics = compute_scaling_metrics(gen_slice)
    gt_metrics = compute_scaling_metrics(gt_slice)

    print(f"\nScaling Comparison:")
    print(f"  Generated mean scale: {gen_metrics.mean_scale:.4f}")
    print(f"  Ground truth mean scale: {gt_metrics.mean_scale:.4f}")
    print(
        f"  Scale ratio: {gen_metrics.mean_scale / (gt_metrics.mean_scale + 1e-8):.2f}x"
    )

    # Validate poses
    overall_result, per_frame_results = validate_pose_sequence(
        generated_joints,
        check_bone_ratios=True,
        check_ric_bounds=False,
        check_kinematic_chain=True,
        report_scaling=True,
    )

    print(f"\nPose Validation:")
    print(f"  Overall valid: {overall_result.is_valid}")
    valid_count = sum(1 for r in per_frame_results if r.is_valid)
    print(f"  Valid frames: {valid_count}/{len(per_frame_results)}")

    # Count severe issues
    severe_issues = sum(
        1
        for r in per_frame_results
        for i in r.kinematic_issues
        if i.severity == "severe"
    )
    print(f"  Severe kinematic issues: {severe_issues}")

    return {
        "position_error": position_error,
        "frame_error": frame_error,
        "gen_metrics": gen_metrics,
        "gt_metrics": gt_metrics,
        "generated_joints": generated_joints,
    }


def main():
    # Configuration
    checkpoint_path = "tests/checkpoints/best.pt"
    text_prompt = "a person walks one way then backtracks"
    gt_joint_path = "sample_data/000070_joint.npy"
    gt_vec_path = "sample_data/000070_vec.npy"
    device = "cpu"

    # Load ground truth
    print_separator("Loading Data")

    gt_joints = torch.from_numpy(np.load(gt_joint_path)).float()
    gt_vecs = torch.from_numpy(np.load(gt_vec_path)).float()

    print(f"Ground truth joints shape: {gt_joints.shape}")
    print(f"Ground truth features shape: {gt_vecs.shape}")
    print(f"Text prompt: {text_prompt}")

    # Load model
    print_separator("Loading Model")

    config = Config()
    generator = HumanMotionGenerator.load_from_checkpoint(
        checkpoint_path, config, device=device
    )
    generator.to(device)
    generator.eval()
    print(f"Model loaded from: {checkpoint_path}")

    # =========================================================================
    # SCENARIO 1: Single Frame Prediction
    # =========================================================================
    results_single = analyze_scenario(
        generator=generator,
        gt_joints=gt_joints,
        gt_vecs=gt_vecs,
        text_prompt=text_prompt,
        num_seed_frames=1,
        num_predict_frames=1,
        scenario_name="Single Frame Prediction (1 seed to 1 frame)",
        device=device,
    )

    # =========================================================================
    # SCENARIO 2: Short Horizon with 20 Seeds
    # =========================================================================
    results_short = analyze_scenario(
        generator=generator,
        gt_joints=gt_joints,
        gt_vecs=gt_vecs,
        text_prompt=text_prompt,
        num_seed_frames=20,
        num_predict_frames=10,
        scenario_name="Short Horizon (20 seeds to 10 frames)",
        device=device,
    )

    # =========================================================================
    # SCENARIO 3: Long Horizon Prediction
    # =========================================================================
    results_long = analyze_scenario(
        generator=generator,
        gt_joints=gt_joints,
        gt_vecs=gt_vecs,
        text_prompt=text_prompt,
        num_seed_frames=1,
        num_predict_frames=60,
        scenario_name="Long Horizon (1 seed to 60 frames)",
        device=device,
    )

    # =========================================================================
    # Summary Comparison
    # =========================================================================
    print_separator("SUMMARY COMPARISON")

    print("\n" + "=" * 80)
    print(
        f"{'Scenario':<40} {'Mean MSE':<12} {'Scale Ratio':<12} {'Severe Issues':<15}"
    )
    print("=" * 80)

    scenarios = [
        ("Single Frame (1 to 1)", results_single),
        ("Short Horizon (20 to 10)", results_short),
        ("Long Horizon (1 to 60)", results_long),
    ]

    for name, results in scenarios:
        mean_mse = results["position_error"].mean().item()
        scale_ratio = results["gen_metrics"].mean_scale / (
            results["gt_metrics"].mean_scale + 1e-8
        )

        # Count severe issues
        _, per_frame = validate_pose_sequence(
            results["generated_joints"],
            check_bone_ratios=True,
            check_ric_bounds=False,
            check_kinematic_chain=True,
        )
        severe = sum(
            1 for r in per_frame for i in r.kinematic_issues if i.severity == "severe"
        )

        print(f"{name:<40} {mean_mse:<12.4f} {scale_ratio:<12.2f}x {severe:<15}")

    print("=" * 80)

    # Analysis
    print("\n" + "=" * 70)
    print(" ANALYSIS")
    print("=" * 70)

    print(
        """
Key Observations:

1. Single Frame Prediction:
   - Should have lowest error if model is working correctly
   - Scale should match ground truth
   
2. Short Horizon (20 seeds):
   - More context should help prediction
   - Error should be moderate
   
3. Long Horizon (1 seed → 60 frames):
   - Error accumulates over time
   - Scale drift is the main issue

The scale ratio shows how much larger/smaller the generated motion is
compared to ground truth. A ratio of ~1.0 is ideal.

If all scenarios show large scale ratios, the issue is in the
preprocessing/reconstruction pipeline, not exposure bias.
"""
    )


if __name__ == "__main__":
    main()
