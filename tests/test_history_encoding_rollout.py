"""
History Encoding and Rollout Performance Test.

Tests:
1. Ground truth history encoding with increasing seed frames
2. Teacher forcing vs autoregressive rollout comparison
3. Long horizon rollout performance

This helps understand:
- How well the encoder uses ground truth history
- Error accumulation in autoregressive rollout
- Long horizon performance limits
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
import numpy as np
from typing import Dict, List


def predict_one_frame(
    generator,
    text_embedding: torch.Tensor,
    history: torch.Tensor,
    device: torch.device,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    Predict a single frame given history.

    Args:
        generator: HumanMotionGenerator model
        text_embedding: (1, 1, 512) text embedding
        history: (1, N, 271) history frames
        device: torch device
        num_steps: flow matching steps

    Returns:
        predicted_72d: (1, 72) predicted frame in 72D format
    """
    from utils.train_utils import extract_prev_frame_features

    with torch.no_grad():
        # Encode context
        context = generator.encoder(
            batch_size=1,
            text=text_embedding,
            input_features=history,
        )[
            :, -1, :, :
        ]  # (1, 22, out_dim)

        # Extract prev frame features
        prev_features = extract_prev_frame_features(history[:, -1])  # (1, 261)

        # Flow matching prediction
        x_t = torch.randn((1, 72), device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((1,), step * dt, device=device)
            v = generator.predictor(
                history_features=context,
                noise_level=t,
                noisy_target=x_t,
                prev_frame_features=prev_features,
            )
            x_t = x_t + v * dt

        return x_t  # (1, 72)


def test_ground_truth_history_encoding():
    """
    Test encoder with increasing ground truth seed frames.
    Predict only 1 frame ahead to isolate encoder quality.
    """
    print("=" * 70)
    print("Ground Truth History Encoding Test")
    print("=" * 70)

    device = torch.device("cpu")

    # Load data
    vecs = np.load("sample_data/000070_vec.npy")
    joints = np.load("sample_data/000070_joint.npy")

    with open("sample_data/000070.txt", "r") as f:
        text = f.readlines()[0].strip().split("#")[0]

    vecs_t = torch.from_numpy(vecs).float()
    joints_t = torch.from_numpy(joints).float()

    # Load model
    from src.models import HumanMotionGenerator
    from src.config import Config
    from src.utils.text_encoder import CLIPEncoder
    from src.utils.train_utils import extract_clean_target

    config = Config()
    config.max_text_seq_len = 1

    generator = HumanMotionGenerator.load_from_checkpoint(
        "tests/checkpoints/best.pt", config, device=str(device)
    )
    generator.eval()

    clip_encoder = CLIPEncoder()
    text_embedding = clip_encoder(text)

    # Test with increasing seed frames
    seed_counts = [1, 3, 5, 10, 20]
    results = {}

    print(f"\n1. Testing with increasing ground truth seed frames:")
    print(f"   (Predicting 1 frame ahead)")

    for num_seeds in seed_counts:
        if num_seeds >= len(vecs_t) - 1:
            continue

        # Use ground truth frames as history
        history = vecs_t[:num_seeds].unsqueeze(0)  # (1, N, 271)

        # Predict next frame
        predicted_72d = predict_one_frame(generator, text_embedding, history, device)

        # Ground truth next frame
        gt_next_271d = vecs_t[num_seeds : num_seeds + 1]  # (1, 271)
        gt_next_72d = extract_clean_target(gt_next_271d)  # (1, 72)

        # Compute errors
        mse_72d = torch.mean((predicted_72d - gt_next_72d) ** 2).item()

        # RIC error (pose)
        pred_ric = predicted_72d[:, 9:72].reshape(1, 21, 3)
        gt_ric = gt_next_72d[:, 9:72].reshape(1, 21, 3)
        ric_mse = torch.mean((pred_ric - gt_ric) ** 2).item()

        # Root error
        pred_root = predicted_72d[:, :9]
        gt_root = gt_next_72d[:, :9]
        root_mse = torch.mean((pred_root - gt_root) ** 2).item()

        results[num_seeds] = {
            "mse_72d": mse_72d,
            "ric_mse": ric_mse,
            "root_mse": root_mse,
        }

        print(
            f"   Seeds: {num_seeds:2d} | 72D MSE: {mse_72d:.4f} | RIC MSE: {ric_mse:.4f} | Root MSE: {root_mse:.4f}"
        )

    # Summary table
    print(f"\n2. Summary Table:")
    print(f"   Seeds | 72D MSE | RIC MSE | Root MSE")
    print(f"   ------|---------|---------|---------")
    for num_seeds in seed_counts:
        if num_seeds in results:
            r = results[num_seeds]
            print(
                f"   {num_seeds:5d} | {r['mse_72d']:7.4f} | {r['ric_mse']:7.4f} | {r['root_mse']:8.4f}"
            )

    print("\n" + "=" * 70)
    print("[DONE] Ground Truth History Encoding Test")
    print("=" * 70)

    return results


def test_teacher_forcing_vs_autoregressive():
    """
    Compare teacher forcing with autoregressive rollout.

    Teacher forcing: Always use ground truth as history
    Autoregressive: Use predicted frames as history
    """
    print("\n" + "=" * 70)
    print("Teacher Forcing vs Autoregressive Rollout Test")
    print("=" * 70)

    device = torch.device("cpu")

    # Load data
    vecs = np.load("sample_data/000070_vec.npy")
    joints = np.load("sample_data/000070_joint.npy")

    with open("sample_data/000070.txt", "r") as f:
        text = f.readlines()[0].strip().split("#")[0]

    vecs_t = torch.from_numpy(vecs).float()
    joints_t = torch.from_numpy(joints).float()

    # Load model
    from src.models import HumanMotionGenerator
    from src.config import Config
    from src.utils.text_encoder import CLIPEncoder
    from src.utils.train_utils import extract_clean_target
    from src.utils.motion_utils import (
        IncrementalFeatureExtractor,
        features_to_positions,
    )

    config = Config()
    config.max_text_seq_len = 1

    generator = HumanMotionGenerator.load_from_checkpoint(
        "tests/checkpoints/best.pt", config, device=str(device)
    )
    generator.eval()

    clip_encoder = CLIPEncoder()
    text_embedding = clip_encoder(text)

    # Test parameters
    num_seeds = 5
    num_predict = 10

    print(f"\n1. Test setup:")
    print(f"   Seed frames: {num_seeds}")
    print(f"   Predict frames: {num_predict}")

    # === Teacher Forcing ===
    print(f"\n2. Running Teacher Forcing (always use GT history)...")
    tf_errors = []
    tf_ric_errors = []

    for i in range(num_predict):
        # Always use ground truth as history
        history = vecs_t[: num_seeds + i].unsqueeze(0)  # (1, N, 271)

        # Predict next frame
        predicted_72d = predict_one_frame(generator, text_embedding, history, device)

        # Ground truth
        gt_next_271d = vecs_t[num_seeds + i : num_seeds + i + 1]
        gt_next_72d = extract_clean_target(gt_next_271d)

        # Errors
        mse_72d = torch.mean((predicted_72d - gt_next_72d) ** 2).item()
        ric_mse = torch.mean(
            (predicted_72d[:, 9:72] - gt_next_72d[:, 9:72]) ** 2
        ).item()

        tf_errors.append(mse_72d)
        tf_ric_errors.append(ric_mse)

    # === Autoregressive ===
    print(f"\n3. Running Autoregressive (use predicted frames as history)...")
    ar_errors = []
    ar_ric_errors = []

    # Initialize with ground truth seeds
    history = vecs_t[:num_seeds].unsqueeze(0)  # (1, N, 271)
    current_frame_271d = vecs_t[num_seeds - 1]  # (271,)

    # Initialize extractor for position reconstruction
    init_joints = features_to_positions(history[0])  # (N, 22, 3)
    extractor = IncrementalFeatureExtractor(dataset_type="t2m", device=device)
    extractor.initialize(init_joints[-1:])

    for i in range(num_predict):
        # Predict next frame
        predicted_72d = predict_one_frame(generator, text_embedding, history, device)

        # Ground truth
        gt_next_271d = vecs_t[num_seeds + i : num_seeds + i + 1]
        gt_next_72d = extract_clean_target(gt_next_271d)

        # Errors
        mse_72d = torch.mean((predicted_72d - gt_next_72d) ** 2).item()
        ric_mse = torch.mean(
            (predicted_72d[:, 9:72] - gt_next_72d[:, 9:72]) ** 2
        ).item()

        ar_errors.append(mse_72d)
        ar_ric_errors.append(ric_mse)

        # Update history with predicted frame (convert 72D to 271D)
        new_frame_271d, _ = extractor.process_flow_output(
            predicted_72d, current_frame_271d.unsqueeze(0)
        )
        current_frame_271d = new_frame_271d[0]
        history = new_frame_271d.unsqueeze(0)  # (1, 1, 271) - single frame history

    # === Comparison ===
    print(f"\n4. Comparison Results:")
    print(f"   Frame | TF Error | AR Error | Gap     | TF RIC  | AR RIC  | RIC Gap")
    print(f"   ------|----------|----------|---------|---------|---------|--------")
    for i in range(num_predict):
        gap = ar_errors[i] - tf_errors[i]
        ric_gap = ar_ric_errors[i] - tf_ric_errors[i]
        print(
            f"   {i + 1:5d} | {tf_errors[i]:8.4f} | {ar_errors[i]:8.4f} | {gap:7.4f} | "
            f"{tf_ric_errors[i]:7.4f} | {ar_ric_errors[i]:7.4f} | {ric_gap:7.4f}"
        )

    # Summary statistics
    print(f"\n5. Summary Statistics:")
    print(f"   Teacher Forcing:")
    print(f"     Mean 72D MSE: {np.mean(tf_errors):.4f}")
    print(f"     Mean RIC MSE: {np.mean(tf_ric_errors):.4f}")
    print(f"   Autoregressive:")
    print(f"     Mean 72D MSE: {np.mean(ar_errors):.4f}")
    print(f"     Mean RIC MSE: {np.mean(ar_ric_errors):.4f}")
    print(f"   Gap (AR - TF):")
    print(f"     Mean 72D Gap: {np.mean(ar_errors) - np.mean(tf_errors):.4f}")
    print(f"     Mean RIC Gap: {np.mean(ar_ric_errors) - np.mean(tf_ric_errors):.4f}")

    # Interpretation
    print(f"\n6. Interpretation:")
    tf_ar_ratio = (
        np.mean(ar_errors) / np.mean(tf_errors)
        if np.mean(tf_errors) > 0
        else float("inf")
    )
    print(f"   AR/TF Error Ratio: {tf_ar_ratio:.2f}x")
    if tf_ar_ratio < 1.5:
        print(
            f"   [GOOD] Autoregressive close to teacher forcing - encoder handles predicted history well"
        )
    elif tf_ar_ratio < 3.0:
        print(f"   [OK] Moderate gap - some error accumulation from predicted history")
    else:
        print(f"   [WARN] Large gap - encoder struggles with predicted history")

    print("\n" + "=" * 70)
    print("[DONE] Teacher Forcing vs Autoregressive Test")
    print("=" * 70)

    return {
        "tf_errors": tf_errors,
        "ar_errors": ar_errors,
        "tf_ric_errors": tf_ric_errors,
        "ar_ric_errors": ar_ric_errors,
    }


def test_long_horizon_rollout():
    """
    Test long horizon rollout performance.
    """
    print("\n" + "=" * 70)
    print("Long Horizon Rollout Test")
    print("=" * 70)

    device = torch.device("cpu")

    # Load data
    vecs = np.load("sample_data/000070_vec.npy")
    joints = np.load("sample_data/000070_joint.npy")

    with open("sample_data/000070.txt", "r") as f:
        text = f.readlines()[0].strip().split("#")[0]

    vecs_t = torch.from_numpy(vecs).float()
    joints_t = torch.from_numpy(joints).float()

    # Load model
    from src.models import HumanMotionGenerator
    from src.config import Config

    config = Config()
    config.max_text_seq_len = 1

    generator = HumanMotionGenerator.load_from_checkpoint(
        "tests/checkpoints/best.pt", config, device=str(device)
    )
    generator.eval()

    # Test parameters
    seed_counts = [5, 10]
    predict_counts = [10, 20, 50]

    print(f"\n1. Test setup:")
    print(f"   Seed counts: {seed_counts}")
    print(f"   Predict counts: {predict_counts}")

    results = {}

    for num_seeds in seed_counts:
        for num_predict in predict_counts:
            if num_seeds + num_predict > len(vecs_t):
                continue

            print(f"\n2. Testing {num_seeds} seeds, {num_predict} predictions...")

            # Use generate_sequence for proper rollout
            seed_features = vecs_t[:num_seeds].unsqueeze(0)

            with torch.no_grad():
                generated = generator.generate_sequence(
                    text=text,
                    num_frames=num_predict,
                    num_steps=10,
                    guidance_scale=1.0,
                    input_features=seed_features,
                    dataset_type="t2m",
                )

            # Compare with ground truth
            gt_joints = joints_t[num_seeds : num_seeds + num_predict]
            gen_joints = generated[0]

            # Per-frame errors
            errors = []
            for i in range(num_predict):
                mse = torch.mean((gen_joints[i] - gt_joints[i]) ** 2).item()
                errors.append(mse)

            key = f"seeds{num_seeds}_predict{num_predict}"
            results[key] = {
                "num_seeds": num_seeds,
                "num_predict": num_predict,
                "errors": errors,
                "mean_error": np.mean(errors),
                "first_error": errors[0],
                "last_error": errors[-1],
            }

            print(f"   Mean Error: {np.mean(errors):.4f}")
            print(f"   First Frame: {errors[0]:.4f}, Last Frame: {errors[-1]:.4f}")

    # Summary table
    print(f"\n3. Summary Table:")
    print(f"   Config        | Mean Error | First    | Last     | Growth")
    print(f"   --------------|------------|----------|----------|--------")
    for key, r in results.items():
        growth = (
            r["last_error"] / r["first_error"] if r["first_error"] > 0 else float("inf")
        )
        print(
            f"   {key:13s} | {r['mean_error']:10.4f} | {r['first_error']:8.4f} | "
            f"{r['last_error']:8.4f} | {growth:6.2f}x"
        )

    print("\n" + "=" * 70)
    print("[DONE] Long Horizon Rollout Test")
    print("=" * 70)

    return results


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# HISTORY ENCODING AND ROLLOUT PERFORMANCE TEST SUITE")
    print("#" * 70)

    # Test 1: Ground truth history encoding
    gt_encoding_results = test_ground_truth_history_encoding()

    # Test 2: Teacher forcing vs autoregressive
    tf_ar_results = test_teacher_forcing_vs_autoregressive()

    # Test 3: Long horizon rollout
    long_horizon_results = test_long_horizon_rollout()

    # Final summary
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    print(f"\nGround Truth History Encoding:")
    for seeds, r in gt_encoding_results.items():
        print(f"  {seeds} seeds: 72D MSE = {r['mse_72d']:.4f}")

    print(f"\nTeacher Forcing vs Autoregressive:")
    print(f"  TF Mean Error: {np.mean(tf_ar_results['tf_errors']):.4f}")
    print(f"  AR Mean Error: {np.mean(tf_ar_results['ar_errors']):.4f}")
    print(
        f"  AR/TF Ratio: {np.mean(tf_ar_results['ar_errors']) / np.mean(tf_ar_results['tf_errors']):.2f}x"
    )

    print(f"\nLong Horizon Rollout:")
    for key, r in long_horizon_results.items():
        print(
            f"  {key}: Mean = {r['mean_error']:.4f}, Growth = {r['last_error'] / r['first_error']:.2f}x"
        )
