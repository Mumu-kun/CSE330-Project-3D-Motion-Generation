"""
Flow Predictor Comparison Test.

Compares FlowMatchingPredictor output with ground truth to understand:
1. How well the model predicts within a small prediction horizon
2. How error accumulates across increasing number of predicted frames
3. Proper alignment of predicted frames with ground truth (accounting for seed frames)
4. RIC position errors (pose correctness)
5. Local velocity errors (motion dynamics)

Data sources:
- sample_data/000070_vec.npy - 271D motion features
- sample_data/000070_joint.npy - Ground truth joint positions
- sample_data/000070.txt - Text prompt
- tests/checkpoints/best.pt - Model checkpoint
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch
import numpy as np
from typing import Tuple, List, Dict


def extract_ground_truth_72d(frame_271d: torch.Tensor) -> torch.Tensor:
    """
    Extract 72D ground truth target from 271D frame.

    Args:
        frame_271d: (B, 271) single frame

    Returns:
        (B, 72) ground truth target
    """
    from utils.train_utils import extract_clean_target

    return extract_clean_target(frame_271d)


def extract_ric_from_72d(flow_output: torch.Tensor) -> torch.Tensor:
    """
    Extract RIC positions from 72D flow output.

    Args:
        flow_output: (B, 72) predicted flow output

    Returns:
        (B, 21, 3) RIC positions for 21 non-root joints
    """
    B = flow_output.shape[0]
    return flow_output[:, 9:72].reshape(B, 21, 3)


def extract_ric_from_271d(frame_271d: torch.Tensor) -> torch.Tensor:
    """
    Extract RIC positions from 271D frame.

    Args:
        frame_271d: (B, 271) single frame

    Returns:
        (B, 21, 3) RIC positions for 21 non-root joints
    """
    B = frame_271d.shape[0]
    # RIC positions for non-root joints are at [6:69]
    return frame_271d[:, 6:69].reshape(B, 21, 3)


def extract_local_velocities_from_271d(frame_271d: torch.Tensor) -> torch.Tensor:
    """
    Extract local velocities from 271D frame.

    Args:
        frame_271d: (B, 271) single frame

    Returns:
        (B, 22, 3) local velocities for all 22 joints
    """
    B = frame_271d.shape[0]
    # Local velocities are at [201:267]
    return frame_271d[:, 201:267].reshape(B, 22, 3)


def extract_root_features_from_72d(
    flow_output: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Extract root features from 72D flow output.

    Args:
        flow_output: (B, 72) predicted flow output

    Returns:
        Dictionary with root_height, root_velocity, root_rotation_6d
    """
    return {
        "root_height": flow_output[:, 0:1],  # (B, 1)
        "root_velocity": flow_output[:, 1:3],  # (B, 2) - X, Z velocity
        "root_rotation_6d": flow_output[:, 3:9],  # (B, 6)
    }


def reconstruct_positions_from_72d(
    flow_output: torch.Tensor,
    prev_root_pos: torch.Tensor,
    prev_root_rot_6d: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct global joint positions from 72D flow output.

    Args:
        flow_output: (B, 72) predicted flow output
        prev_root_pos: (B, 3) previous root position
        prev_root_rot_6d: (B, 6) previous root rotation in 6D

    Returns:
        (B, 22, 3) global joint positions
    """
    from utils.motion_utils import flow_output_to_positions

    return flow_output_to_positions(flow_output, prev_root_pos, prev_root_rot_6d)


def extract_root_state_from_frame(
    frame_271d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract root position and rotation from 271D frame.

    Args:
        frame_271d: (B, 271) single frame

    Returns:
        root_pos: (B, 3) root position [x, y, z]
        root_rot_6d: (B, 6) root rotation in 6D
    """
    # Root height Y is at index 0
    # Root velocity X, Z are at indices 1, 2 (need to accumulate for position)
    # Root rotation 6D is at indices 69:75

    root_height = frame_271d[:, 0:1]  # Y
    root_vel_x = frame_271d[:, 1:2]  # velocity X
    root_vel_z = frame_271d[:, 2:3]  # velocity Z
    root_rot_6d = frame_271d[:, 69:75]  # 6D rotation

    # For a single frame, position X and Z are the velocities (will accumulate)
    # But for ground truth comparison, we need actual positions
    # The root position should come from cumulative sum of velocities
    root_pos = torch.cat([root_vel_x, root_height, root_vel_z], dim=-1)

    return root_pos, root_rot_6d


def compute_per_joint_error(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute per-joint position errors.

    Args:
        predicted: (B, 22, 3) predicted positions
        ground_truth: (B, 22, 3) ground truth positions

    Returns:
        Dictionary with error statistics
    """
    diff = predicted - ground_truth  # (B, 22, 3)
    per_joint_mse = torch.mean(diff**2, dim=(0, 2))  # (22,)
    per_joint_rmse = torch.sqrt(per_joint_mse)  # (22,)

    return {
        "mean_rmse": per_joint_rmse.mean().item(),
        "max_rmse": per_joint_rmse.max().item(),
        "min_rmse": per_joint_rmse.min().item(),
        "joint_rmse": per_joint_rmse.tolist(),
    }


def test_single_frame_prediction():
    """
    Test single frame prediction: compare 1-step prediction with ground truth.
    """
    print("=" * 70)
    print("Single Frame Prediction Test")
    print("=" * 70)

    device = torch.device("cpu")

    # Load data
    vecs = np.load("sample_data/000070_vec.npy")  # (T, 271)
    joints = np.load("sample_data/000070_joint.npy")  # (T, 22, 3)

    with open("sample_data/000070.txt", "r") as f:
        text = f.readlines()[0].strip().split("#")[0]

    print(f"\n1. Data loaded:")
    print(f"   vecs: {vecs.shape}")
    print(f"   joints: {joints.shape}")
    print(f"   text: '{text}'")

    # Load model
    from src.models import HumanMotionGenerator
    from src.config import Config

    config = Config()
    config.max_text_seq_len = 1

    generator = HumanMotionGenerator.load_from_checkpoint(
        "tests/checkpoints/best.pt", config, device=str(device)
    )
    generator.eval()

    # Use frame 0 as history, predict frame 1
    vecs_t = torch.from_numpy(vecs).float()
    joints_t = torch.from_numpy(joints).float()

    # Setup for single frame prediction
    seed_frame = vecs_t[0:1]  # (1, 271) - frame 0
    gt_next_frame = vecs_t[1:2]  # (1, 271) - frame 1 (ground truth)
    gt_next_joints = joints_t[1:2]  # (1, 22, 3) - ground truth joints for frame 1

    print(f"\n2. Single frame prediction setup:")
    print(f"   Seed frame: frame 0")
    print(f"   Predicting: frame 1")
    print(f"   Ground truth available: Yes")

    # Extract ground truth 72D target
    gt_72d = extract_ground_truth_72d(gt_next_frame)  # (1, 72)
    print(f"\n3. Ground truth 72D target extracted: shape {gt_72d.shape}")

    # Get root state from seed frame for position reconstruction
    # For proper reconstruction, we need the cumulative position
    # Frame 0 has velocity, so position starts at origin
    prev_root_pos = torch.zeros((1, 3))
    prev_root_pos[:, 1] = seed_frame[:, 0]  # Y height from frame
    prev_root_rot_6d = seed_frame[:, 69:75]  # 6D rotation

    # Encode text
    from src.utils.text_encoder import CLIPEncoder

    clip_encoder = CLIPEncoder()
    text_embedding = clip_encoder(text)  # (1, 1, 512)

    # Run prediction through the model's internal components
    from src.utils.train_utils import extract_prev_frame_features
    from src.utils.motion_utils import IncrementalFeatureExtractor

    # Extract previous frame features (261D)
    prev_features = extract_prev_frame_features(seed_frame)  # (1, 261)

    # Encode context
    history = seed_frame.unsqueeze(0)  # (1, 1, 271)

    with torch.no_grad():
        context = generator.encoder(
            batch_size=1,
            text=text_embedding,
            input_features=history,
        )[
            :, -1, :, :
        ]  # (1, 22, out_dim)

    # Flow matching prediction
    num_steps = 10
    x_t = torch.randn((1, 72), device=device)
    dt = 1.0 / num_steps

    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full((1,), step * dt, device=device)

            v_cond = generator.predictor(
                history_features=context,
                noise_level=t,
                noisy_target=x_t,
                prev_frame_features=prev_features,
            )

            # No CFG for simplicity
            x_t = x_t + v_cond * dt

    predicted_72d = x_t

    print(f"\n4. Prediction complete:")
    print(f"   Predicted 72D shape: {predicted_72d.shape}")

    # Compare 72D outputs
    mse_72d = torch.mean((predicted_72d - gt_72d) ** 2).item()
    mae_72d = torch.mean(torch.abs(predicted_72d - gt_72d)).item()

    print(f"\n5. 72D Comparison:")
    print(f"   MSE: {mse_72d:.6f}")
    print(f"   MAE: {mae_72d:.6f}")

    # Reconstruct positions from predicted 72D
    predicted_positions = reconstruct_positions_from_72d(
        predicted_72d, prev_root_pos, prev_root_rot_6d
    )

    # Reconstruct positions from ground truth 72D (for fair comparison)
    gt_positions_from_72d = reconstruct_positions_from_72d(
        gt_72d, prev_root_pos, prev_root_rot_6d
    )

    # Compare with ground truth joints
    mse_pred_vs_gt = torch.mean((predicted_positions - gt_next_joints) ** 2).item()
    mse_gt72d_vs_gt = torch.mean((gt_positions_from_72d - gt_next_joints) ** 2).item()

    print(f"\n6. Position Reconstruction Comparison:")
    print(f"   Predicted vs GT joints MSE: {mse_pred_vs_gt:.6f}")
    print(f"   GT 72D vs GT joints MSE: {mse_gt72d_vs_gt:.6f}")
    print(f"   (GT 72D vs GT joints should be ~0 if reconstruction is correct)")

    # Per-joint error
    per_joint = compute_per_joint_error(predicted_positions, gt_next_joints)
    print(f"\n7. Per-Joint Error:")
    print(f"   Mean RMSE: {per_joint['mean_rmse']:.4f}")
    print(f"   Max RMSE: {per_joint['max_rmse']:.4f}")
    print(f"   Min RMSE: {per_joint['min_rmse']:.4f}")

    print("\n" + "=" * 70)
    print("[DONE] Single Frame Prediction Test")
    print("=" * 70)

    return {
        "mse_72d": mse_72d,
        "mae_72d": mae_72d,
        "mse_positions": mse_pred_vs_gt,
        "per_joint_error": per_joint,
    }


def test_short_horizon_prediction():
    """
    Test short horizon prediction: track error for 1-10 frame predictions.
    Uses generate_sequence for proper autoregressive rollout.
    Also tracks velocity magnitude errors.
    """
    print("\n" + "=" * 70)
    print("Short Horizon Prediction Test (1-10 frames)")
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
    from src.utils.train_utils import extract_prev_frame_features, extract_clean_target
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
    num_seed_frames = 5  # Use 5 frames as seed
    num_predict_frames = 10  # Predict next 10 frames

    print(f"\n1. Test setup:")
    print(f"   Seed frames: {num_seed_frames}")
    print(f"   Predict frames: {num_predict_frames}")
    print(f"   Total frames needed: {num_seed_frames + num_predict_frames}")
    print(f"   Available frames: {vecs_t.shape[0]}")

    # Run autoregressive prediction with velocity tracking
    seed_features = vecs_t[:num_seed_frames].unsqueeze(0)  # (1, N, 271)

    print(f"\n2. Running autoregressive prediction with velocity tracking...")

    # Initialize for autoregressive rollout
    history = seed_features
    current_frame_271d = seed_features[0, -1, :]  # (271,)

    init_joints = features_to_positions(history[0])
    extractor = IncrementalFeatureExtractor(dataset_type="t2m", device=device)
    extractor.initialize(init_joints[-1:])

    # Track errors per frame
    errors_pos = []
    root_errors = []
    velocity_mag_errors = []
    velocity_dir_errors = []
    local_vel_errors = []

    for i in range(num_predict_frames):
        # Predict next frame
        with torch.no_grad():
            context = generator.encoder(
                batch_size=1, text=text_embedding, input_features=history
            )[:, -1, :, :]
            prev_features = extract_prev_frame_features(history[:, -1])

            x_t = torch.randn((1, 72), device=device)
            dt = 1.0 / 10
            for step in range(10):
                t = torch.full((1,), step * dt, device=device)
                v = generator.predictor(
                    history_features=context,
                    noise_level=t,
                    noisy_target=x_t,
                    prev_frame_features=prev_features,
                )
                x_t = x_t + v * dt

        predicted_72d = x_t

        # Get predicted 271D for velocity analysis
        new_frame_271d, _ = extractor.process_flow_output(
            predicted_72d, current_frame_271d.unsqueeze(0)
        )

        # Ground truth
        gt_frame_idx = num_seed_frames + i
        gt_271d = vecs_t[gt_frame_idx : gt_frame_idx + 1]
        gt_joints_frame = joints_t[gt_frame_idx]

        # Reconstruct positions from predicted 72D
        pred_positions = features_to_positions(new_frame_271d)[0]  # (22, 3)

        # Position errors
        mse_pos = torch.mean((pred_positions - gt_joints_frame) ** 2).item()
        root_err = torch.mean((pred_positions[0] - gt_joints_frame[0]) ** 2).item()
        errors_pos.append(mse_pos)
        root_errors.append(root_err)

        # Velocity analysis
        pred_vel = new_frame_271d[0, 201:267].reshape(22, 3)  # (22, 3)
        gt_vel = gt_271d[0, 201:267].reshape(22, 3)  # (22, 3)

        # Velocity magnitude
        pred_vel_mag = torch.norm(pred_vel, dim=-1)  # (22,)
        gt_vel_mag = torch.norm(gt_vel, dim=-1)  # (22,)
        vel_mag_mse = torch.mean((pred_vel_mag - gt_vel_mag) ** 2).item()
        vel_mag_mae = torch.mean(torch.abs(pred_vel_mag - gt_vel_mag)).item()
        velocity_mag_errors.append({"mse": vel_mag_mse, "mae": vel_mag_mae})

        # Velocity direction (cosine similarity)
        pred_vel_norm = pred_vel / (pred_vel_mag.unsqueeze(-1) + 1e-8)
        gt_vel_norm = gt_vel / (gt_vel_mag.unsqueeze(-1) + 1e-8)
        cosine_sim = torch.sum(pred_vel_norm * gt_vel_norm, dim=-1)  # (22,)
        vel_dir_mse = torch.mean((1 - cosine_sim) ** 2).item()
        vel_dir_mae = torch.mean(torch.abs(1 - cosine_sim)).item()
        velocity_dir_errors.append({"mse": vel_dir_mse, "mae": vel_dir_mae})

        # Local velocity MSE
        local_vel_mse = torch.mean((pred_vel - gt_vel) ** 2).item()
        local_vel_errors.append(local_vel_mse)

        # Update state for next frame
        current_frame_271d = new_frame_271d[0]
        history = new_frame_271d.unsqueeze(0)

        print(
            f"   Frame {i + 1}: Pos={mse_pos:.4f}, VelMag={vel_mag_mae:.4f}, VelDir={vel_dir_mae:.4f}"
        )

    # Analysis
    print(f"\n3. Position Error Accumulation:")
    print(f"   Frame | Pos MSE | Root MSE")
    print(f"   ------|---------|--------")
    for i in range(num_predict_frames):
        print(f"   {i + 1:5d} | {errors_pos[i]:7.4f} | {root_errors[i]:7.4f}")

    # Velocity magnitude analysis
    print(f"\n4. Velocity Magnitude Errors:")
    print(f"   Frame | Vel Mag MSE | Vel Mag MAE | Vel Dir MSE | Vel Dir MAE")
    print(f"   ------|-------------|-------------|-------------|------------")
    for i in range(num_predict_frames):
        vm = velocity_mag_errors[i]
        vd = velocity_dir_errors[i]
        print(
            f"   {i + 1:5d} | {vm['mse']:11.4f} | {vm['mae']:11.4f} | {vd['mse']:11.4f} | {vd['mae']:10.4f}"
        )

    # Velocity magnitude statistics
    print(f"\n5. Velocity Magnitude Statistics:")
    avg_mag_mse = np.mean([e["mse"] for e in velocity_mag_errors])
    avg_mag_mae = np.mean([e["mae"] for e in velocity_mag_errors])
    avg_dir_mse = np.mean([e["mse"] for e in velocity_dir_errors])
    avg_dir_mae = np.mean([e["mae"] for e in velocity_dir_errors])
    print(f"   Mean Velocity Magnitude MSE: {avg_mag_mse:.4f}")
    print(f"   Mean Velocity Magnitude MAE: {avg_mag_mae:.4f}")
    print(f"   Mean Velocity Direction MSE: {avg_dir_mse:.4f}")
    print(f"   Mean Velocity Direction MAE: {avg_dir_mae:.4f}")

    # Error growth rate
    if len(errors_pos) > 1:
        growth_rates = []
        for i in range(1, len(errors_pos)):
            if errors_pos[i - 1] > 0:
                rate = (errors_pos[i] - errors_pos[i - 1]) / errors_pos[i - 1] * 100
                growth_rates.append(rate)

        avg_growth = np.mean(growth_rates) if growth_rates else 0
        print(f"\n6. Error Growth Rate:")
        print(f"   Average position error increase per frame: {avg_growth:.2f}%")

    print("\n" + "=" * 70)
    print("[DONE] Short Horizon Prediction Test")
    print("=" * 70)

    return {
        "errors_pos": errors_pos,
        "root_errors": root_errors,
        "velocity_mag_errors": velocity_mag_errors,
        "velocity_dir_errors": velocity_dir_errors,
        "local_vel_errors": local_vel_errors,
    }


def test_error_accumulation():
    """
    Test error accumulation with different seed frame counts.
    """
    print("\n" + "=" * 70)
    print("Error Accumulation Test (varying seed frames)")
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

    # Test with different seed frame counts
    seed_counts = [1, 3, 5, 10]
    num_predict = 5  # Predict 5 frames for each test

    print(f"\n1. Testing with different seed frame counts:")
    print(f"   Seed counts: {seed_counts}")
    print(f"   Predict frames: {num_predict}")

    results = {}

    for num_seeds in seed_counts:
        print(f"\n2. Testing with {num_seeds} seed frames...")

        # Use generate_sequence for simplicity
        seed_features = vecs_t[:num_seeds].unsqueeze(0)  # (1, N, 271)

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
        gt_joints = joints_t[num_seeds : num_seeds + num_predict]  # (T, 22, 3)
        gen_joints = generated[0]  # (T, 22, 3)

        # Compute errors
        mse_per_frame = []
        for i in range(num_predict):
            mse = torch.mean((gen_joints[i] - gt_joints[i]) ** 2).item()
            mse_per_frame.append(mse)

        results[num_seeds] = {
            "mse_per_frame": mse_per_frame,
            "mean_mse": np.mean(mse_per_frame),
        }

        print(f"   Mean MSE: {results[num_seeds]['mean_mse']:.4f}")
        print(f"   Per-frame MSE: {[f'{m:.4f}' for m in mse_per_frame]}")

    # Summary
    print(f"\n3. Summary:")
    print(f"   Seeds | Mean MSE | Frame 1 | Frame 5")
    print(f"   ------|----------|---------|--------")
    for num_seeds in seed_counts:
        r = results[num_seeds]
        print(
            f"   {num_seeds:5d} | {r['mean_mse']:8.4f} | {r['mse_per_frame'][0]:7.4f} | {r['mse_per_frame'][-1]:7.4f}"
        )

    print("\n" + "=" * 70)
    print("[DONE] Error Accumulation Test")
    print("=" * 70)

    return results


def test_ric_and_velocity_analysis():
    """
    Analyze RIC positions and local velocities separately.

    This test breaks down the 72D flow output into:
    1. RIC positions (63D) - pose correctness
    2. Root features (9D) - root motion

    And analyzes the 271D features for:
    1. RIC positions - local pose
    2. Local velocities - motion dynamics
    """
    print("\n" + "=" * 70)
    print("RIC Position and Local Velocity Analysis")
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
    num_seed_frames = 5
    num_predict_frames = 10

    print(f"\n1. Test setup:")
    print(f"   Seed frames: {num_seed_frames}")
    print(f"   Predict frames: {num_predict_frames}")

    # Run single frame prediction for detailed analysis
    from src.utils.text_encoder import CLIPEncoder
    from src.utils.train_utils import extract_prev_frame_features, extract_clean_target

    clip_encoder = CLIPEncoder()
    text_embedding = clip_encoder(text)

    seed_features = vecs_t[:num_seed_frames].unsqueeze(0)  # (1, N, 271)

    # Track errors
    ric_errors = []
    root_height_errors = []
    root_vel_errors = []
    root_rot_errors = []
    local_vel_errors = []

    print(f"\n2. Running prediction and analyzing components...")

    with torch.no_grad():
        # Initialize
        history = seed_features
        current_frame_271d = seed_features[0, -1, :]  # (271,)

        for frame_idx in range(num_predict_frames):
            # Ground truth
            gt_frame_idx = num_seed_frames + frame_idx
            gt_frame_271d = vecs_t[gt_frame_idx : gt_frame_idx + 1]  # (1, 271)
            gt_72d = extract_clean_target(gt_frame_271d)  # (1, 72)

            # Encode context
            context = generator.encoder(
                batch_size=1,
                text=text_embedding,
                input_features=history,
            )[:, -1, :, :]

            # Extract prev frame features
            prev_features = extract_prev_frame_features(history[:, -1])

            # Flow matching prediction
            num_steps = 10
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

            predicted_72d = x_t

            # === Analyze RIC positions (pose correctness) ===
            pred_ric = extract_ric_from_72d(predicted_72d)  # (1, 21, 3)
            gt_ric = extract_ric_from_271d(gt_frame_271d)  # (1, 21, 3)
            ric_mse = torch.mean((pred_ric - gt_ric) ** 2).item()
            ric_errors.append(ric_mse)

            # === Analyze root features ===
            pred_root = extract_root_features_from_72d(predicted_72d)
            gt_root = extract_root_features_from_72d(gt_72d)

            # Root height error
            root_height_mse = torch.mean(
                (pred_root["root_height"] - gt_root["root_height"]) ** 2
            ).item()
            root_height_errors.append(root_height_mse)

            # Root velocity error
            root_vel_mse = torch.mean(
                (pred_root["root_velocity"] - gt_root["root_velocity"]) ** 2
            ).item()
            root_vel_errors.append(root_vel_mse)

            # Root rotation error
            root_rot_mse = torch.mean(
                (pred_root["root_rotation_6d"] - gt_root["root_rotation_6d"]) ** 2
            ).item()
            root_rot_errors.append(root_rot_mse)

            # === Analyze local velocities from 271D ===
            # Get predicted 271D from extractor
            from src.utils.motion_utils import (
                IncrementalFeatureExtractor,
                features_to_positions,
            )

            # Initialize extractor for position reconstruction
            init_joints = features_to_positions(history[0])  # (N, 22, 3)
            extractor = IncrementalFeatureExtractor(dataset_type="t2m", device=device)
            extractor.initialize(init_joints[-1:])

            new_frame_271d, _ = extractor.process_flow_output(
                predicted_72d, current_frame_271d.unsqueeze(0)
            )

            # Extract local velocities
            pred_local_vel = extract_local_velocities_from_271d(
                new_frame_271d
            )  # (1, 22, 3)
            gt_local_vel = extract_local_velocities_from_271d(
                gt_frame_271d
            )  # (1, 22, 3)
            local_vel_mse = torch.mean((pred_local_vel - gt_local_vel) ** 2).item()
            local_vel_errors.append(local_vel_mse)

            # Update state
            current_frame_271d = new_frame_271d[0]
            history = new_frame_271d.unsqueeze(0)  # (1, 1, 271)

            print(
                f"   Frame {frame_idx + 1}: RIC MSE={ric_mse:.4f}, "
                f"RootVel MSE={root_vel_mse:.4f}, LocalVel MSE={local_vel_mse:.4f}"
            )

    # Analysis
    print(f"\n3. RIC Position (Pose) Errors:")
    print(f"   Frame | RIC MSE | Height MSE | Rot MSE")
    print(f"   ------|---------|------------|--------")
    for i in range(num_predict_frames):
        print(
            f"   {i + 1:5d} | {ric_errors[i]:7.4f} | {root_height_errors[i]:10.4f} | {root_rot_errors[i]:7.4f}"
        )

    print(f"\n4. Motion Dynamics Errors:")
    print(f"   Frame | Root Vel MSE | Local Vel MSE")
    print(f"   ------|--------------|---------------")
    for i in range(num_predict_frames):
        print(
            f"   {i + 1:5d} | {root_vel_errors[i]:12.4f} | {local_vel_errors[i]:14.4f}"
        )

    # Summary statistics
    print(f"\n5. Summary Statistics:")
    print(f"   RIC Position (Pose):")
    print(f"     Mean MSE: {np.mean(ric_errors):.4f}")
    print(f"     First frame: {ric_errors[0]:.4f}")
    print(f"     Last frame: {ric_errors[-1]:.4f}")
    print(f"     Growth: {ric_errors[-1] / ric_errors[0]:.2f}x")

    print(f"   Root Velocity (Motion):")
    print(f"     Mean MSE: {np.mean(root_vel_errors):.4f}")
    print(f"     First frame: {root_vel_errors[0]:.4f}")
    print(f"     Last frame: {root_vel_errors[-1]:.4f}")

    print(f"   Local Velocity (Motion):")
    print(f"     Mean MSE: {np.mean(local_vel_errors):.4f}")
    print(f"     First frame: {local_vel_errors[0]:.4f}")
    print(f"     Last frame: {local_vel_errors[-1]:.4f}")

    # Interpretation
    print(f"\n6. Interpretation:")
    if np.mean(ric_errors) < 0.1:
        print(f"   [GOOD] RIC positions are accurate - pose is correct")
    elif np.mean(ric_errors) < 1.0:
        print(f"   [OK] RIC positions have moderate error - pose is reasonable")
    else:
        print(f"   [WARN] RIC positions have high error - pose may be incorrect")

    if np.mean(local_vel_errors) < np.mean(ric_errors):
        print(
            f"   [INFO] Local velocities more accurate than RIC - motion dynamics learned better"
        )
    else:
        print(
            f"   [INFO] RIC more accurate than velocities - pose learned better than motion"
        )

    print("\n" + "=" * 70)
    print("[DONE] RIC Position and Local Velocity Analysis")
    print("=" * 70)

    return {
        "ric_errors": ric_errors,
        "root_height_errors": root_height_errors,
        "root_vel_errors": root_vel_errors,
        "root_rot_errors": root_rot_errors,
        "local_vel_errors": local_vel_errors,
    }


def compute_all_component_errors(
    predicted_72d: torch.Tensor,
    gt_72d: torch.Tensor,
    predicted_271d: torch.Tensor = None,
    gt_271d: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute all component-wise errors between predicted and ground truth.

    Args:
        predicted_72d: (B, 72) predicted flow output
        gt_72d: (B, 72) ground truth flow output
        predicted_271d: (B, 271) predicted 271D features (optional)
        gt_271d: (B, 271) ground truth 271D features (optional)

    Returns:
        Dictionary with MSE for each component:
        - root_height_mse: Root height Y error
        - root_vel_mse: Root velocity X, Z error
        - root_rot_mse: Root rotation 6D error
        - root_total_mse: All root features (9D)
        - ric_mse: RIC positions (63D)
        - velocity_mse: Local velocities (66D, if 271D provided)
    """
    errors = {}

    # Root height (1D)
    errors["root_height_mse"] = torch.mean(
        (predicted_72d[:, 0:1] - gt_72d[:, 0:1]) ** 2
    ).item()

    # Root velocity (2D)
    errors["root_vel_mse"] = torch.mean(
        (predicted_72d[:, 1:3] - gt_72d[:, 1:3]) ** 2
    ).item()

    # Root rotation (6D)
    errors["root_rot_mse"] = torch.mean(
        (predicted_72d[:, 3:9] - gt_72d[:, 3:9]) ** 2
    ).item()

    # Root total (9D)
    errors["root_total_mse"] = torch.mean(
        (predicted_72d[:, 0:9] - gt_72d[:, 0:9]) ** 2
    ).item()

    # RIC positions (63D)
    errors["ric_mse"] = torch.mean(
        (predicted_72d[:, 9:72] - gt_72d[:, 9:72]) ** 2
    ).item()

    # Per-joint RIC error
    pred_ric = predicted_72d[:, 9:72].reshape(-1, 21, 3)
    gt_ric = gt_72d[:, 9:72].reshape(-1, 21, 3)
    errors["ric_per_joint_mse"] = torch.mean((pred_ric - gt_ric) ** 2).item()

    # Local velocities from 271D (if provided)
    if predicted_271d is not None and gt_271d is not None:
        pred_vel = predicted_271d[:, 201:267].reshape(-1, 22, 3)
        gt_vel = gt_271d[:, 201:267].reshape(-1, 22, 3)
        errors["velocity_mse"] = torch.mean((pred_vel - gt_vel) ** 2).item()
        errors["velocity_per_joint_mse"] = torch.mean((pred_vel - gt_vel) ** 2).item()

        # Velocity magnitude errors
        # Compute magnitude (L2 norm) for each joint's velocity vector
        pred_vel_mag = torch.norm(pred_vel, dim=-1)  # (B, 22)
        gt_vel_mag = torch.norm(gt_vel, dim=-1)  # (B, 22)
        errors["velocity_mag_mse"] = torch.mean((pred_vel_mag - gt_vel_mag) ** 2).item()
        errors["velocity_mag_mae"] = torch.mean(
            torch.abs(pred_vel_mag - gt_vel_mag)
        ).item()

        # Per-joint velocity magnitude error
        errors["velocity_mag_per_joint_mse"] = torch.mean(
            (pred_vel_mag - gt_vel_mag) ** 2
        ).item()

        # Velocity direction errors (cosine similarity)
        # Normalize velocity vectors
        pred_vel_norm = pred_vel / (pred_vel_mag.unsqueeze(-1) + 1e-8)
        gt_vel_norm = gt_vel / (gt_vel_mag.unsqueeze(-1) + 1e-8)
        # Cosine similarity: dot product of normalized vectors
        cosine_sim = torch.sum(pred_vel_norm * gt_vel_norm, dim=-1)  # (B, 22)
        errors["velocity_dir_mse"] = torch.mean((1 - cosine_sim) ** 2).item()
        errors["velocity_dir_mae"] = torch.mean(torch.abs(1 - cosine_sim)).item()

    return errors


def test_component_errors_detailed():
    """
    Detailed component-wise error analysis.

    Tests three scenarios:
    1. Single frame prediction with varying GT history lengths
    2. Teacher forcing vs autoregressive rollout comparison
    3. Long horizon rollout with component tracking

    Reports errors for:
    - Root height (1D)
    - Root velocity (2D)
    - Root rotation (6D)
    - RIC positions (63D)
    - Local velocities (66D)
    """
    print("\n" + "=" * 80)
    print("DETAILED COMPONENT-WISE ERROR ANALYSIS")
    print("=" * 80)

    device = torch.device("cpu")

    # Load data
    vecs = np.load("sample_data/000070_vec.npy")
    with open("sample_data/000070.txt", "r") as f:
        text = f.readlines()[0].strip().split("#")[0]

    vecs_t = torch.from_numpy(vecs).float()

    # Load model
    from src.models import HumanMotionGenerator
    from src.config import Config
    from src.utils.text_encoder import CLIPEncoder
    from src.utils.train_utils import extract_prev_frame_features, extract_clean_target
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

    # =========================================================================
    # SCENARIO 1: Single Frame Prediction with Varying GT History
    # =========================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 1: Single Frame Prediction with Ground Truth History")
    print("-" * 80)

    seed_counts = [1, 3, 5, 10, 20]
    scenario1_results = {}

    print(f"\n   Seeds | Root H | Root V | Root R | RIC    | Total 72D")
    print(f"   ------|--------|--------|--------|--------|----------")

    for num_seeds in seed_counts:
        if num_seeds >= len(vecs_t) - 1:
            continue

        # Predict one frame with GT history
        history = vecs_t[:num_seeds].unsqueeze(0)  # (1, N, 271)

        with torch.no_grad():
            context = generator.encoder(
                batch_size=1, text=text_embedding, input_features=history
            )[:, -1, :, :]
            prev_features = extract_prev_frame_features(history[:, -1])

            x_t = torch.randn((1, 72), device=device)
            dt = 1.0 / 10
            for step in range(10):
                t = torch.full((1,), step * dt, device=device)
                v = generator.predictor(
                    history_features=context,
                    noise_level=t,
                    noisy_target=x_t,
                    prev_frame_features=prev_features,
                )
                x_t = x_t + v * dt

        predicted_72d = x_t
        gt_271d = vecs_t[num_seeds : num_seeds + 1]
        gt_72d = extract_clean_target(gt_271d)

        errors = compute_all_component_errors(predicted_72d, gt_72d)
        errors["total_72d"] = torch.mean((predicted_72d - gt_72d) ** 2).item()
        scenario1_results[num_seeds] = errors

        print(
            f"   {num_seeds:5d} | {errors['root_height_mse']:6.4f} | "
            f"{errors['root_vel_mse']:6.4f} | {errors['root_rot_mse']:6.4f} | "
            f"{errors['ric_mse']:6.4f} | {errors['total_72d']:8.4f}"
        )

    # =========================================================================
    # SCENARIO 2: Teacher Forcing vs Autoregressive Rollout
    # =========================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 2: Teacher Forcing vs Autoregressive Rollout")
    print("-" * 80)

    num_seeds = 5
    num_predict = 10

    print(f"\n   Setup: {num_seeds} seed frames, {num_predict} predictions")

    # --- Teacher Forcing ---
    tf_errors = []
    for i in range(num_predict):
        history = vecs_t[: num_seeds + i].unsqueeze(0)

        with torch.no_grad():
            context = generator.encoder(
                batch_size=1, text=text_embedding, input_features=history
            )[:, -1, :, :]
            prev_features = extract_prev_frame_features(history[:, -1])

            x_t = torch.randn((1, 72), device=device)
            dt = 1.0 / 10
            for step in range(10):
                t = torch.full((1,), step * dt, device=device)
                v = generator.predictor(
                    history_features=context,
                    noise_level=t,
                    noisy_target=x_t,
                    prev_frame_features=prev_features,
                )
                x_t = x_t + v * dt

        gt_271d = vecs_t[num_seeds + i : num_seeds + i + 1]
        gt_72d = extract_clean_target(gt_271d)
        errors = compute_all_component_errors(x_t, gt_72d)
        tf_errors.append(errors)

    # --- Autoregressive ---
    ar_errors = []
    history = vecs_t[:num_seeds].unsqueeze(0)
    current_frame_271d = vecs_t[num_seeds - 1]

    init_joints = features_to_positions(history[0])
    extractor = IncrementalFeatureExtractor(dataset_type="t2m", device=device)
    extractor.initialize(init_joints[-1:])

    for i in range(num_predict):
        with torch.no_grad():
            context = generator.encoder(
                batch_size=1, text=text_embedding, input_features=history
            )[:, -1, :, :]
            prev_features = extract_prev_frame_features(history[:, -1])

            x_t = torch.randn((1, 72), device=device)
            dt = 1.0 / 10
            for step in range(10):
                t = torch.full((1,), step * dt, device=device)
                v = generator.predictor(
                    history_features=context,
                    noise_level=t,
                    noisy_target=x_t,
                    prev_frame_features=prev_features,
                )
                x_t = x_t + v * dt

        predicted_72d = x_t
        gt_271d = vecs_t[num_seeds + i : num_seeds + i + 1]
        gt_72d = extract_clean_target(gt_271d)

        # Get predicted 271D for velocity comparison
        new_frame_271d, _ = extractor.process_flow_output(
            predicted_72d, current_frame_271d.unsqueeze(0)
        )

        errors = compute_all_component_errors(
            predicted_72d, gt_72d, new_frame_271d, gt_271d
        )
        ar_errors.append(errors)

        current_frame_271d = new_frame_271d[0]
        history = new_frame_271d.unsqueeze(0)

    # Print comparison table
    print(
        f"\n   Frame | Root Height    | Root Velocity   | Root Rotation   | RIC Positions    | Local Velocities"
    )
    print(
        f"         | TF    | AR     | TF    | AR      | TF    | AR      | TF    | AR       | TF    | AR"
    )
    print(
        f"   ------|-------|--------|-------|---------|-------|---------|-------|----------|-------|----------"
    )

    for i in range(num_predict):
        tf = tf_errors[i]
        ar = ar_errors[i]
        print(
            f"   {i + 1:5d} | {tf['root_height_mse']:5.3f} | {ar['root_height_mse']:6.3f} | "
            f"{tf['root_vel_mse']:5.3f} | {ar['root_vel_mse']:7.3f} | "
            f"{tf['root_rot_mse']:5.3f} | {ar['root_rot_mse']:7.3f} | "
            f"{tf['ric_mse']:5.3f} | {ar['ric_mse']:8.3f} | "
            f"{tf.get('velocity_mse', 0):5.3f} | {ar.get('velocity_mse', 0):8.3f}"
        )

    # Summary statistics
    print(f"\n   Summary Statistics:")
    print(f"   Component        | TF Mean | AR Mean | AR/TF Ratio")
    print(f"   -----------------|---------|---------|------------")

    components = [
        ("Root Height", "root_height_mse"),
        ("Root Velocity", "root_vel_mse"),
        ("Root Rotation", "root_rot_mse"),
        ("RIC Positions", "ric_mse"),
        ("Local Velocities", "velocity_mse"),
    ]

    for name, key in components:
        tf_mean = np.mean([e.get(key, 0) for e in tf_errors])
        ar_mean = np.mean([e.get(key, 0) for e in ar_errors])
        ratio = ar_mean / tf_mean if tf_mean > 0 else float("inf")
        print(f"   {name:16s} | {tf_mean:7.4f} | {ar_mean:7.4f} | {ratio:10.1f}x")

    # Velocity magnitude and direction statistics
    print(f"\n   Velocity Magnitude & Direction Statistics:")
    print(f"   Metric            | TF Mean | AR Mean | AR/TF Ratio")
    print(f"   ------------------|---------|---------|------------")

    vel_components = [
        ("Vel Magnitude MSE", "velocity_mag_mse"),
        ("Vel Magnitude MAE", "velocity_mag_mae"),
        ("Vel Direction MSE", "velocity_dir_mse"),
        ("Vel Direction MAE", "velocity_dir_mae"),
    ]

    for name, key in vel_components:
        tf_mean = np.mean([e.get(key, 0) for e in tf_errors])
        ar_mean = np.mean([e.get(key, 0) for e in ar_errors])
        ratio = ar_mean / tf_mean if tf_mean > 0 else float("inf")
        print(f"   {name:17s} | {tf_mean:7.4f} | {ar_mean:7.4f} | {ratio:10.1f}x")

    # =========================================================================
    # SCENARIO 3: Long Horizon Rollout with Component Tracking
    # =========================================================================
    print("\n" + "-" * 80)
    print("SCENARIO 3: Long Horizon Rollout (Autoregressive)")
    print("-" * 80)

    horizon_configs = [(5, 10), (5, 20), (10, 10), (10, 20)]
    scenario3_results = {}

    for num_seeds, num_predict in horizon_configs:
        if num_seeds + num_predict > len(vecs_t):
            continue

        key = f"seeds{num_seeds}_predict{num_predict}"
        print(f"\n   Testing {num_seeds} seeds, {num_predict} predictions...")

        history = vecs_t[:num_seeds].unsqueeze(0)
        current_frame_271d = vecs_t[num_seeds - 1]

        init_joints = features_to_positions(history[0])
        extractor = IncrementalFeatureExtractor(dataset_type="t2m", device=device)
        extractor.initialize(init_joints[-1:])

        frame_errors = []

        for i in range(num_predict):
            with torch.no_grad():
                context = generator.encoder(
                    batch_size=1, text=text_embedding, input_features=history
                )[:, -1, :, :]
                prev_features = extract_prev_frame_features(history[:, -1])

                x_t = torch.randn((1, 72), device=device)
                dt = 1.0 / 10
                for step in range(10):
                    t = torch.full((1,), step * dt, device=device)
                    v = generator.predictor(
                        history_features=context,
                        noise_level=t,
                        noisy_target=x_t,
                        prev_frame_features=prev_features,
                    )
                    x_t = x_t + v * dt

            predicted_72d = x_t
            gt_271d = vecs_t[num_seeds + i : num_seeds + i + 1]
            gt_72d = extract_clean_target(gt_271d)

            new_frame_271d, _ = extractor.process_flow_output(
                predicted_72d, current_frame_271d.unsqueeze(0)
            )

            errors = compute_all_component_errors(
                predicted_72d, gt_72d, new_frame_271d, gt_271d
            )
            frame_errors.append(errors)

            current_frame_271d = new_frame_271d[0]
            history = new_frame_271d.unsqueeze(0)

        scenario3_results[key] = frame_errors

        # Print summary for this config
        avg_root_h = np.mean([e["root_height_mse"] for e in frame_errors])
        avg_root_v = np.mean([e["root_vel_mse"] for e in frame_errors])
        avg_root_r = np.mean([e["root_rot_mse"] for e in frame_errors])
        avg_ric = np.mean([e["ric_mse"] for e in frame_errors])
        avg_vel = np.mean([e.get("velocity_mse", 0) for e in frame_errors])

        print(
            f"   Mean: RootH={avg_root_h:.4f}, RootV={avg_root_v:.4f}, "
            f"RootR={avg_root_r:.4f}, RIC={avg_ric:.4f}, Vel={avg_vel:.4f}"
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n1. Single Frame Prediction (GT History):")
    print(f"   Component errors are consistent across different history lengths")
    if 5 in scenario1_results:
        e = scenario1_results[5]
        print(
            f"   5 seeds: RootH={e['root_height_mse']:.4f}, RootV={e['root_vel_mse']:.4f}, "
            f"RootR={e['root_rot_mse']:.4f}, RIC={e['ric_mse']:.4f}"
        )

    print(f"\n2. Teacher Forcing vs Autoregressive:")
    for name, key in components:
        tf_mean = np.mean([e.get(key, 0) for e in tf_errors])
        ar_mean = np.mean([e.get(key, 0) for e in ar_errors])
        ratio = ar_mean / tf_mean if tf_mean > 0 else float("inf")
        print(f"   {name}: TF={tf_mean:.4f}, AR={ar_mean:.4f}, Ratio={ratio:.1f}x")

    print(f"\n   Velocity Magnitude & Direction:")
    for name, key in vel_components:
        tf_mean = np.mean([e.get(key, 0) for e in tf_errors])
        ar_mean = np.mean([e.get(key, 0) for e in ar_errors])
        ratio = ar_mean / tf_mean if tf_mean > 0 else float("inf")
        print(f"   {name}: TF={tf_mean:.4f}, AR={ar_mean:.4f}, Ratio={ratio:.1f}x")

    print(f"\n3. Key Findings:")
    # Find the component with highest AR/TF ratio
    ratios = []
    for name, key in components:
        tf_mean = np.mean([e.get(key, 0) for e in tf_errors])
        ar_mean = np.mean([e.get(key, 0) for e in ar_errors])
        ratio = ar_mean / tf_mean if tf_mean > 0 else float("inf")
        ratios.append((name, ratio))

    ratios.sort(key=lambda x: x[1], reverse=True)
    print(f"   Highest error accumulation: {ratios[0][0]} ({ratios[0][1]:.1f}x)")
    print(f"   Lowest error accumulation: {ratios[-1][0]} ({ratios[-1][1]:.1f}x)")

    print("\n" + "=" * 80)

    return {
        "scenario1": scenario1_results,
        "tf_errors": tf_errors,
        "ar_errors": ar_errors,
        "scenario3": scenario3_results,
    }


if __name__ == "__main__":
    # Run all tests
    print("\n" + "#" * 70)
    print("# FLOW PREDICTOR COMPARISON TEST SUITE")
    print("#" * 70)

    # Test 1: Single frame prediction
    single_frame_results = test_single_frame_prediction()

    # Test 2: Short horizon prediction
    short_horizon_results = test_short_horizon_prediction()

    # Test 3: Error accumulation
    error_accum_results = test_error_accumulation()

    # Test 4: RIC and velocity analysis
    ric_velocity_results = test_ric_and_velocity_analysis()

    # Test 5: Detailed component-wise error analysis
    component_results = test_component_errors_detailed()

    # Final summary
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)
    print(f"\nSingle Frame Prediction:")
    print(f"  72D MSE: {single_frame_results['mse_72d']:.4f}")
    print(f"  Position MSE: {single_frame_results['mse_positions']:.4f}")

    print(f"\nShort Horizon (10 frames, 5 seeds):")
    print(f"  Mean Position MSE: {np.mean(short_horizon_results['errors_pos']):.4f}")
    print(f"  First Frame MSE: {short_horizon_results['errors_pos'][0]:.4f}")
    print(f"  Last Frame MSE: {short_horizon_results['errors_pos'][-1]:.4f}")

    print(f"\nError Accumulation:")
    for seeds, result in error_accum_results.items():
        print(f"  {seeds} seeds: Mean MSE = {result['mean_mse']:.4f}")

    print(f"\nRIC Position and Velocity Analysis:")
    print(f"  RIC (Pose) Mean MSE: {np.mean(ric_velocity_results['ric_errors']):.4f}")
    print(
        f"  Root Velocity Mean MSE: {np.mean(ric_velocity_results['root_vel_errors']):.4f}"
    )
    print(
        f"  Local Velocity Mean MSE: {np.mean(ric_velocity_results['local_vel_errors']):.4f}"
    )
