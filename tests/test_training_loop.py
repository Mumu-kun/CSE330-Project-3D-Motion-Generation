"""
Test training loop and HumanMotionGenerator for error-free execution.

This test verifies:
1. Feature extraction helpers work correctly
2. Training loop runs without errors
3. HumanMotionGenerator generates valid sequences
4. End-to-end pipeline works

Assumes normalizer will be provided during actual training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator
from utils.train_utils import (
    extract_clean_target,
    EMAModel,
)
from utils.motion_utils import FeatureNormalizer, extract_prev_frame_features


# =============================================================================
# Test Configuration
# =============================================================================


def get_test_config():
    """Get test configuration with minimal settings."""
    config = Config()
    config.device = "cpu"  # Use CPU for testing
    config.batch_size = 2
    config.num_epochs = 1
    config.encoder_hidden_dim = 256  # Smaller for faster testing
    config.encoder_num_layers = 1
    config.predictor_num_layers = 1
    config.encoder_per_joint_dim = 64
    return config


def get_mock_normalizer():
    """Create a mock FeatureNormalizer with sample mean/std."""
    # Load from sample data if available
    mean_path = Path(__file__).parent.parent / "sample_data" / "Mean.npy"
    std_path = Path(__file__).parent.parent / "sample_data" / "Std.npy"

    if mean_path.exists() and std_path.exists():
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        # Create mock mean/std with correct shapes
        mean = np.zeros(271, dtype=np.float32)
        std = np.ones(271, dtype=np.float32)

    return FeatureNormalizer(
        mean=torch.from_numpy(mean).float(), std=torch.from_numpy(std).float()
    )


# =============================================================================
# Test 1: Feature Extraction Helpers
# =============================================================================


def test_extract_clean_target():
    """Test extract_clean_target shape and values."""
    print("=" * 60)
    print("Test 2: extract_clean_target")
    print("=" * 60)

    # Create mock 271D frame
    B = 4
    frame = torch.randn(B, 271)

    # Extract target
    target = extract_clean_target(frame)

    # Check shape
    assert target.shape == (B, 72), f"Expected shape (B, 72), got {target.shape}"
    print(f"Input shape: {frame.shape}")
    print(f"Output shape: {target.shape}")

    # Verify slicing
    # Root: height(1) + vel(2) + rot_6d(6) = 9D
    expected_root = torch.cat(
        [
            frame[:, 0:1],  # height_y
            frame[:, 1:3],  # vel_x, vel_z
            frame[:, 69:75],  # root_rot_6d
        ],
        dim=-1,
    )

    assert torch.allclose(target[:, :9], expected_root), "Root target mismatch!"
    print("Root target: OK")

    # Joint RIC: 21 x 3 = 63D (excluding root joint at index 0)
    expected_joints = frame[:, 6:69]
    assert torch.allclose(target[:, 9:], expected_joints), "Joint RIC mismatch!"
    print("Joint RIC: OK")

    print("[PASS] extract_clean_target works correctly\n")
    return True


# =============================================================================
# Test 2: Model Initialization
# =============================================================================


def test_model_initialization():
    """Test that models can be initialized correctly."""
    print("=" * 60)
    print("Test 3: Model Initialization")
    print("=" * 60)

    config = get_test_config()
    normalizer = get_mock_normalizer()

    # Initialize encoder
    encoder = MotionHistoryEncoder(
        frame_feature_dim=config.motion_dim,
        text_embedding_dim=config.encoder_text_dim,
        text_proj_dim=config.encoder_text_proj_dim,
        per_joint_out_dim=config.encoder_per_joint_dim,
        joint_count=config.num_joints,
        model_dim=config.encoder_hidden_dim,
        num_layers=config.encoder_num_layers,
        text_scale=config.encoder_text_scale,
        dropout=config.encoder_dropout,
        normalizer=normalizer,
    )

    print(
        f"Encoder initialized with {sum(p.numel() for p in encoder.parameters())} parameters"
    )

    # Initialize predictor
    predictor = FlowMatchingPredictor(
        per_joint_dim=config.predictor_per_joint_dim,
        model_dim=config.predictor_model_dim,
        num_layers=config.predictor_num_layers,
        joint_count=config.num_joints,
        time_embed_dim=config.predictor_time_embed_dim,
        dropout=config.predictor_dropout,
        normalizer=normalizer,
    )

    print(
        f"Predictor initialized with {sum(p.numel() for p in predictor.parameters())} parameters"
    )

    # Check normalizer is set
    assert encoder.normalizer is not None, "Encoder normalizer not set!"
    assert predictor.normalizer is not None, "Predictor normalizer not set!"
    print("Normalizers: OK")

    print("[PASS] Model initialization works correctly\n")
    return encoder, predictor, config


# =============================================================================
# Test 3: Encoder Forward Pass
# =============================================================================


def test_encoder_forward(encoder, config):
    """Test encoder forward pass with mock inputs."""
    print("=" * 60)
    print("Test 4: Encoder Forward Pass")
    print("=" * 60)

    B = config.batch_size
    T = 16  # History length
    device = config.device

    # Create mock inputs (RAW features - unnormalized)
    text = torch.randn(B, config.encoder_text_dim)
    motion = torch.randn(B, T, config.motion_dim)

    # Forward pass with normalize=True (inference mode)
    with torch.no_grad():
        output = encoder(
            motion_seq=motion,
            text_emb=text.squeeze(1),  # (B, text_dim)
        )

    print(f"Input text shape: {text.shape}")
    print(f"Input motion shape: {motion.shape}")
    print(f"Output shape: {output.shape}")

    # Check output shape: (B, 22, per_joint_out_dim)
    expected_shape = (B, config.num_joints, config.encoder_per_joint_dim)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"

    # Check no NaN
    assert not torch.isnan(output).any(), "NaN in encoder output!"
    print("No NaN in output: OK")

    print("[PASS] Encoder forward pass works correctly\n")
    return True


# =============================================================================
# Test 4: Predictor Forward Pass
# =============================================================================


def test_predictor_forward(predictor, config):
    """Test predictor forward pass with mock inputs."""
    print("=" * 60)
    print("Test 5: Predictor Forward Pass")
    print("=" * 60)

    B = config.batch_size
    device = config.device

    # Create mock inputs
    history_features = torch.randn(B, config.num_joints, config.encoder_per_joint_dim)
    noise_level = torch.rand(B)
    # 64D noisy target: 1D root height + 21*3D joint RIC = 1 + 63 = 64D
    noisy_target = torch.randn(B, 64)

    # Forward pass
    with torch.no_grad():
        output = predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target=noisy_target,
        )

    print(f"History features shape: {history_features.shape}")
    print(f"Noisy target shape: {noisy_target.shape}")
    print(f"Output shape: {output.shape}")

    # Check output shape: (B, 72)
    assert output.shape == (B, 72), f"Expected (B, 72), got {output.shape}"

    # Check no NaN
    assert not torch.isnan(output).any(), "NaN in predictor output!"
    print("No NaN in output: OK")

    print("[PASS] Predictor forward pass works correctly\n")
    return True


# =============================================================================
# Test 5: Training Iteration
# =============================================================================


def test_training_iteration(encoder, predictor, config):
    """Test a single training iteration."""
    print("=" * 60)
    print("Test 6: Training Iteration")
    print("=" * 60)

    B = config.batch_size
    T = 32  # Total sequence length
    horizon = 16
    device = config.device

    # Create mock batch (RAW features - unnormalized)
    motion = torch.randn(B, T, config.motion_dim)
    text = torch.randn(B, config.encoder_text_dim)  # (B, text_dim)
    lengths = torch.tensor([T, T - 5])  # Variable lengths

    # Get normalizer
    normalizer = encoder.normalizer

    # Normalize features (as done in training loop)
    motion_norm = normalizer.normalize(motion)

    # Sample window
    start_idx = 0
    end_idx = start_idx + horizon

    hist = motion_norm[:, start_idx:end_idx]
    target_frames = motion_norm[:, start_idx + 1 : end_idx + 1]

    print(f"History shape: {hist.shape}")
    print(f"Target frames shape: {target_frames.shape}")

    # Encode context (normalize=False since already normalized)
    # The encoder returns (B, 22, per_joint_dim) - last timestep's features
    contexts = encoder(
        motion_seq=hist,
        text_emb=text,
    )

    print(f"Context shape: {contexts.shape}")

    # For training iteration, we predict ONE frame at a time
    # Get the last target frame as our prediction target
    # target_frames shape: (B, horizon, 271), get last frame: (B, 271)
    target = target_frames[:, -1, :]  # (B, 271)

    # Extract clean target (72D)
    clean_targets = extract_clean_target(target)  # (B, 72)

    print(f"Clean targets shape: {clean_targets.shape}")

    # New flow matching format: 64D noisy target (1D height + 63D joints)
    x1_h = clean_targets[..., 0:1]
    x1_vel = clean_targets[..., 1:3]
    x1_rot = clean_targets[..., 3:9]
    x1_joints = clean_targets[..., 9:]

    # Noise only on height and joints (velocity and rotation are zero)
    x0_h = torch.randn_like(x1_h)
    x0_joints = torch.randn_like(x1_joints)

    t = torch.rand(B)
    t_ = t.view(B, 1)

    # Create 64D noisy target
    xt_h = t_ * x1_h + (1 - t_) * x0_h
    xt_joints = t_ * x1_joints + (1 - t_) * x0_joints
    noisy_target = torch.cat([xt_h, xt_joints], dim=-1)  # (B, 64)

    # Get prev_frame_features
    prev_frame = hist[:, -1]  # (B, 271)
    prev_features = extract_prev_frame_features(prev_frame)  # (B, 261)

    # Predict
    pred = predictor(
        history_features=contexts,
        noise_level=t,
        noisy_target=noisy_target,
        prev_frame_features=prev_features,
    )

    # Compute loss using new format with separate components
    pred_v_h = pred[..., 0:1]
    pred_vel = pred[..., 1:3]
    pred_rot = pred[..., 3:9]
    pred_v_joints = pred[..., 9:]
    pred_v_pos = torch.cat([pred_v_h, pred_v_joints], dim=-1)

    target_v_h = x1_h - x0_h
    target_v_joints = x1_joints - x0_joints
    target_v_pos = torch.cat([target_v_h, target_v_joints], dim=-1)

    L_flow = F.smooth_l1_loss(pred_v_pos, target_v_pos)
    cos_sim = F.cosine_similarity(pred_rot, x1_rot, dim=-1)
    L_dir = (1 - cos_sim).mean()
    L_vel = F.mse_loss(pred_vel, x1_vel)
    L_rot = F.smooth_l1_loss(pred_rot, x1_rot)

    loss = 1.0 * L_flow + 0.5 * L_dir + 1.0 * L_vel + 1.0 * L_rot

    print(f"Loss: {loss.item():.6f}")

    # Check loss is valid
    assert not torch.isnan(loss), "NaN in loss!"
    assert loss.item() >= 0, "Negative loss!"
    print("Loss is valid: OK")

    # Test backward pass
    loss.backward()
    print("Backward pass: OK")

    print("[PASS] Training iteration works correctly\n")
    return True


# =============================================================================
# Test 6: HumanMotionGenerator
# =============================================================================


def test_human_motion_generator(encoder, predictor, config):
    """Test HumanMotionGenerator end-to-end."""
    print("=" * 60)
    print("Test 7: HumanMotionGenerator")
    print("=" * 60)

    # Create generator
    generator = HumanMotionGenerator(encoder, predictor)
    generator.eval()

    B = 1
    num_frames = 5  # Small for testing
    num_steps = 3  # Small for testing

    # Create mock text input (pre-encoded)
    text = torch.randn(B, config.encoder_text_dim)

    # Create mock initial features
    input_features = torch.randn(B, 1, config.motion_dim)

    print(f"Text shape: {text.shape}")
    print(f"Input features shape: {input_features.shape}")
    print(f"Generating {num_frames} frames with {num_steps} ODE steps...")

    # Generate sequence
    with torch.no_grad():
        positions = generator.generate_sequence(
            text=text,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=1.0,
            input_features=input_features,
        )

    print(f"Output positions shape: {positions.shape}")

    # Check output shape: (B, T, 22, 3)
    expected_frames = input_features.shape[1] + num_frames
    expected_shape = (B, expected_frames, config.num_joints, 3)
    assert (
        positions.shape == expected_shape
    ), f"Expected {expected_shape}, got {positions.shape}"

    # Check no NaN
    assert not torch.isnan(positions).any(), "NaN in generated positions!"
    print("No NaN in positions: OK")

    # Note: With random weights, positions may be large - this is expected for untrained models
    # The key check is no NaN values
    max_pos = positions.abs().max().item()
    print(f"Max position value: {max_pos:.2f}")
    print("Positions generated: OK (untrained model may have large values)")

    print("[PASS] HumanMotionGenerator works correctly\n")
    return True


# =============================================================================
# Test 7: EMA Model
# =============================================================================


def test_ema_model(encoder, config):
    """Test EMA model wrapper."""
    print("=" * 60)
    print("Test 8: EMA Model")
    print("=" * 60)

    # Create EMA
    ema = EMAModel(encoder, decay=0.999)

    # Check parameters are copied
    for (name, p), (ema_name, ema_p) in zip(
        encoder.named_parameters(), ema.model.named_parameters()
    ):
        assert torch.allclose(p, ema_p), f"EMA parameter {name} not copied correctly!"

    print("Parameters copied correctly: OK")

    # Test update
    # Modify encoder parameters
    for p in encoder.parameters():
        p.data.add_(0.1)

    # Update EMA
    ema.update(encoder)

    # Check EMA parameters changed
    changed = False
    for p, ema_p in zip(encoder.parameters(), ema.model.parameters()):
        if not torch.allclose(p, ema_p):
            changed = True
            break

    assert changed, "EMA parameters not updated!"
    print("EMA update works: OK")

    print("[PASS] EMA model works correctly\n")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "=" * 60)
    print("RUNNING TRAINING LOOP VERIFICATION TESTS")
    print("=" * 60 + "\n")

    results = {}

    # Test 1: Feature extraction
    try:
        results["extract_clean_target"] = test_extract_clean_target()
    except Exception as e:
        results["extract_clean_target"] = False
        print(f"[FAIL] extract_clean_target: {e}\n")

    # Test 2: Model initialization
    try:
        encoder, predictor, config = test_model_initialization()
        results["model_initialization"] = True
    except Exception as e:
        results["model_initialization"] = False
        print(f"[FAIL] model_initialization: {e}\n")
        return results  # Can't continue without models

    # Test 3: Encoder forward
    try:
        results["encoder_forward"] = test_encoder_forward(encoder, config)
    except Exception as e:
        results["encoder_forward"] = False
        print(f"[FAIL] encoder_forward: {e}\n")

    # Test 4: Predictor forward
    try:
        results["predictor_forward"] = test_predictor_forward(predictor, config)
    except Exception as e:
        results["predictor_forward"] = False
        print(f"[FAIL] predictor_forward: {e}\n")

    # Test 5: Training iteration
    try:
        results["training_iteration"] = test_training_iteration(
            encoder, predictor, config
        )
    except Exception as e:
        results["training_iteration"] = False
        print(f"[FAIL] training_iteration: {e}\n")

    # Test 6: HumanMotionGenerator
    try:
        results["human_motion_generator"] = test_human_motion_generator(
            encoder, predictor, config
        )
    except Exception as e:
        results["human_motion_generator"] = False
        print(f"[FAIL] human_motion_generator: {e}\n")

    # Test 7: EMA model
    try:
        results["ema_model"] = test_ema_model(encoder, config)
    except Exception as e:
        results["ema_model"] = False
        print(f"[FAIL] ema_model: {e}\n")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] All tests passed! Training loop is ready for Kaggle.")
    else:
        print("\n[WARN] Some tests failed. Please fix issues before Kaggle training.")

    return results


if __name__ == "__main__":
    run_all_tests()
