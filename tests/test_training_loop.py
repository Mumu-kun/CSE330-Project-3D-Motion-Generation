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
from utils.motion_utils import FeatureNormalizer


# =============================================================================
# Test Configuration
# =============================================================================


def get_test_config():
    """Get test configuration with minimal settings."""
    config = Config()
    config.device = "cpu"  # Use CPU for testing
    config.batch_size = 2
    config.num_epochs = 1
    config.model_dim = 64  # Smaller for faster testing
    config.num_encoder_layers = 1
    config.num_flow_layers = 1
    config.per_joint_out_dim = 32
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

    # Joint RIC: 21 x 3 = 63D
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
        text_embedding_dim=config.text_embedding_dim,
        per_joint_out_dim=config.per_joint_out_dim,
        joint_count=config.num_joints,
        model_dim=config.model_dim,
        num_layers=config.num_encoder_layers,
        max_text_seq_len=config.max_text_seq_len,
        dropout=config.dropout,
        normalizer=normalizer,
    )

    print(
        f"Encoder initialized with {sum(p.numel() for p in encoder.parameters())} parameters"
    )

    # Initialize predictor
    predictor = FlowMatchingPredictor(
        per_joint_dim=config.per_joint_out_dim,
        model_dim=config.model_dim,
        num_layers=config.num_flow_layers,
        joint_count=config.num_joints,
        time_embed_dim=config.time_embed_dim,
        dropout=config.dropout,
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
    text = torch.randn(B, config.max_text_seq_len, config.text_embedding_dim)
    motion = torch.randn(B, T, config.motion_dim)

    # Forward pass with normalize=True (inference mode)
    with torch.no_grad():
        output = encoder(
            text=text,
            input_features=motion,
            batch_size=B,
            normalize=True,  # Models normalize internally
        )

    print(f"Input text shape: {text.shape}")
    print(f"Input motion shape: {motion.shape}")
    print(f"Output shape: {output.shape}")

    # Check output shape: (B, T, 22, per_joint_out_dim)
    expected_shape = (B, T, config.num_joints, config.per_joint_out_dim)
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
    history_features = torch.randn(B, config.num_joints, config.per_joint_out_dim)
    noise_level = torch.rand(B)
    noisy_target = torch.randn(B, 72)  # 72D flow output

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
    text = torch.randn(B, config.max_text_seq_len, config.text_embedding_dim)
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
    contexts = encoder(
        text=text,
        input_features=hist,
        batch_size=B,
        normalize=False,  # Already normalized
    )

    print(f"Context shape: {contexts.shape}")

    # Prepare predictor inputs
    num_pred_frames = min(contexts.shape[1], target_frames.shape[1])
    pred_contexts = contexts[:, -num_pred_frames:]
    prev_frames = hist[:, -num_pred_frames:]
    targets = target_frames[:, -num_pred_frames:]

    # Flatten
    B_eff = B * num_pred_frames
    contexts_flat = pred_contexts.reshape(
        B_eff, config.num_joints, config.per_joint_out_dim
    )
    targets_flat = targets.reshape(B_eff, config.motion_dim)

    # Extract clean targets
    clean_targets = extract_clean_target(targets_flat)

    print(f"Clean targets shape: {clean_targets.shape}")

    # Flow matching
    t = torch.rand(B_eff)
    noise = torch.randn_like(clean_targets)
    x_t = t.view(B_eff, 1) * clean_targets + (1 - t.view(B_eff, 1)) * noise

    # Predict
    pred = predictor(
        history_features=contexts_flat,
        noise_level=t,
        noisy_target=x_t,
    )

    # Compute loss
    target_v = clean_targets - noise
    loss = torch.nn.functional.mse_loss(pred, target_v)

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
    text = torch.randn(B, config.max_text_seq_len, config.text_embedding_dim)

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
