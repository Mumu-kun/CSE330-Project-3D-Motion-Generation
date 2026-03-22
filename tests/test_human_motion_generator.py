"""
Comprehensive tests for HumanMotionGenerator class.

Tests various input types and configurations to ensure
the generator works correctly after MotionHistoryEncoder changes.
"""

import sys

sys.path.insert(0, "src")

import torch
import numpy as np
from pathlib import Path
from config import Config, FlowMatchingPredictorConfig
from models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator
from utils.motion_utils import FeatureNormalizer


def create_mock_normalizer():
    """Create a mock FeatureNormalizer."""
    mean = np.zeros(271, dtype=np.float32)
    std = np.ones(271, dtype=np.float32)
    return FeatureNormalizer(
        torch.from_numpy(mean).float(), torch.from_numpy(std).float()
    )


def create_generator():
    """Create a HumanMotionGenerator instance for testing."""
    normalizer = create_mock_normalizer()

    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        text_proj_dim=128,
        model_dim=256,
        per_joint_out_dim=64,
        num_layers=2,
        joint_count=22,
        text_scale=1.0,
        dropout=0.1,
        normalizer=normalizer,
    )

    # Initialize FlowMatchingPredictorConfig with new API
    pred_config = FlowMatchingPredictorConfig()
    pred_config.hidden_size = 256
    pred_config.intermediate_size = 512
    pred_config.num_hidden_layers = 2
    pred_config.num_attention_heads = 4
    pred_config.track_dimensionality = 3

    predictor = FlowMatchingPredictor(
        feature_size=64,  # per_joint_out_dim from encoder
        config=pred_config,
        use_relative_shift=True,
    )

    generator = HumanMotionGenerator(encoder, predictor)
    generator.eval()

    return generator


def test_tensor_input_3d():
    """Test with 3D tensor input (batch, seq, features)."""
    print("=" * 60)
    print("Test: 3D Tensor Input")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 512)
    input_features = torch.randn(1, 5, 271)  # 5 history frames

    with torch.no_grad():
        positions = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
            input_features=input_features,
        )

    expected_shape = (1, 5 + 3, 22, 3)  # input + generated
    print(f"Input features: {input_features.shape}")
    print(f"Output shape: {positions.shape}")
    print(f"Expected: {expected_shape}")

    if positions.shape == expected_shape:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_tensor_input_2d():
    """Test with 2D tensor input (batch, features) - single frame."""
    print("=" * 60)
    print("Test: 2D Tensor Input (Single Frame)")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 512)
    input_features = torch.randn(1, 271)  # Single frame

    with torch.no_grad():
        positions = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
            input_features=input_features,
        )

    expected_shape = (1, 1 + 3, 22, 3)  # 1 frame + generated
    print(f"Input features: {input_features.shape}")
    print(f"Output shape: {positions.shape}")
    print(f"Expected: {expected_shape}")

    if positions.shape == expected_shape:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_cold_start():
    """Test without input_features (cold start generation)."""
    print("=" * 60)
    print("Test: Cold Start (No Input Features)")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 512)

    with torch.no_grad():
        positions = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
            input_features=None,
        )

    expected_shape = (1, 1 + 3, 22, 3)  # 1 init frame + generated
    print(f"Output shape: {positions.shape}")
    print(f"Expected: {expected_shape}")

    if positions.shape == expected_shape:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_batch_input():
    """Test with batch of 2 samples."""
    print("=" * 60)
    print("Test: Batch Input (Batch Size 2)")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(2, 512)  # Batch of 2

    with torch.no_grad():
        positions = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
        )

    expected_shape = (2, 1 + 3, 22, 3)  # 1 init frame + generated
    print(f"Output shape: {positions.shape}")
    print(f"Expected: {expected_shape}")

    if positions.shape == expected_shape:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_different_num_frames():
    """Test with different number of frames."""
    print("=" * 60)
    print("Test: Different Number of Frames")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 512)
    input_features = torch.randn(1, 2, 271)

    results = []
    for num_frames in [1, 5, 10]:
        with torch.no_grad():
            positions = generator.generate_sequence(
                text=text_emb,
                num_frames=num_frames,
                num_steps=2,
                input_features=input_features,
            )

        expected_shape = (1, 2 + num_frames, 22, 3)
        passed = positions.shape == expected_shape
        results.append((num_frames, passed))
        print(
            f"  num_frames={num_frames}: {positions.shape} == {expected_shape} -> {'PASS' if passed else 'FAIL'}"
        )

    if all(r[1] for r in results):
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_no_nan():
    """Test that output contains no NaN values."""
    print("=" * 60)
    print("Test: No NaN Values")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 512)
    input_features = torch.randn(1, 5, 271)

    with torch.no_grad():
        positions = generator.generate_sequence(
            text=text_emb,
            num_frames=10,
            num_steps=5,
            input_features=input_features,
        )

    has_nan = torch.isnan(positions).any()
    print(f"Has NaN: {has_nan}")

    if not has_nan:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_model_methods():
    """Test model methods (eval, train, to, parameters)."""
    print("=" * 60)
    print("Test: Model Methods")
    print("=" * 60)

    generator = create_generator()

    # Test eval()
    generator.eval()
    assert not generator.encoder.training
    assert not generator.predictor.training
    print("eval(): OK")

    # Test train()
    generator.train()
    assert generator.encoder.training
    assert generator.predictor.training
    print("train(): OK")

    # Test parameters()
    param_count = sum(p.numel() for p in generator.parameters())
    print(f"parameters(): {param_count} parameters")

    # Test to(device)
    generator.to("cpu")
    print("to(device): OK")

    print("[PASS]")
    return True


def test_checkpoint_loading():
    """Test checkpoint save and load functionality."""
    print("=" * 60)
    print("Test: Checkpoint Save/Load")
    print("=" * 60)

    import tempfile
    import os

    # Create a generator and save checkpoint
    generator = create_generator()

    # Create a config that matches the generator's configuration
    test_config = Config()
    test_config.encoder_num_layers = 2  # Match create_generator()
    test_config.predictor_num_layers = 2  # Match create_generator()

    # Create a temporary checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")

        # Save checkpoint
        checkpoint = {
            "encoder": generator.encoder.state_dict(),
            "predictor": generator.predictor.state_dict(),
            "config": test_config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Load checkpoint with matching config
        loaded_generator = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=test_config,
            device="cpu",
            normalizer=create_mock_normalizer(),
        )
        print("Loaded checkpoint successfully")

        # Verify the loaded model works
        text_emb = torch.randn(1, 512)
        input_features = torch.randn(1, 3, 271)

        with torch.no_grad():
            positions = loaded_generator.generate_sequence(
                text=text_emb,
                num_frames=2,
                num_steps=2,
                input_features=input_features,
            )

        expected_shape = (1, 3 + 2, 22, 3)
        print(f"Output shape: {positions.shape}")
        print(f"Expected: {expected_shape}")

        if positions.shape == expected_shape:
            print("[PASS]")
            return True
        else:
            print("[FAIL]")
            return False


def run_all_tests():
    """Run all tests and return results."""
    results = {}

    tests = [
        ("test_tensor_input_3d", test_tensor_input_3d),
        ("test_tensor_input_2d", test_tensor_input_2d),
        ("test_cold_start", test_cold_start),
        ("test_batch_input", test_batch_input),
        ("test_different_num_frames", test_different_num_frames),
        ("test_no_nan", test_no_nan),
        ("test_model_methods", test_model_methods),
        ("test_checkpoint_loading", test_checkpoint_loading),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results[name] = False
        print()

    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")

    return results


if __name__ == "__main__":
    run_all_tests()
