"""
Comprehensive tests for HumanMotionGenerator class.

Tests various input types and configurations to ensure
the generator works correctly after MotionHistoryEncoder changes.
"""

import sys

sys.path.insert(0, "src")

import torch
import numpy as np
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
    config = Config()

    config.encoder_config.hidden_size = 256
    config.encoder_config.intermediate_size = 512
    config.encoder_config.per_joint_output_dim = 64
    config.encoder_config.num_hidden_layers = 2
    config.encoder_config.num_attention_heads = 4
    config.encoder_config.dropout = 0.1
    config.encoder_config.attention_dropout = 0.1
    config.predictor_config = FlowMatchingPredictorConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        track_dimensionality=3,
    )

    encoder = MotionHistoryEncoder(config.encoder_config)

    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
    )

    generator = HumanMotionGenerator(encoder, predictor, config, normalizer=normalizer)
    generator.eval()

    return generator


def test_tensor_input_3d():
    """Test with 4D tensor input positions (batch, seq, joints, xyz)."""
    print("=" * 60)
    print("Test: 3D Tensor Input")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 1, 512)
    input_positions = torch.randn(1, 5, 22, 3)  # 5 history frames

    with torch.no_grad():
        positions, features, relative_shifts = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
            input_positions=input_positions,
        )

    expected_pos_shape = (1, 5 + 3, 22, 3)  # input + generated
    expected_feat_shape = (1, 5 + 3, 271)
    expected_shift_shape = (1, 5 + 3, 22, 3)
    print(f"Input positions: {input_positions.shape}")
    print(f"Output shape: {positions.shape}")
    print(f"Feature history shape: {features.shape}")
    print(f"Relative shifts shape: {relative_shifts.shape}")
    print(f"Expected positions: {expected_pos_shape}")
    print(f"Expected features: {expected_feat_shape}")
    print(f"Expected shifts: {expected_shift_shape}")

    if (
        positions.shape == expected_pos_shape
        and features.shape == expected_feat_shape
        and relative_shifts.shape == expected_shift_shape
    ):
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_tensor_input_2d():
    """Test with 3D tensor input positions (batch, joints, xyz) - single frame."""
    print("=" * 60)
    print("Test: 2D Tensor Input (Single Frame)")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 1, 512)
    input_positions = torch.randn(1, 22, 3)  # Single frame

    with torch.no_grad():
        positions, _, _ = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
            input_positions=input_positions,
        )

    expected_shape = (1, 1 + 3, 22, 3)  # 1 frame + generated
    print(f"Input positions: {input_positions.shape}")
    print(f"Output shape: {positions.shape}")
    print(f"Expected: {expected_shape}")

    if positions.shape == expected_shape:
        print("[PASS]")
        return True
    else:
        print("[FAIL]")
        return False


def test_cold_start():
    """Test without input_positions (cold-start generation)."""
    print("=" * 60)
    print("Test: Cold Start (No Input Positions)")
    print("=" * 60)

    generator = create_generator()
    text_emb = torch.randn(1, 1, 512)

    with torch.no_grad():
        positions, _, _ = generator.generate_sequence(
            text=text_emb,
            num_frames=3,
            num_steps=2,
            input_positions=None,
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
    text_emb = torch.randn(2, 1, 512)  # Batch of 2

    with torch.no_grad():
        positions, _, _ = generator.generate_sequence(
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
    text_emb = torch.randn(1, 1, 512)
    input_positions = torch.randn(1, 2, 22, 3)

    results = []
    for num_frames in [1, 5, 10]:
        with torch.no_grad():
            positions, _, _ = generator.generate_sequence(
                text=text_emb,
                num_frames=num_frames,
                num_steps=2,
                input_positions=input_positions,
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
    text_emb = torch.randn(1, 1, 512)
    input_positions = torch.randn(1, 5, 22, 3)

    with torch.no_grad():
        positions, features, relative_shifts = generator.generate_sequence(
            text=text_emb,
            num_frames=10,
            num_steps=5,
            input_positions=input_positions,
        )

    has_nan = (
        torch.isnan(positions).any()
        or torch.isnan(features).any()
        or torch.isnan(relative_shifts).any()
    )
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

    # Reuse the generator's config so checkpoint loading reconstructs the same model.
    test_config = generator.config

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
        text_emb = torch.randn(1, 1, 512)
        input_positions = torch.randn(1, 3, 22, 3)

        with torch.no_grad():
            positions, _, _ = loaded_generator.generate_sequence(
                text=text_emb,
                num_frames=2,
                num_steps=2,
                input_positions=input_positions,
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
