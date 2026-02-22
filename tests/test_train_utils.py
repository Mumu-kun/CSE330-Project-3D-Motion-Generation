"""
Unit tests for training utility functions.

Tests the new training utilities for the progressive horizon curriculum:
- extract_prev_frame_features: 271D -> 261D conversion
- extract_clean_target: 271D -> 72D conversion
- EMAModel: EMA model wrapper
- train: Full training loop
- validate: Validation function
- generate_free_running: Free-running generation

Feature Format Reference:
- 271D Input: [0:3]root, [3:69]RIC, [69:201]rot6d, [201:267]vel, [267:271]foot
- 261D Prev: [0:9]root, [9:261]joints (21 x 12D)
- 72D Target: [0:9]root, [9:72]joint_RIC (21 x 3D)
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.utils.train_utils import (
    extract_prev_frame_features,
    extract_clean_target,
    EMAModel,
    train,
    validate,
    generate_free_running,
)


# =============================================================================
# Test Datasets
# =============================================================================


class MockDataset(Dataset):
    """Mock dataset for testing with pre-encoded text."""

    def __init__(self, num_samples=10, seq_len=100, motion_dim=271):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.motion_dim = motion_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "motion": torch.randn(self.seq_len, self.motion_dim),
            "text_clip": torch.randn(77, 512),  # Pre-encoded CLIP embeddings
            "lengths": torch.tensor(self.seq_len - idx % 10),
        }


class MockDatasetWithCaptions(Dataset):
    """Mock dataset for testing with raw captions."""

    def __init__(self, num_samples=10, seq_len=100, motion_dim=271):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.motion_dim = motion_dim
        self.captions = [
            "a person walks forward",
            "someone jumps up and down",
            "a person dances",
            "someone runs",
            "a person sits down",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "motion": torch.randn(self.seq_len, self.motion_dim),
            "captions": self.captions[idx % len(self.captions)],
            "lengths": torch.tensor(self.seq_len - idx % 10),
        }


# =============================================================================
# Feature Extraction Tests
# =============================================================================


def test_extract_prev_frame_features():
    """Test 271D -> 261D conversion."""
    print("\n=== Test: extract_prev_frame_features ===")

    B = 4
    frame_271d = torch.randn(B, 271)

    prev_features = extract_prev_frame_features(frame_271d)

    # Check output shape
    assert prev_features.shape == (
        B,
        261,
    ), f"Expected (B, 261), got {prev_features.shape}"
    print(f"  Input shape: {frame_271d.shape}")
    print(f"  Output shape: {prev_features.shape} - PASS")

    # Check root portion (first 9D)
    root_features = prev_features[:, :9]
    assert root_features.shape == (
        B,
        9,
    ), f"Root shape: expected (B, 9), got {root_features.shape}"
    print(f"  Root features shape: {root_features.shape} - PASS")

    # Check joint portion (next 252D)
    joint_features = prev_features[:, 9:]
    assert joint_features.shape == (
        B,
        252,
    ), f"Joint shape: expected (B, 252), got {joint_features.shape}"
    print(f"  Joint features shape: {joint_features.shape} - PASS")


def test_extract_clean_target():
    """Test 271D -> 72D conversion."""
    print("\n=== Test: extract_clean_target ===")

    B = 4
    frame_271d = torch.randn(B, 271)

    clean_target = extract_clean_target(frame_271d)

    # Check output shape
    assert clean_target.shape == (B, 72), f"Expected (B, 72), got {clean_target.shape}"
    print(f"  Input shape: {frame_271d.shape}")
    print(f"  Output shape: {clean_target.shape} - PASS")

    # Check root portion (first 9D)
    root_features = clean_target[:, :9]
    assert root_features.shape == (
        B,
        9,
    ), f"Root shape: expected (B, 9), got {root_features.shape}"
    print(f"  Root features shape: {root_features.shape} - PASS")

    # Check joint RIC portion (next 63D)
    joint_ric = clean_target[:, 9:]
    assert joint_ric.shape == (
        B,
        63,
    ), f"Joint RIC shape: expected (B, 63), got {joint_ric.shape}"
    print(f"  Joint RIC shape: {joint_ric.shape} - PASS")


def test_feature_extraction_consistency():
    """Test that feature extraction is consistent across batch."""
    print("\n=== Test: Feature extraction consistency ===")

    B = 2
    # Create identical frames
    frame = torch.randn(1, 271)
    frames = frame.expand(B, -1)

    prev_features = extract_prev_frame_features(frames)
    clean_target = extract_clean_target(frames)

    # All batch elements should be identical
    assert torch.allclose(
        prev_features[0], prev_features[1]
    ), "prev_features not consistent"
    assert torch.allclose(
        clean_target[0], clean_target[1]
    ), "clean_target not consistent"
    print("  Feature extraction is consistent across batch - PASS")


# =============================================================================
# EMAModel Tests
# =============================================================================


def test_ema_model_creation():
    """Test EMAModel creation."""
    print("\n=== Test: EMAModel creation ===")

    model = nn.Linear(10, 10)
    ema = EMAModel(model, decay=0.999)

    # Check that EMA model has same structure
    assert isinstance(ema.model, nn.Linear), "EMA model should be Linear"
    print(f"  EMA model type: {type(ema.model).__name__} - PASS")

    # Check that gradients are disabled
    for p in ema.model.parameters():
        assert not p.requires_grad, "EMA parameters should not require grad"
    print("  EMA gradients disabled - PASS")


def test_ema_model_update():
    """Test EMAModel update."""
    print("\n=== Test: EMAModel update ===")

    model = nn.Linear(10, 10)
    ema = EMAModel(model, decay=0.9)  # Lower decay for visible change

    # Store initial weights
    initial_ema_weight = ema.model.weight.data.clone()
    initial_model_weight = model.weight.data.clone()

    # Modify model weights
    model.weight.data += 1.0

    # Update EMA
    ema.update(model)

    # EMA should have changed
    assert not torch.allclose(
        ema.model.weight.data, initial_ema_weight
    ), "EMA weights should have changed"
    print("  EMA weights updated - PASS")

    # EMA should be closer to initial than model (due to decay)
    ema_diff = torch.abs(ema.model.weight.data - initial_ema_weight).mean()
    model_diff = torch.abs(model.weight.data - initial_model_weight).mean()
    assert ema_diff < model_diff, "EMA should change slower than model"
    print(f"  EMA diff: {ema_diff:.4f}, Model diff: {model_diff:.4f} - PASS")


def test_ema_model_to_device():
    """Test EMAModel device transfer."""
    print("\n=== Test: EMAModel device transfer ===")

    model = nn.Linear(10, 10)
    ema = EMAModel(model, decay=0.999)

    # Test to() method
    ema_cpu = ema.to("cpu")
    assert ema_cpu is ema, "to() should return self"
    print("  EMAModel.to() works - PASS")


# =============================================================================
# Training Loop Tests
# =============================================================================


def test_train_basic():
    """Test basic training loop."""
    print("\n=== Test: Basic training loop ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Import models
    from src.models import MotionHistoryEncoder, FlowMatchingPredictor

    # Initialize models (smaller for testing)
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        per_joint_out_dim=64,
        joint_count=22,
        model_dim=128,
        num_layers=2,
    ).to(device)

    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=64,
        num_layers=2,
        joint_count=22,
    ).to(device)

    # Create dataloader
    dataset = MockDataset(num_samples=4, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Run training
    save_dir = "./test_checkpoints_train"
    encoder_ema, predictor_ema = train(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        num_epochs=1,
        save_dir=save_dir,
        horizon=16,
        device=device,
    )

    # Check that EMA models are returned
    assert isinstance(encoder_ema, EMAModel), "Should return EMAModel for encoder"
    assert isinstance(predictor_ema, EMAModel), "Should return EMAModel for predictor"
    print("  Training completed - PASS")

    # Check checkpoints
    assert os.path.exists(os.path.join(save_dir, "latest.pt")), "latest.pt should exist"
    assert os.path.exists(os.path.join(save_dir, "best.pt")), "best.pt should exist"
    print("  Checkpoints created - PASS")

    # Cleanup
    shutil.rmtree(save_dir, ignore_errors=True)


def test_train_with_clip_encoder():
    """Test training with CLIP encoder for raw captions."""
    print("\n=== Test: Training with CLIP encoder ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models import MotionHistoryEncoder, FlowMatchingPredictor

    # Initialize CLIP encoder (skip if transformers not available)
    try:
        from src.utils.text_encoder import CLIPEncoder

        clip_encoder = CLIPEncoder("openai/clip-vit-base-patch32")
        clip_encoder.to(device)
        clip_encoder.eval()
    except ImportError as e:
        print(f"  Skipping test (transformers not available): {e}")
        return
    except Exception as e:
        print(f"  Skipping test (CLIP initialization failed): {e}")
        return

    # Initialize models
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        per_joint_out_dim=64,
        joint_count=22,
        model_dim=128,
        num_layers=2,
    ).to(device)

    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=64,
        num_layers=2,
        joint_count=22,
    ).to(device)

    # Create dataloader with raw captions
    dataset = MockDatasetWithCaptions(num_samples=2, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Run training
    save_dir = "./test_checkpoints_clip"
    encoder_ema, predictor_ema = train(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        num_epochs=1,
        save_dir=save_dir,
        horizon=16,
        device=device,
        clip_encoder=clip_encoder,
    )

    print("  Training with CLIP encoder completed - PASS")

    # Cleanup
    shutil.rmtree(save_dir, ignore_errors=True)


def test_train_resume():
    """Test training resume from checkpoint."""
    print("\n=== Test: Training resume ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models import MotionHistoryEncoder, FlowMatchingPredictor

    # Initialize models
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        per_joint_out_dim=64,
        joint_count=22,
        model_dim=128,
        num_layers=2,
    ).to(device)

    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=64,
        num_layers=2,
        joint_count=22,
    ).to(device)

    # Create dataloader
    dataset = MockDataset(num_samples=2, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    save_dir = "./test_checkpoints_resume"

    # First training run
    train(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        num_epochs=1,
        save_dir=save_dir,
        horizon=16,
        device=device,
    )

    # Resume from checkpoint
    encoder_ema, predictor_ema = train(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        num_epochs=2,
        save_dir=save_dir,
        horizon=16,
        device=device,
        resume_from=os.path.join(save_dir, "latest.pt"),
    )

    print("  Training resume completed - PASS")

    # Cleanup
    shutil.rmtree(save_dir, ignore_errors=True)


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate():
    """Test validation function."""
    print("\n=== Test: Validation function ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models import MotionHistoryEncoder, FlowMatchingPredictor

    # Initialize models
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        per_joint_out_dim=64,
        joint_count=22,
        model_dim=128,
        num_layers=2,
    ).to(device)

    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=64,
        num_layers=2,
        joint_count=22,
    ).to(device)

    # Create dataloader
    dataset = MockDataset(num_samples=4, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Run validation
    metrics = validate(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        horizon=16,
        device=device,
        num_batches=2,
    )

    assert "val_loss" in metrics, "Should return val_loss"
    assert isinstance(metrics["val_loss"], float), "val_loss should be float"
    print(f"  Validation loss: {metrics['val_loss']:.6f} - PASS")


# =============================================================================
# Free-Running Generation Tests
# =============================================================================


def test_generate_free_running():
    """Test free-running generation."""
    print("\n=== Test: Free-running generation ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models import MotionHistoryEncoder, FlowMatchingPredictor

    # Initialize models
    encoder = MotionHistoryEncoder(
        frame_feature_dim=271,
        text_embedding_dim=512,
        per_joint_out_dim=64,
        joint_count=22,
        model_dim=128,
        num_layers=2,
    ).to(device)

    predictor = FlowMatchingPredictor(
        per_joint_dim=64,
        model_dim=64,
        num_layers=2,
        joint_count=22,
    ).to(device)

    encoder.eval()
    predictor.eval()

    # Generate
    B = 2
    text = torch.randn(B, 77, 512).to(device)

    generated = generate_free_running(
        encoder=encoder,
        predictor=predictor,
        text=text,
        num_frames=10,
        num_flow_steps=5,
        device=device,
    )

    assert generated.shape == (
        B,
        10,
        72,
    ), f"Expected (B, 10, 72), got {generated.shape}"
    print(f"  Generated shape: {generated.shape} - PASS")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING UTILITIES TESTS")
    print("=" * 60)

    # Feature extraction tests
    test_extract_prev_frame_features()
    test_extract_clean_target()
    test_feature_extraction_consistency()

    # EMA tests
    test_ema_model_creation()
    test_ema_model_update()
    test_ema_model_to_device()

    # Training tests
    test_train_basic()
    test_train_with_clip_encoder()
    test_train_resume()

    # Validation tests
    test_validate()

    # Generation tests
    test_generate_free_running()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
