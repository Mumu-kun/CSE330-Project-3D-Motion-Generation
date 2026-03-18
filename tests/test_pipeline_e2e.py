"""
Minimal End-to-End Pipeline Test Script

This script tests the complete training pipeline from src/pipeline.ipynb at minimal scale
to verify all components work correctly together.

Test scope:
- Data loading from sample_data/humanml3d-subset
- Model initialization with minimal dimensions
- Forward pass verification
- Training step execution
- Checkpoint saving

Run from project root:
    python tests/test_pipeline_e2e.py
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from pathlib import Path

# Import pipeline components
from config import Config
from models import MotionHistoryEncoder, FlowMatchingPredictor
from utils.dataset import create_dataloader
from utils.train_utils import train
from utils.motion_utils import FeatureNormalizer


def get_minimal_config():
    """Get minimal configuration for end-to-end testing."""
    config = Config()

    # Use sample_data instead of dataset
    config.dataset_path = Path("./sample_data/humanml3d-subset")

    # Minimal device settings
    config.device = "cpu"  # Use CPU for testing

    # Tiny batch size for fast testing
    config.batch_size = 2
    config.num_epochs = 1

    # Minimal model dimensions for fast testing
    config.encoder_hidden_dim = 64
    config.encoder_num_layers = 1
    config.encoder_per_joint_dim = 32
    config.encoder_text_proj_dim = 32

    config.predictor_model_dim = 32
    config.predictor_num_layers = 1
    config.predictor_per_joint_dim = 32
    config.predictor_time_embed_dim = 32

    # Minimal training settings
    config.learning_rate = 1e-4
    config.gradient_clip = 1.0
    config.ema_decay = 0.999

    # Disable curriculum for simpler testing
    config.curriculum = None
    config.horizon = 10  # Fixed small horizon

    # Minimal validation settings
    config.val_interval = 1
    config.val_batches = 1

    # Single worker to avoid multiprocessing issues
    config.num_workers = 0

    # Short max motion length
    config.max_motion_length = 20

    return config


def test_data_loading(config):
    """Test that data loading works correctly."""
    print("\n" + "=" * 60)
    print("TEST: Data Loading")
    print("=" * 60)

    # Create dataloader
    dataloader, normalizer = create_dataloader(config, split="train", shuffle=True)

    print(f"Dataset path: {config.dataset_path}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {config.batch_size}")

    # Get a sample batch
    sample_batch = next(iter(dataloader))

    print(f"\nSample batch:")
    print(f"  Captions: {len(sample_batch['captions'])} samples")
    print(f"  Motion shape: {sample_batch['motion'].shape}")  # (B, T, 271)
    print(f"  Joints shape: {sample_batch['joints'].shape}")  # (B, T, 22, 3)
    print(f"  Text embeddings shape: {sample_batch['text_clip'].shape}")  # (B, 77, 512)
    print(f"  Lengths shape: {sample_batch['lengths'].shape}")  # (B,)
    print(f"\nSample caption: '{sample_batch['captions'][0]}'")

    # Verify shapes
    assert sample_batch["motion"].shape[0] == config.batch_size, "Batch size mismatch"
    assert (
        sample_batch["motion"].shape[2] == 271
    ), "Motion feature dimension should be 271"
    assert sample_batch["joints"].shape[2] == 22, "Should have 22 joints"
    assert sample_batch["joints"].shape[3] == 3, "Each joint should have 3D coordinates"

    print("\n[OK] Data loading test PASSED")

    return dataloader, normalizer


def test_model_initialization(config, normalizer):
    """Test that models initialize correctly."""
    print("\n" + "=" * 60)
    print("TEST: Model Initialization")
    print("=" * 60)

    device = torch.device(config.device)

    # Initialize Motion History Encoder
    motion_encoder = MotionHistoryEncoder(
        frame_feature_dim=config.encoder_motion_dim,
        text_embedding_dim=config.encoder_text_dim,
        text_proj_dim=config.encoder_text_proj_dim,
        model_dim=config.encoder_hidden_dim,
        per_joint_out_dim=config.encoder_per_joint_dim,
        num_layers=config.encoder_num_layers,
        joint_count=config.encoder_num_joints,
        text_scale=config.encoder_text_scale,
        dropout=config.encoder_dropout,
        normalizer=normalizer,
    ).to(device)

    # Initialize Flow Matching Predictor
    flow_predictor = FlowMatchingPredictor(
        per_joint_dim=config.predictor_per_joint_dim,
        model_dim=config.predictor_model_dim,
        num_layers=config.predictor_num_layers,
        joint_count=config.encoder_num_joints,
        time_embed_dim=config.predictor_time_embed_dim,
        dropout=config.predictor_dropout,
        normalizer=normalizer,
    ).to(device)

    encoder_params = sum(p.numel() for p in motion_encoder.parameters())
    predictor_params = sum(p.numel() for p in flow_predictor.parameters())
    total_params = encoder_params + predictor_params

    print(f"Motion Encoder parameters: {encoder_params:,}")
    print(f"Flow Predictor parameters: {predictor_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Verify models have parameters
    assert encoder_params > 0, "Encoder should have parameters"
    assert predictor_params > 0, "Predictor should have parameters"

    print("\n[OK] Model initialization test PASSED")

    return motion_encoder, flow_predictor


def test_forward_pass(config, motion_encoder, flow_predictor, dataloader):
    """Test that forward pass works correctly."""
    print("\n" + "=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)

    device = torch.device(config.device)

    # Get a sample batch
    batch = next(iter(dataloader))

    # Move batch to device
    motion = batch["motion"].to(device)  # (B, T, 271)
    text_clip = batch["text_clip"].to(device)  # (B, 77, 512) or (B, 1, 512)
    lengths = batch["lengths"].to(device)  # (B,)

    B, T, _ = motion.shape

    # Use first text embedding (average over sequence if needed)
    if text_clip.dim() == 3:
        if text_clip.shape[1] == 77:
            # CLIP sequence embeddings - use mean
            text_emb = text_clip.mean(dim=1)  # (B, 512)
        else:
            text_emb = text_clip.squeeze(1)  # (B, 512)
    else:
        text_emb = text_clip

    # Normalize motion features
    normalizer = motion_encoder.normalizer
    if normalizer is not None:
        motion_norm = normalizer.normalize(motion)
    else:
        motion_norm = motion

    # Test encoder
    motion_encoder.eval()
    with torch.no_grad():
        history_features = motion_encoder(motion_norm, text_emb)

    print(f"History features shape: {history_features.shape}")
    assert history_features.shape == (
        B,
        22,
        config.encoder_per_joint_dim,
    ), f"Expected shape ({B}, 22, {config.encoder_per_joint_dim})"

    # Test predictor with noise
    flow_predictor.eval()
    noise_level = torch.rand(B, device=device)  # Random noise levels

    # Create noisy target (72D)
    noisy_target = torch.randn(B, 72, device=device)

    with torch.no_grad():
        pred_frame = flow_predictor(
            history_features=history_features,
            noise_level=noise_level,
            noisy_target=noisy_target,
        )

    print(f"Predicted frame shape: {pred_frame.shape}")
    assert pred_frame.shape == (B, 72), f"Expected shape ({B}, 72)"

    print("\n[OK] Forward pass test PASSED")

    return True


def test_training_step(config, motion_encoder, flow_predictor, dataloader, normalizer):
    """Test that a training step executes correctly."""
    print("\n" + "=" * 60)
    print("TEST: Training Step")
    print("=" * 60)

    device = torch.device(config.device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        list(motion_encoder.parameters()) + list(flow_predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Set models to train mode
    motion_encoder.train()
    flow_predictor.train()

    # Get a sample batch
    batch = next(iter(dataloader))

    # Move batch to device
    motion = batch["motion"].to(device)  # (B, T, 271)
    text_clip = batch["text_clip"].to(device)
    lengths = batch["lengths"].to(device)

    B, T, _ = motion.shape

    # Use first text embedding
    if text_clip.dim() == 3:
        if text_clip.shape[1] == 77:
            text_emb = text_clip.mean(dim=1)  # (B, 512)
        else:
            text_emb = text_clip.squeeze(1)
    else:
        text_emb = text_clip

    # Normalize motion
    motion_norm = normalizer.normalize(motion)

    # Sample a history window and target frame
    # Use fixed horizon for simplicity
    horizon = min(config.horizon, T)
    hist = motion_norm[:, :horizon, :]  # (B, horizon, 271)
    target = motion_norm[:, horizon, :]  # (B, 271) - next frame

    # Extract 72D target (root + joints)
    root_height_vel = target[:, :3]  # (B, 3)
    root_rot = target[:, 69:75]  # (B, 6)
    root_features = torch.cat([root_height_vel, root_rot], dim=-1)  # (B, 9)
    joint_features = target[:, 6:69]  # (B, 63) - 21 joints
    target_72d = torch.cat([root_features, joint_features], dim=-1)  # (B, 72)

    # Create noisy target for flow matching
    noise = torch.randn_like(target_72d)
    t = torch.rand(B, device=device)  # Noise level in [0, 1]
    noisy_target = t.unsqueeze(-1) * target_72d + (1 - t).unsqueeze(-1) * noise

    # Forward pass through encoder
    history_features = motion_encoder(hist, text_emb)

    # Forward pass through predictor
    pred_velocity = flow_predictor(
        history_features=history_features,
        noise_level=t,
        noisy_target=noisy_target,
    )

    # Compute loss (simple L2 loss on velocity)
    target_velocity = target_72d - noisy_target  # Ground truth velocity
    loss = nn.functional.mse_loss(pred_velocity, target_velocity)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        list(motion_encoder.parameters()) + list(flow_predictor.parameters()),
        config.gradient_clip,
    )

    optimizer.step()

    print(f"Loss: {loss.item():.6f}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Horizon used: {horizon}")

    # Verify loss is computed
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 1e6, "Loss should be reasonable"

    print("\n[OK] Training step test PASSED")

    return True


def test_checkpoint_saving(config, motion_encoder, flow_predictor):
    """Test that checkpoint saving works."""
    print("\n" + "=" * 60)
    print("TEST: Checkpoint Saving")
    print("=" * 60)

    # Create checkpoint directory
    checkpoint_dir = Path("./checkpoints/test_e2e")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save a simple checkpoint
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    torch.save(
        {
            "encoder": motion_encoder.state_dict(),
            "predictor": flow_predictor.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")

    # Verify checkpoint exists
    assert checkpoint_path.exists(), "Checkpoint should exist"

    # Load checkpoint
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "encoder" in loaded, "Checkpoint should contain encoder"
    assert "predictor" in loaded, "Checkpoint should contain predictor"

    print("\n[OK] Checkpoint saving test PASSED")

    # Cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return True


def run_e2e_test():
    """Run the complete end-to-end test."""
    print("=" * 60)
    print("MINIMAL END-TO-END PIPELINE TEST")
    print("=" * 60)
    print("Testing the full pipeline from src/pipeline.ipynb")
    print("with minimal scale to verify all components functional")
    print("=" * 60)

    # Get minimal config
    config = get_minimal_config()

    print(f"\nConfiguration:")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Horizon: {config.horizon}")
    print(f"  Encoder hidden dim: {config.encoder_hidden_dim}")
    print(f"  Predictor model dim: {config.predictor_model_dim}")

    # Run tests
    try:
        # Test 1: Data loading
        dataloader, normalizer = test_data_loading(config)

        # Test 2: Model initialization
        motion_encoder, flow_predictor = test_model_initialization(config, normalizer)

        # Test 3: Forward pass
        test_forward_pass(config, motion_encoder, flow_predictor, dataloader)

        # Test 4: Training step
        test_training_step(
            config, motion_encoder, flow_predictor, dataloader, normalizer
        )

        # Test 5: Checkpoint saving
        test_checkpoint_saving(config, motion_encoder, flow_predictor)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
        print("\nThe pipeline is fully functional at minimal scale!")
        print("\nTo run the full pipeline, use:")
        print("  jupyter notebook src/pipeline.ipynb")

        return True

    except Exception as e:
        print(f"\n[X] TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
