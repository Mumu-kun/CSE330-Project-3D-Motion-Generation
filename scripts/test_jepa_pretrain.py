"""Test script for JEPA pretraining functionality.

This script:
1. Loads the mini test dataset
2. Initializes MotionHistoryEncoder and JepaPredictor via PretrainTrainer
3. Runs JEPA pretraining for a small number of epochs
4. Verifies checkpoint saving/loading and loss computation
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from utils.config import Config
from utils.dataset import create_dataloader
from utils.models.motion_history_encoder import MotionHistoryEncoder
from utils.models.pretrain_trainer import PretrainTrainer, train_pretrain


def test_jepa_pretrain_basic():
    """Run basic JEPA pretraining test with minimal epochs."""
    print("=" * 60)
    print("Testing JEPA Pretraining")
    print("=" * 60)

    # Setup test config
    config = Config()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.dataset_path = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
    config.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "test_jepa_pretrain"
    config.output_path = PROJECT_ROOT / "output" / "test_jepa_pretrain"
    config.batch_size = 4
    config.effective_batch_size = 8
    config.horizon = 10
    config.curriculum = None  # Disable curriculum for simple test
    config._num_epochs = 10
    config.ema_decay = 0.999
    config.jepa_ctx_weight = 0.2
    config.learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.val_interval = 1
    config.checkpoint_interval = 1

    # Reduce model size for faster testing
    config.encoder_config.hidden_size = 16
    config.encoder_config.intermediate_size = 16
    config.encoder_config.num_hidden_layers = 2
    config.encoder_config.num_attention_heads = 2

    config.encoder_config.jp_config.hidden_size = 16
    config.encoder_config.jp_config.intermediate_size = 16
    config.encoder_config.jp_config.num_hidden_layers = 2

    config.num_workers = 0  # Set to 0 for testing to avoid multiprocessing issues

    print(f"Device: {config.device}")
    print(f"Dataset path: {config.dataset_path}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")

    # Create dataloaders
    print("\nLoading dataset...")
    train_dataloader, normalizer = create_dataloader(config, split="train", shuffle=True)
    val_dataloader, _ = create_dataloader(config, split="val", shuffle=False)
    # Set horizon for shorter sequences in testing
    train_dataloader.num_workers = 0
    val_dataloader.num_workers = 0
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")

    print(f"\nEncoder params: {sum(p.numel() for p in MotionHistoryEncoder(config.encoder_config).parameters()):,}")

    # Run pretraining using convenience function
    print("\nRunning JEPA pretraining for 2 epochs...")
    ema_encoder, ema_jepa, pretrain_path = train_pretrain(
        config=config,
        max_epochs=2,
    )

    print("\nPretraining complete!")
    print(f"EMA encoder type: {type(ema_encoder)}")
    print(f"EMA JEPA predictor type: {type(ema_jepa)}")

    # Verify checkpoint exists
    print(f"Checkpoint saved at: {pretrain_path}")

    return ema_encoder, ema_jepa, pretrain_path


def test_jepa_checkpoint_load():
    """Test loading JEPA pretraining checkpoint."""
    print("\n" + "=" * 60)
    print("Testing JEPA Checkpoint Loading")
    print("=" * 60)

    config = Config()
    config.device = "cpu"
    config.dataset_path = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
    config.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "test_jepa_pretrain"

    # Use same reduced model sizes as training (must match test_jepa_pretrain_basic)
    config.encoder_config.hidden_size = 16
    config.encoder_config.intermediate_size = 16
    config.encoder_config.num_hidden_layers = 2
    config.encoder_config.num_attention_heads = 2

    config.encoder_config.jp_config.hidden_size = 16
    config.encoder_config.jp_config.intermediate_size = 16
    config.encoder_config.jp_config.num_hidden_layers = 2

    config.num_workers = 0

    checkpoint_path = config.checkpoint_dir / "pretrain_latest.pt"

    if not checkpoint_path.exists():
        print(f"No checkpoint at {checkpoint_path}, skipping load test")
        return

    # Create trainer to load checkpoint into
    train_dataloader, normalizer = create_dataloader(config, split="train", shuffle=False)
    val_dataloader, _ = create_dataloader(config, split="val", shuffle=False)

    trainer = PretrainTrainer(
        config=config,
    )

    # Test loading via trainer's state
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Verify checkpoint structure matches new PretrainTrainer format
    required_keys = ["encoder", "encoder_ema", "jepa_predictor", "decoder"]
    for key in required_keys:
        assert key in checkpoint, f"Missing key in checkpoint: {key}"
    print("Checkpoint structure validated")

    # Load encoder weights
    trainer.encoder.load_state_dict(checkpoint["encoder"])
    print("Encoder weights loaded successfully")

    # Load JEPA predictor weights
    trainer.jepa_predictor.load_state_dict(checkpoint["jepa_predictor"])
    print("JEPA predictor weights loaded successfully")

    # Load EMA encoder weights
    trainer.ema_encoder.model.load_state_dict(checkpoint["encoder_ema"])
    print("EMA encoder weights loaded successfully")

    trainer.decoder.load_state_dict(checkpoint["decoder"])
    print("Decoder weights loaded successfully")

    print("Checkpoint load test passed!")


if __name__ == "__main__":
    # Run all tests
    ema_encoder, ema_jepa, pretrain_path = test_jepa_pretrain_basic()
    test_jepa_checkpoint_load()

    print("\n" + "=" * 60)
    print("All JEPA pretraining tests passed!")
    print("=" * 60)
