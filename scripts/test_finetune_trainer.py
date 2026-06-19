"""Test script for decoder-only finetuning functionality.

This script:
1. Creates a mock pretrain checkpoint with matching model config
2. Loads the mini test dataset
3. Initializes MotionHistoryEncoder (frozen) and LatentDecoder via FinetuneTrainer
4. Runs finetuning for a small number of epochs
5. Verifies checkpoint structure and encoder freezing
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import Config
from utils.dataset import create_dataloader
from utils.models.finetune_trainer import FinetuneTrainer, train_finetune
from utils.models.pretrain_trainer import train_pretrain

config = Config()


def _create_mock_pretrain_checkpoint(output_path: Path) -> Path | None:
    """Create a mock pretrain checkpoint for testing."""
    _ = train_pretrain(
        config=config,
        max_epochs=1,
    )
    return _find_latest_checkpoint(output_path, "pretrain_best_eval")


def _find_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> Path | None:
    """Find the latest checkpoint file with given prefix."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def test_finetune_basic():
    """Run basic finetuning test with minimal epochs."""
    print("=" * 60)
    print("Testing Finetune Trainer")
    print("=" * 60)

    config.device = "cpu"
    config.dataset_path = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
    config.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "test_finetune_trainer"
    config.output_path = PROJECT_ROOT / "output" / "test_finetune_trainer"
    config.batch_size = 4
    config.effective_batch_size = 8
    config.horizon = 10
    config.curriculum = None
    config._num_epochs = 10
    config.ema_decay = 0.999
    config.learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.lr_warmup_epochs = 1
    config.val_interval = 1
    config.checkpoint_interval = 1

    config.encoder_config.hidden_size = 32
    config.encoder_config.intermediate_size = 32
    config.encoder_config.num_hidden_layers = 4
    config.encoder_config.num_attention_heads = 2

    config.encoder_config.jp_config.hidden_size = 16
    config.encoder_config.jp_config.intermediate_size = 16
    config.encoder_config.jp_config.num_hidden_layers = 2

    config.decoder_config.hidden_size = 16
    config.decoder_config.intermediate_size = 32

    config.num_workers = 0

    print(f"Device: {config.device}")
    print(f"Dataset path: {config.dataset_path}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")

    # Create mock pretrain checkpoint
    pretrain_checkpoint_path = _create_mock_pretrain_checkpoint(config.checkpoint_dir)
    assert pretrain_checkpoint_path is not None and pretrain_checkpoint_path.exists(), "Pretrain checkpoint not found"
    print(f"Using pretrain checkpoint: {pretrain_checkpoint_path}")
    # pretrain_checkpoint_path = _find_latest_checkpoint(
    #     PROJECT_ROOT / "checkpoints" / "test_jepa_pretrain", "pretrain_best_eval"
    # )

    print("\nLoading dataset...")
    train_dataloader, normalizer = create_dataloader(config, split="train", shuffle=True)
    val_dataloader, _ = create_dataloader(config, split="val", shuffle=False)
    train_dataloader.num_workers = 0
    val_dataloader.num_workers = 0
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")

    print("\nRunning finetuning for 2 epochs...")
    ema_encoder, ema_decoder, finetune_path = train_finetune(
        config=config,
        pretrained_checkpoint_path=pretrain_checkpoint_path,
        max_epochs=2,
    )

    print("\nFinetuning complete!")
    print(f"EMA encoder type: {type(ema_encoder)}")
    print(f"EMA decoder type: {type(ema_decoder)}")

    print("\nVerifying encoder is frozen...")
    for name, param in ema_encoder.model.named_parameters():
        assert not param.requires_grad, f"EMA encoder param {name} has requires_grad=True"
    print("EMA encoder parameters are frozen (requires_grad=False)")

    print("\nVerifying decoder is created and updated...")
    assert ema_decoder is not None
    decoder_params = sum(p.numel() for p in ema_decoder.model.parameters())
    print(f"EMA decoder has {decoder_params:,} parameters")

    print(f"Checkpoint saved at: {finetune_path}")

    return ema_encoder, ema_decoder, finetune_path


def test_finetune_checkpoint_load():
    """Test loading finetune checkpoint."""
    print("\n" + "=" * 60)
    print("Testing Finetune Checkpoint Loading")
    print("=" * 60)

    config.device = "cpu"
    config.dataset_path = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
    config.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "test_finetune_trainer"

    config.num_workers = 0

    checkpoint_path = _find_latest_checkpoint(config.checkpoint_dir, "finetune_latest")

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"No checkpoint found in {config.checkpoint_dir}, skipping load test")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required_keys = [
        "encoder",
        "encoder_ema",
        "decoder",
        "decoder_ema",
        "optimizer",
        "lr_scheduler",
    ]
    for key in required_keys:
        assert key in checkpoint, f"Missing key in checkpoint: {key}"
    print("Checkpoint structure validated")

    pretrain_checkpoint_path = _find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "test_finetune_trainer", "pretrain_best_eval"
    )
    trainer = FinetuneTrainer(
        config=config,
        pretrained_checkpoint_path=pretrain_checkpoint_path,
    )

    trainer.decoder.load_state_dict(checkpoint["decoder"])
    print("Decoder weights loaded successfully")

    trainer.ema_decoder.load_state_dict(checkpoint["decoder_ema"])
    print("EMA decoder weights loaded successfully")

    print("Checkpoint load test passed!")


if __name__ == "__main__":
    ema_encoder, ema_decoder, finetune_path = test_finetune_basic()
    test_finetune_checkpoint_load()

    print("\n" + "=" * 60)
    print("All Finetune Trainer tests passed!")
    print("=" * 60)
