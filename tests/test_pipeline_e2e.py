"""
Minimal end-to-end pipeline smoke test against current training APIs.

This script validates:
- Data loading from sample_data/humanml3d-subset
- MotionHistoryEncoder + FlowMatchingPredictor forward compatibility
- Training setup utilities with W&B disabled
- Checkpoint artifact creation through training utility helpers

Run from project root:
    python tests/test_pipeline_e2e.py
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, FlowMatchingPredictorConfig
from models import FlowMatchingPredictor, MotionHistoryEncoder
from utils.dataset import create_dataloader


def _read_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_ids(path: Path, ids: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample_id in ids:
            f.write(f"{sample_id}\n")


def prepare_mini_dataset(
    source_root: Path,
    target_root: Path,
    train_size: int = 32,
    val_size: int = 8,
    test_size: int = 8,
) -> Path:
    """Create a tiny dataset fixture so e2e tests run quickly on CPU."""
    if target_root.exists() and (target_root / "train.txt").exists():
        return target_root

    if target_root.exists():
        shutil.rmtree(target_root)
    (target_root / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (target_root / "new_joints").mkdir(parents=True, exist_ok=True)
    (target_root / "texts").mkdir(parents=True, exist_ok=True)

    for filename in ("Mean.npy", "Std.npy", "text_embeddings_cache.pt"):
        src = source_root / filename
        if src.exists():
            shutil.copy2(src, target_root / filename)

    train_ids = _read_ids(source_root / "train.txt")[:train_size]
    val_ids = _read_ids(source_root / "val.txt")[:val_size]
    test_ids = _read_ids(source_root / "test.txt")[:test_size]
    all_ids = sorted(set(train_ids + val_ids + test_ids))

    _write_ids(target_root / "train.txt", train_ids)
    _write_ids(target_root / "val.txt", val_ids)
    _write_ids(target_root / "test.txt", test_ids)
    _write_ids(target_root / "all.txt", all_ids)

    for sample_id in all_ids:
        shutil.copy2(
            source_root / "new_joint_vecs" / f"{sample_id}.npy",
            target_root / "new_joint_vecs" / f"{sample_id}.npy",
        )
        shutil.copy2(
            source_root / "new_joints" / f"{sample_id}.npy",
            target_root / "new_joints" / f"{sample_id}.npy",
        )
        shutil.copy2(
            source_root / "texts" / f"{sample_id}.txt",
            target_root / "texts" / f"{sample_id}.txt",
        )

    return target_root


def get_minimal_config() -> Config:
    """Get a tiny deterministic config suitable for e2e smoke testing."""
    config = Config()

    # Use a tiny fixture derived from sample_data to keep test startup fast.
    source_dataset = Path("./sample_data/humanml3d-subset")
    mini_dataset = Path("./tests/dataset/humanml3d-subset-mini")
    config.dataset_path = prepare_mini_dataset(source_dataset, mini_dataset)
    config.checkpoint_dir = Path("./tests/checkpoints/test_e2e")
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # CPU-only for portability.
    config.device = "cpu"
    config.pin_memory = False
    config.num_workers = 0

    # Keep runtime low.
    config.batch_size = 4
    config.num_epochs = 1
    config.max_motion_length = 20

    # Encoder: compact settings.
    config.encoder_hidden_dim = 64
    config.encoder_num_layers = 1
    config.encoder_per_joint_dim = 32
    config.encoder_text_proj_dim = 32
    config.encoder_dropout = 0.0

    # Predictor: new structured config API.
    config.predictor_config = FlowMatchingPredictorConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_bias=True,
        attention_dropout=0.0,
        mlp_bias=True,
        track_dimensionality=3,
        global_cond_dim=config.encoder_text_dim,
        head_dim=None,
    )

    # Stable, short training configuration.
    config.learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.gradient_clip = 1.0
    config.ema_decay = 0.999
    config.cfg_dropout = 0.0

    # Fixed horizon to avoid curriculum complexity in smoke test.
    config.curriculum = None
    config.horizon = 5

    # Skip validation pass for speed; this test is train-loop smoke coverage.
    config.val_interval = 9999
    config.val_batches = 1
    config.val_use_ema = True
    config.save_best_val = True

    return config


def _get_text_embedding(batch_text: torch.Tensor) -> torch.Tensor:
    """Convert strict text embeddings (B, 1, 512) to (B, 512)."""
    if batch_text.ndim != 3 or batch_text.shape[1] != 1:
        raise ValueError(
            f"Expected strict text shape (B, 1, 512), got {tuple(batch_text.shape)}"
        )
    return batch_text.squeeze(1)


def test_data_loading(config: Config):
    """Test that dataloader and feature tensors match current contracts."""
    print("\n" + "=" * 60)
    print("TEST: Data Loading")
    print("=" * 60)

    dataloader, normalizer = create_dataloader(config, split="train", shuffle=True)
    batch = next(iter(dataloader))

    print(f"Dataset path: {config.dataset_path}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Motion shape: {batch['motion'].shape}")
    print(f"Joints shape: {batch['joints'].shape}")
    print(f"Text shape: {batch['text_clip'].shape}")
    print(f"Lengths shape: {batch['lengths'].shape}")

    assert batch["motion"].ndim == 3 and batch["motion"].shape[-1] == 271
    assert batch["joints"].ndim == 4 and batch["joints"].shape[-2:] == (22, 3)
    assert batch["lengths"].ndim == 1
    assert batch["text_clip"].ndim == 3
    assert batch["text_clip"].shape[1:] == (1, config.encoder_text_dim)

    print("[OK] Data loading test PASSED")
    return dataloader, normalizer


def test_model_initialization(config: Config, normalizer):
    """Test model construction with current constructor signatures."""
    print("\n" + "=" * 60)
    print("TEST: Model Initialization")
    print("=" * 60)

    device = torch.device(config.device)

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

    flow_predictor = FlowMatchingPredictor(
        feature_size=config.encoder_per_joint_dim,
        config=config.predictor_config,
        out_channels=3,
        use_relative_shift=True,
    ).to(device)

    encoder_params = sum(p.numel() for p in motion_encoder.parameters())
    predictor_params = sum(p.numel() for p in flow_predictor.parameters())
    print(f"Motion Encoder parameters: {encoder_params:,}")
    print(f"Flow Predictor parameters: {predictor_params:,}")

    assert encoder_params > 0
    assert predictor_params > 0

    print("[OK] Model initialization test PASSED")
    return motion_encoder, flow_predictor


def test_forward_pass(
    config: Config,
    motion_encoder: MotionHistoryEncoder,
    flow_predictor: FlowMatchingPredictor,
    dataloader,
):
    """Test encoder and predictor forward pass with current tensor contracts."""
    print("\n" + "=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)

    device = torch.device(config.device)
    batch = next(iter(dataloader))

    motion = batch["motion"].to(device)
    joints = batch["joints"].to(device)
    text_clip = batch["text_clip"].to(device)

    B, T, _ = motion.shape
    text_emb = _get_text_embedding(text_clip)
    assert text_emb.shape == (B, config.encoder_text_dim)

    normalizer = motion_encoder.normalizer
    motion_norm = normalizer.normalize(motion) if normalizer is not None else motion

    motion_encoder.eval()
    flow_predictor.eval()

    with torch.no_grad():
        track_features = motion_encoder(motion_norm, text_emb)
        assert track_features.shape == (B, 22, config.encoder_per_joint_dim)

        # One-step flow matching style input.
        x1 = joints[:, -1] - joints[:, -2] if T > 1 else torch.zeros_like(joints[:, -1])
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device)
        x_t = t.view(B, 1, 1) * x1 + (1 - t.view(B, 1, 1)) * x0

        pred_vel, _, _ = flow_predictor(
            noised_tracks=x_t,
            timesteps=t,
            text_embedding=text_emb,
            track_features=track_features,
            relative_shifts=x1,
            output_attentions=False,
            output_hidden_states=False,
        )

    print(f"Track features shape: {track_features.shape}")
    print(f"Predictor output shape: {pred_vel.shape}")

    assert pred_vel.shape == (B, 22, 3)
    assert not torch.isnan(pred_vel).any()

    print("[OK] Forward pass test PASSED")


def test_training_step(
    config: Config,
    motion_encoder: MotionHistoryEncoder,
    flow_predictor: FlowMatchingPredictor,
    dataloader,
):
    """Test one manual optimization step using current predictor API."""
    print("\n" + "=" * 60)
    print("TEST: Training Step")
    print("=" * 60)

    device = torch.device(config.device)
    optimizer = torch.optim.AdamW(
        list(motion_encoder.parameters()) + list(flow_predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    batch = next(iter(dataloader))
    motion = batch["motion"].to(device)
    joints = batch["joints"].to(device)
    text_emb = _get_text_embedding(batch["text_clip"].to(device))

    normalizer = motion_encoder.normalizer
    motion_norm = normalizer.normalize(motion) if normalizer is not None else motion

    motion_encoder.train()
    flow_predictor.train()

    B, T, _ = motion_norm.shape
    if T < 2:
        raise ValueError(
            "Need at least 2 frames to form a flow-matching training step."
        )
    history_len = min(5, T - 1)
    history = motion_norm[:, :history_len]

    track_features = motion_encoder(history, text_emb)

    x1 = joints[:, history_len] - joints[:, history_len - 1]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)
    x_t = t.view(B, 1, 1) * x1 + (1 - t.view(B, 1, 1)) * x0

    pred_vel, _, _ = flow_predictor(
        noised_tracks=x_t,
        timesteps=t,
        text_embedding=text_emb,
        track_features=track_features,
        relative_shifts=x1,
    )

    target_vel = x1 - x0
    loss = F.mse_loss(pred_vel, target_vel)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(motion_encoder.parameters()) + list(flow_predictor.parameters()),
        config.gradient_clip,
    )
    optimizer.step()

    print(f"Loss: {loss.item():.6f}")
    assert loss.item() > 0.0
    assert loss.item() < 1e6

    print("[OK] Training step test PASSED")


def test_training_utilities_without_wandb(
    config: Config,
    motion_encoder: MotionHistoryEncoder,
    flow_predictor: FlowMatchingPredictor,
    dataloader,
    normalizer,
):
    """Validate training setup/checkpoint helpers with W&B disabled."""
    print("\n" + "=" * 60)
    print("TEST: Training Utilities Without W&B")
    print("=" * 60)

    # Ensure checkpoint assertions validate this run only.
    if config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Import lazily to avoid pulling heavy optional deps before smoke checks.
    from utils.train_utils import Trainer

    trainer = Trainer(
        encoder=motion_encoder,
        predictor=flow_predictor,
        dataloader=dataloader,
        config=config,
        normalizer=normalizer,
        val_dataloader=None,
    )

    (
        _device,
        save_dir,
        wandb_logger,
        encoder_ema,
        predictor_ema,
        optimizer,
        scaler,
        _start_epoch,
        training_state,
        _device_str,
        _use_amp,
    ) = trainer.setup_training_environment()

    assert wandb_logger is None, "wandb logger should be disabled when project is None"

    # Run one tiny optimizer step to verify returned optimizer/scaler are usable.
    batch = next(iter(dataloader))
    motion = batch["motion"].to(config.device)
    joints = batch["joints"].to(config.device)
    text_emb = _get_text_embedding(batch["text_clip"].to(config.device))
    motion_norm = normalizer.normalize(motion)

    history_len = min(5, motion_norm.shape[1] - 1)
    history = motion_norm[:, :history_len]
    track_features = motion_encoder(history, text_emb)

    x1 = joints[:, history_len] - joints[:, history_len - 1]
    x0 = torch.randn_like(x1)
    t = torch.rand(motion.shape[0], device=config.device)
    x_t = t.view(motion.shape[0], 1, 1) * x1 + (1 - t.view(motion.shape[0], 1, 1)) * x0

    pred_vel, _, _ = flow_predictor(
        noised_tracks=x_t,
        timesteps=t,
        text_embedding=text_emb,
        track_features=track_features,
        relative_shifts=x1,
    )
    loss = F.mse_loss(pred_vel, x1 - x0)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    curriculum_state = {
        "use_curriculum": False,
        "current_horizon": config.horizon,
        "max_horizon": config.horizon,
    }
    trainer.save_training_checkpoint(
        save_dir=save_dir,
        filename="latest.pt",
        encoder_ema=encoder_ema,
        predictor_ema=predictor_ema,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        global_step=training_state["global_step"],
        loss=float(loss.detach().cpu().item()),
        curriculum_state=curriculum_state,
        training_state=training_state,
    )

    latest_ckpt = config.checkpoint_dir / "latest.pt"
    best_ckpt = config.checkpoint_dir / "best.pt"

    assert encoder_ema is not None
    assert predictor_ema is not None
    assert latest_ckpt.exists(), f"Expected checkpoint not found: {latest_ckpt}"
    if best_ckpt.exists():
        print(f"Checkpoint created: {best_ckpt}")

    print(f"Checkpoint created: {latest_ckpt}")
    print("[OK] Training utilities without W&B test PASSED")


def run_e2e_test() -> bool:
    """Run the complete end-to-end smoke test."""
    print("=" * 60)
    print("MINIMAL END-TO-END PIPELINE TEST")
    print("=" * 60)
    print("Testing current train() and model APIs at tiny scale")
    print("=" * 60)

    config = get_minimal_config()

    print("\nConfiguration:")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Horizon: {config.horizon}")
    print(f"  Encoder per-joint dim: {config.encoder_per_joint_dim}")
    print(f"  Predictor hidden size: {config.predictor_config.hidden_size}")

    try:
        dataloader, normalizer = test_data_loading(config)
        motion_encoder, flow_predictor = test_model_initialization(config, normalizer)
        test_forward_pass(config, motion_encoder, flow_predictor, dataloader)
        test_training_step(config, motion_encoder, flow_predictor, dataloader)
        test_training_utilities_without_wandb(
            config,
            motion_encoder,
            flow_predictor,
            dataloader,
            normalizer,
        )

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
        print("\nCurrent training pipeline APIs are valid in e2e smoke mode.")
        return True
    except Exception as exc:
        print(f"\n[X] TEST FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
