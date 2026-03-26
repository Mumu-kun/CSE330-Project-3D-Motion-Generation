"""
Notebook-style end-to-end smoke test for the motion generation pipeline.

This script mirrors the flow in src/pipeline.ipynb, but uses a tiny fixture
dataset and a much smaller model so it can run quickly on CPU.

Run from project root:
    python tests/test_pipeline_e2e.py
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, FlowMatchingPredictorConfig
from models import MotionHistoryEncoder, FlowMatchingPredictor
from utils.dataset import create_dataloader
from utils.train_utils import Trainer


SOURCE_DATASET = Path("./sample_data/humanml3d-subset")
MINI_DATASET = Path("./tests/dataset/humanml3d-subset-mini")
CHECKPOINT_DIR = Path("./tests/checkpoints/test_e2e")


def _read_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _write_ids(path: Path, ids: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sample_id in ids:
            handle.write(f"{sample_id}\n")


def _fixture_is_complete(target_root: Path) -> bool:
    required_files = [
        "Mean.npy",
        "Std.npy",
        "text_embeddings_cache.pt",
        "train.txt",
        "val.txt",
        "test.txt",
        "all.txt",
    ]
    if not target_root.exists():
        return False

    for filename in required_files:
        if not (target_root / filename).exists():
            return False

    try:
        split_ids = []
        for split_name in ("train", "val", "test"):
            split_ids.extend(_read_ids(target_root / f"{split_name}.txt"))
        unique_ids = sorted(set(split_ids))
    except FileNotFoundError:
        return False

    for sample_id in unique_ids:
        if not (target_root / "new_joint_vecs" / f"{sample_id}.npy").exists():
            return False
        if not (target_root / "new_joints" / f"{sample_id}.npy").exists():
            return False
        if not (target_root / "texts" / f"{sample_id}.txt").exists():
            return False

    return True


def prepare_mini_dataset(
    source_root: Path,
    target_root: Path,
    train_size: int = 16,
    val_size: int = 4,
    test_size: int = 4,
) -> Path:
    """Create or refresh a tiny fixture dataset for CPU smoke tests."""
    if _fixture_is_complete(target_root):
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


def get_smoke_config() -> Config:
    """Build a compact config that mirrors the notebook but is fast on CPU."""
    config = Config()

    config.dataset_path = prepare_mini_dataset(SOURCE_DATASET, MINI_DATASET)
    config.checkpoint_dir = CHECKPOINT_DIR
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config.device = torch.device("cpu")
    config.pin_memory = False
    config.num_workers = 0

    config.batch_size = 4
    config.num_epochs = 1
    config.max_motion_length = 20
    config.horizon = 5

    config.encoder_motion_dim = 271
    config.encoder_text_dim = 512
    config.encoder_text_proj_dim = 32
    config.encoder_hidden_dim = 64
    config.encoder_per_joint_dim = 64
    config.encoder_num_layers = 1
    config.encoder_text_scale = 1.0
    config.encoder_dropout = 0.0

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

    config.learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.gradient_clip = 1.0
    config.ema_decay = 0.999
    config.cfg_dropout = 0.0
    config.use_fk = False
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    config.rollout_integration_steps = 1
    config.use_consistency_loss = False
    config.curriculum = None

    config.val_interval = 9999
    config.val_batches = 1
    config.val_use_ema = True
    config.save_best_val = True

    config.enable_profiling = False
    config.timing_log_interval = 100

    return config


def _ensure_text_embedding(batch_text: torch.Tensor) -> torch.Tensor:
    if batch_text.ndim != 3 or batch_text.shape[1] != 1:
        raise ValueError(
            f"Expected text embedding shape (B, 1, 512), got {tuple(batch_text.shape)}"
        )
    return batch_text.squeeze(1)


def _print_batch_summary(config: Config, batch: dict) -> None:
    print(f"Dataset path: {config.dataset_path}")
    print(f"Batch size: {config.batch_size}")
    print(f"Captions: {len(batch['captions'])} samples")
    print(f"Motion shape: {batch['motion'].shape}")
    print(f"Joints shape: {batch['joints'].shape}")
    print(f"Text embeddings shape: {batch['text_clip'].shape}")
    print(f"Lengths shape: {batch['lengths'].shape}")
    print(f"Sample caption: '{batch['captions'][0]}'")


def _build_models(config: Config, normalizer):
    encoder = MotionHistoryEncoder(
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
    ).to(config.device)

    predictor = FlowMatchingPredictor(
        feature_size=config.encoder_per_joint_dim,
        config=config.predictor_config,
        out_channels=3,
        use_relative_shift=True,
    ).to(config.device)

    return encoder, predictor


def _run_forward_check(
    config: Config,
    encoder: MotionHistoryEncoder,
    predictor: FlowMatchingPredictor,
    batch: dict,
) -> None:
    device = torch.device(config.device)
    motion = batch["motion"].to(device)
    joints = batch["joints"].to(device)
    text_emb = _ensure_text_embedding(batch["text_clip"].to(device))

    normalizer = encoder.normalizer
    motion_input = normalizer.normalize(motion) if normalizer is not None else motion

    encoder.eval()
    predictor.eval()

    with torch.no_grad():
        track_features = encoder(motion_input, text_emb)
        if track_features.shape != (
            motion.shape[0],
            config.encoder_num_joints,
            config.encoder_per_joint_dim,
        ):
            raise AssertionError(
                f"Unexpected encoder output shape: {tuple(track_features.shape)}"
            )

        last_positions = joints[:, -1]
        prev_positions = (
            joints[:, -2] if joints.shape[1] > 1 else torch.zeros_like(last_positions)
        )
        relative_shifts = last_positions - prev_positions
        timesteps = torch.rand(motion.shape[0], device=device)

        pred_vel, _, _ = predictor(
            noised_tracks=last_positions,
            timesteps=timesteps,
            text_embedding=text_emb,
            track_features=track_features,
            prev_relative_shifts=relative_shifts,
            output_attentions=False,
            output_hidden_states=False,
        )

    if pred_vel.shape != (motion.shape[0], config.encoder_num_joints, 3):
        raise AssertionError(
            f"Unexpected predictor output shape: {tuple(pred_vel.shape)}"
        )
    if torch.isnan(pred_vel).any():
        raise AssertionError("Predictor output contains NaNs")

    print(f"Encoder output shape: {track_features.shape}")
    print(f"Predictor output shape: {pred_vel.shape}")


def run_e2e_test() -> bool:
    """Run the notebook-style smoke test end to end."""
    print("=" * 60)
    print("3D HUMAN MOTION GENERATION PIPELINE SMOKE TEST")
    print("=" * 60)

    config = get_smoke_config()

    print("\nConfiguration")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Horizon: {config.horizon}")
    print(f"  Encoder hidden dim: {config.encoder_hidden_dim}")
    print(f"  Encoder per-joint dim: {config.encoder_per_joint_dim}")
    print(f"  Predictor hidden size: {config.predictor_config.hidden_size}")

    try:
        print("\n" + "=" * 60)
        print("LOAD DATASET")
        print("=" * 60)
        train_dataloader, normalizer = create_dataloader(
            config, split="train", shuffle=True
        )
        val_dataloader, _ = create_dataloader(config, split="val", shuffle=False)
        batch = next(iter(train_dataloader))
        _print_batch_summary(config, batch)
        print(f"Train batches: {len(train_dataloader)}")
        print(f"Validation batches: {len(val_dataloader)}")

        print("\n" + "=" * 60)
        print("SETUP MODELS")
        print("=" * 60)
        encoder, predictor = _build_models(config, normalizer)
        encoder_params = sum(p.numel() for p in encoder.parameters())
        predictor_params = sum(p.numel() for p in predictor.parameters())
        print(f"Motion encoder parameters: {encoder_params:,}")
        print(f"Flow predictor parameters: {predictor_params:,}")

        print("\n" + "=" * 60)
        print("FORWARD CHECK")
        print("=" * 60)
        _run_forward_check(config, encoder, predictor, batch)

        print("\n" + "=" * 60)
        print("TRAIN")
        print("=" * 60)
        if config.checkpoint_dir.exists():
            shutil.rmtree(config.checkpoint_dir)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ema_encoder, ema_predictor = Trainer.train(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            normalizer=normalizer,
            wandb_project=None,
            wandb_run_name=None,
            encoder_override=encoder,
            predictor_override=predictor,
        )

        latest_ckpt = config.checkpoint_dir / "latest.pt"
        best_ckpt = config.checkpoint_dir / "best.pt"

        if not latest_ckpt.exists():
            raise AssertionError(f"Expected checkpoint not found: {latest_ckpt}")
        if not best_ckpt.exists():
            raise AssertionError(f"Expected checkpoint not found: {best_ckpt}")
        if ema_encoder is None or ema_predictor is None:
            raise AssertionError("EMA models were not returned by Trainer.train")

        print(f"Latest checkpoint: {latest_ckpt}")
        print(f"Best checkpoint: {best_ckpt}")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return True
    except Exception as exc:
        print(f"\n[X] TEST FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_e2e_smoke() -> None:
    assert run_e2e_test()


if __name__ == "__main__":
    raise SystemExit(0 if run_e2e_test() else 1)
