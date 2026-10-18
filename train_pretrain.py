#!/usr/bin/env python3
"""Phase 1 Motion History Encoder & Manifold Pretraining Script.

Extracts and standalone-executes the Phase 1 pretraining loop from src/pretrain.ipynb,
including I-JEPA masked contextual representation learning, linear text-probe alignment,
64D bottleneck manifold projection, Kinetic Geodesic Action Loss (L_geom),
and Covariance Whitening (L_cov).

Usage:
    # 1. Edit the CONFIGURATION & HYPERPARAMETERS section below.
    # 2. Run in foreground on GPU:
        python train_pretrain.py

    # 3. Or run in background with logging:
        nohup python train_pretrain.py > pretrain_training.log 2>&1 &
        tail -f pretrain_training.log

    # 4. Or override parameters via CLI:
        python train_pretrain.py --epochs 500 --batch_size 128 --effective_batch_size 256 --latent_dim 64
"""

# =============================================================================
# OPTIONABLE VARIABLES & HYPERPARAMETERS
# (Edit values here directly, or override via CLI arguments)
# =============================================================================

# --- Training & Optimization ---
NUM_EPOCHS: int = 500                # Total pretraining epochs
BATCH_SIZE: int = 128                # Micro-batch size per forward/backward pass
EFFECTIVE_BATCH_SIZE: int = 256      # Effective batch size via gradient accumulation (256 // 128 = 2 steps)
LEARNING_RATE: float = 0.5e-4        # Peak AdamW learning rate
WEIGHT_DECAY: float = 1e-5           # Weight decay for AdamW
GRADIENT_CLIP: float = 30.0          # Max gradient norm clipping threshold
EMA_DECAY: float = 0.999             # Exponential moving average decay for target encoder

# --- Motion Sequences & Horizon ---
HORIZON: int = 80                    # Motion window length (frames at 20fps = 4.0s)
CURRICULUM: list | None = None       # Set to None for fixed horizon, or list of curriculum stages

# --- 64D Geodesic Latent Manifold ---
LATENT_DIM: int = 64                 # Latent manifold dimensionality (exact physical kinematic rank)
LOSS_GEOM_WEIGHT: float = 0.10       # Kinetic Geodesic Action Loss weight L_geom (velocity + acceleration)
LOSS_COV_WEIGHT: float = 0.05        # Latent Covariance Whitening weight L_cov (eliminates anisotropy)
JEPA_CTX_WEIGHT: float = 0.20        # Context reconstruction weight in I-JEPA loss
CFG_DROPOUT: float = 0.15            # Conditioning dropout probability for text alignment

# --- Validation & Checkpoint Frequencies ---
VAL_INTERVAL: int = 25               # Evaluate on validation split every N epochs
CHECKPOINT_INTERVAL: int = 50        # Save periodic checkpoints every N epochs

# --- Paths & Logging ---
# Paths can be relative (to project root or src) or absolute. Auto-detected if None.
DATASET_PATH: str | None = None       # Defaults to ./dataset/humanml3d-subset or src/dataset/humanml3d-subset
CHECKPOINT_DIR: str | None = None     # Defaults to ./checkpoints/pretrain
OUTPUT_PATH: str | None = None        # Defaults to ./output/pretrain
LOG_FILE: str | None = "pretrain_training.log"  # Log file for background logging

# --- Hardware & Tracking ---
DEVICE: str = "cuda"                  # "cuda" or "cpu"
USE_WANDB: bool = False               # Set True to enable Weights & Biases logging
WANDB_PROJECT: str = "pretrain"       # W&B project name if enabled

# =============================================================================
# END OF USER OPTIONS
# =============================================================================

import argparse
from datetime import datetime, timezone, timedelta
import gc
import os
from pathlib import Path
import signal
import sys
import time

# Unbuffer stdout and stderr so background redirection is realtime
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Safe unpickling for checkpoints saved on Windows or Linux
import pathlib
try:
    pathlib.WindowsPath = pathlib.PosixPath
except Exception:
    pass

import torch

# Resolve directories
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "utils").exists():
    SRC_DIR = SCRIPT_DIR
    ROOT_DIR = SCRIPT_DIR.parent
elif (SCRIPT_DIR / "src" / "utils").exists():
    SRC_DIR = SCRIPT_DIR / "src"
    ROOT_DIR = SCRIPT_DIR
else:
    SRC_DIR = SCRIPT_DIR
    ROOT_DIR = SCRIPT_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import Config
from utils.models import _find_latest_checkpoint
from utils.models.pretrain_trainer import PretrainTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 Motion History Encoder & 64D Manifold Pretraining Script")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Number of pretraining epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--effective_batch_size", type=int, default=EFFECTIVE_BATCH_SIZE, help=f"Effective batch size (default: {EFFECTIVE_BATCH_SIZE})")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help=f"Peak learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help=f"Weight decay (default: {WEIGHT_DECAY})")
    parser.add_argument("--gradient_clip", type=float, default=GRADIENT_CLIP, help=f"Gradient clipping max norm (default: {GRADIENT_CLIP})")
    parser.add_argument("--horizon", type=int, default=HORIZON, help=f"Motion horizon in frames (default: {HORIZON})")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM, help=f"Latent manifold dimensionality (default: {LATENT_DIM})")
    parser.add_argument("--loss_geom_weight", type=float, default=LOSS_GEOM_WEIGHT, help=f"Kinetic Geodesic loss weight L_geom (default: {LOSS_GEOM_WEIGHT})")
    parser.add_argument("--loss_cov_weight", type=float, default=LOSS_COV_WEIGHT, help=f"Covariance whitening loss weight L_cov (default: {LOSS_COV_WEIGHT})")
    parser.add_argument("--jepa_ctx_weight", type=float, default=JEPA_CTX_WEIGHT, help=f"JEPA context loss weight (default: {JEPA_CTX_WEIGHT})")
    parser.add_argument("--cfg_dropout", type=float, default=CFG_DROPOUT, help=f"CFG text dropout probability (default: {CFG_DROPOUT})")
    parser.add_argument("--val_interval", type=int, default=VAL_INTERVAL, help=f"Validation interval in epochs (default: {VAL_INTERVAL})")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL, help=f"Periodic checkpoint save interval (default: {CHECKPOINT_INTERVAL})")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to dataset directory (default: auto-detect)")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Base directory for checkpoints (default: auto-detect)")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Output directory (default: auto-detect)")
    parser.add_argument("--device", type=str, default=DEVICE, help=f"Compute device (default: {DEVICE})")
    parser.add_argument("--wandb", action="store_true", default=USE_WANDB, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT, help=f"W&B project name (default: {WANDB_PROJECT})")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print("=" * 75)
    print("      Phase 1 Motion History Encoder 64D Manifold Pretraining")
    print("=" * 75)
    print(f"Start Time         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Target Epochs      : {args.epochs}")
    print(f"Batch Size         : {args.batch_size} (Effective: {args.effective_batch_size})")
    print(f"Learning Rate      : {args.learning_rate}")
    print(f"Motion Horizon     : {args.horizon} frames ({args.horizon / 20.0:.1f}s)")
    print(f"Latent Manifold    : {args.latent_dim}D (Geodesic-Regularized)")
    print(f"L_geom Weight      : {args.loss_geom_weight} (Flash & Hogan minimum-jerk action)")
    print(f"L_cov Weight       : {args.loss_cov_weight} (Covariance whitening / anti-collapse)")
    print(f"JEPA Context Weight: {args.jepa_ctx_weight}")
    print(f"Val Interval       : Every {args.val_interval} epochs")
    print(f"Periodic Save      : Every {args.checkpoint_interval} epochs")
    print("=" * 75)

    # 1. Resolve dataset and checkpoint paths
    dataset_candidates = [
        Path(args.dataset_path) if args.dataset_path else None,
        SRC_DIR / "dataset" / "humanml3d-subset",
        ROOT_DIR / "dataset" / "humanml3d-subset",
        Path("./dataset/humanml3d-subset"),
    ]
    dataset_path = None
    for cand in dataset_candidates:
        if cand is not None and cand.exists():
            dataset_path = cand.resolve()
            break

    if dataset_path is None:
        raise FileNotFoundError(
            "Could not locate HumanML3D dataset directory. "
            "Please specify --dataset_path /path/to/dataset"
        )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        SRC_DIR / "checkpoints" / "pretrain" if (SRC_DIR / "checkpoints").exists()
        else ROOT_DIR / "checkpoints" / "pretrain"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output_path) if args.output_path else (
        SRC_DIR / "output" / "pretrain" if (SRC_DIR / "output").exists()
        else ROOT_DIR / "output" / "pretrain"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Setup Configuration
    config = Config()
    config.device = device
    config.dataset_path = dataset_path
    config.checkpoint_dir = checkpoint_dir
    config.output_path = output_path
    config.batch_size = args.batch_size
    config.effective_batch_size = args.effective_batch_size
    config._num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.gradient_clip = args.gradient_clip
    config.ema_decay = EMA_DECAY
    config.curriculum = None
    config.horizon = args.horizon
    config.val_interval = args.val_interval
    config.checkpoint_interval = args.checkpoint_interval

    # 64D Latent Manifold Sub-configs
    config.latent_dim = args.latent_dim
    config.encoder_config.latent_dim = args.latent_dim
    config.decoder_config.latent_dim = args.latent_dim
    config.predictor_config.latent_dim = args.latent_dim

    # Regularization loss weights
    config.pre_conf.loss_geom_weight = args.loss_geom_weight
    config.pre_conf.loss_cov_weight = args.loss_cov_weight
    config.pre_conf.jepa_ctx_weight = args.jepa_ctx_weight
    config.pre_conf.cfg_dropout = args.cfg_dropout

    print(f"\n[1/3] Dataset location   : {dataset_path}")
    print(f"      Checkpoints dir    : {checkpoint_dir}")
    print(f"      Outputs dir        : {output_path}")

    # 3. Clean VRAM before allocating models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Instantiate PretrainTrainer
    print(f"\n[2/3] Initializing PretrainTrainer (64D Manifold)...")
    trainer = PretrainTrainer(
        config=config,
        wandb_project=args.wandb_project if args.wandb else None,
    )

    # Signal handlers for clean interruption
    def handle_signal(sig, frame):
        print(f"\n[!] Received signal {sig}. Initiating graceful termination...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 5. Execute Training Loop
    print(f"\n[3/3] Starting pretraining execution across {args.epochs} epochs...")
    start_time = time.time()
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n[!] Pretraining paused via KeyboardInterrupt.")
    except Exception as e:
        print(f"\n[X] Pretraining encountered error: {e}")
        raise e
    finally:
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\n" + "=" * 75)
        print(f"Training session ended. Total runtime: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")

        latest_ckpt = _find_latest_checkpoint(checkpoint_dir, prefix="pretrain_best_val")
        if latest_ckpt:
            print(f"Best validation checkpoint: {latest_ckpt}")
        latest_file = checkpoint_dir / "pretrain_latest.pt"
        if latest_file.exists():
            print(f"Latest checkpoint         : {latest_file}")
        print("=" * 75)


if __name__ == "__main__":
    main()
