#!/usr/bin/env python3
"""Phase 3 Flow Matching Predictor & LatentDecoder Training Script.

Extracts and standalone-executes the Phase 3 training loop from src/phase3.ipynb,
including prefix self-attention, rectified flow matching with uniform velocity loss,
relative kinematic loss (L_vel), and deterministic latent statistics serialization.

Usage:
    # 1. Edit the CONFIGURATION & HYPERPARAMETERS section below.
    # 2. Run in foreground:
        python train_phase3.py

    # 3. Or run in background with logging:
        nohup python train_phase3.py > phase3_training.log 2>&1 &
        tail -f phase3_training.log
"""

# =============================================================================
# OPTIONABLE VARIABLES & HYPERPARAMETERS
# (Edit values here directly, or override via CLI arguments)
# =============================================================================

# --- Component Training Toggles ---
TRAIN_DECODER: bool = False          # Set True to train/fine-tune the 72D LatentDecoder simultaneously
TRAIN_PREDICTOR: bool = True         # Set True to train the FlowMatchingPredictor

# --- Training & Optimization ---
NUM_EPOCHS: int = 1000               # Total training epochs
BATCH_SIZE: int = 128                # Batch size per step
EFFECTIVE_BATCH_SIZE: int = 256      # Effective batch size via gradient accumulation (256 // 128 = 2 steps)
GRADIENT_CLIP: float = 30.0          # Max norm for gradient clipping

# --- Motion Sequences & Context ---
HORIZON: int = 80                    # Target future motion prediction window (frames at 20fps = 4.0s)
HISTORY_LENGTH: int = 40             # Historical motion context window fed to encoder (frames = 2.0s)

# --- 64D Latent Manifold ---
LATENT_DIM: int = 64                 # Latent manifold dimensionality (exact physical kinematic rank)

# --- Flow Matching & Conditioning (Track B Calibrated) ---
CFG_DROPOUT: float = 0.15            # Conditioning dropout probability (p_uncond = 0.15 for clean null prior)
HISTORY_DROPOUT: float = 0.25        # History context dropout probability (forces pure text conditioning)
LOSS_VEL_WEIGHT: float = 0.1         # Relative velocity loss weight L_vel (smooth kinematic derivatives)
T_SAMPLING_POWER: float = 0.0        # Flow matching timestep distribution (0.0 = uniform Lipman/Albergo sampling)

# --- Validation & Checkpoint Frequencies ---
VAL_INTERVAL: int = 25               # Evaluate on validation split every N epochs
CHECKPOINT_INTERVAL: int = 100       # Save periodic checkpoints every N epochs (during final 100 epochs)

# --- Checkpoints & Resume ---
# Set RESUME_CHECKPOINT to a .pt path to resume training (e.g. "./checkpoints/phase3/predictor/phase3_predictor_latest_*.pt")
# Set to None to train a fresh Track B predictor from scratch
RESUME_CHECKPOINT: str | None = None

# Optional explicit paths to pretrained components (leave None for automatic discovery)
DECODER_CHECKPOINT: str | None = None
PRETRAIN_CHECKPOINT: str | None = None

# --- Paths & Logging ---
# Paths can be relative (to project root or src) or absolute. Auto-detected if None.
DATASET_PATH: str | None = None       # Defaults to ./dataset/humanml3d-subset or src/dataset/humanml3d-subset
CHECKPOINT_DIR: str | None = None     # Defaults to ./checkpoints/phase3
OUTPUT_PATH: str | None = None        # Defaults to ./output/phase3
LOG_FILE: str | None = "phase3_training.log"  # Log file for background logging

# --- Hardware & Tracking ---
DEVICE: str = "cuda"                  # "cuda" or "cpu"
USE_WANDB: bool = False               # Set True to enable Weights & Biases logging

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
import torch.nn.functional as F

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
from utils.dataset import create_dataloader
from utils.models import _find_latest_checkpoint
from utils.models.decoder_trainer import FinetuneTrainer
from utils.models.flow_matching_trainer import FlowMatchingTrainer
from utils.motion_utils import (
    FeatureNormalizer,
    Features,
    x271_to_x72,
    x72_to_positions,
)


def compute_decoder_loss(
    decoded: torch.Tensor,
    motion: torch.Tensor,
    joints: torch.Tensor,
    normalizer: FeatureNormalizer,
) -> torch.Tensor:
    """Reconstruction loss for 72D LatentDecoder with direct RIC position prediction."""
    if not hasattr(normalizer, "denormalize_x72"):
        normalizer = FeatureNormalizer(normalizer.mean, normalizer.std)

    prev_positions = joints[:, :-1, :].flatten(0, 1)
    prev_frames = motion[:, :-1, :].flatten(0, 1)
    y_frames = motion[:, 1:, :].flatten(0, 1)

    decoded_flat = decoded.flatten(0, 1)
    decoded_denorm = normalizer.denormalize_x72(decoded_flat)

    y_72d = x271_to_x72(y_frames, normalizer, prev_positions=prev_positions, prev_x271=prev_frames)
    y_72d_denorm = normalizer.denormalize_x72(y_72d)

    loss_root_y = F.mse_loss(decoded_denorm[:, :1], y_72d_denorm[:, :1])
    loss_root_xz = F.mse_loss(decoded_denorm[:, 1:3], y_72d_denorm[:, 1:3])
    loss_yaw = F.mse_loss(decoded_denorm[:, 3:5], y_72d_denorm[:, 3:5])
    loss_ric = F.mse_loss(decoded_denorm[:, 5:68], y_72d_denorm[:, 5:68])
    loss_contact = F.binary_cross_entropy_with_logits(decoded_flat[:, 68:72], y_frames[:, Features.CONTACTS])

    dec_positions = x72_to_positions(
        decoded_flat,
        normalizer,
        prev_x271=prev_frames,
        prev_positions=prev_positions,
    )
    loss_joint = F.mse_loss(dec_positions, joints[:, 1:, :].flatten(0, 1))

    return (
        1.0 * loss_root_y
        + 1.0 * loss_root_xz
        + 1.0 * loss_yaw
        + 1.0 * loss_ric
        + 0.5 * loss_contact
        + 1.0 * loss_joint
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 Flow Matching Predictor & Decoder Training Script")
    parser.add_argument("--train_decoder", action="store_true", default=TRAIN_DECODER, help=f"Train LatentDecoder simultaneously (default: {TRAIN_DECODER})")
    parser.add_argument("--no_train_decoder", action="store_false", dest="train_decoder", help="Disable Decoder training")
    parser.add_argument("--train_predictor", action="store_true", default=TRAIN_PREDICTOR, help=f"Train FlowMatchingPredictor (default: {TRAIN_PREDICTOR})")
    parser.add_argument("--no_train_predictor", action="store_false", dest="train_predictor", help="Disable Predictor training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--effective_batch_size", type=int, default=EFFECTIVE_BATCH_SIZE, help=f"Effective batch size (default: {EFFECTIVE_BATCH_SIZE})")
    parser.add_argument("--horizon", type=int, default=HORIZON, help=f"Prediction horizon in frames (default: {HORIZON})")
    parser.add_argument("--history_length", type=int, default=HISTORY_LENGTH, help=f"History context frames (default: {HISTORY_LENGTH})")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM, help=f"Latent manifold dimensionality (default: {LATENT_DIM})")
    parser.add_argument("--cfg_dropout", type=float, default=CFG_DROPOUT, help=f"Conditioning dropout probability (default: {CFG_DROPOUT})")
    parser.add_argument("--history_dropout", type=float, default=HISTORY_DROPOUT, help=f"History context dropout probability (default: {HISTORY_DROPOUT})")
    parser.add_argument("--loss_vel_weight", type=float, default=LOSS_VEL_WEIGHT, help=f"Relative velocity loss weight L_vel (default: {LOSS_VEL_WEIGHT})")
    parser.add_argument("--gradient_clip", type=float, default=GRADIENT_CLIP, help=f"Gradient clipping max norm (default: {GRADIENT_CLIP})")
    parser.add_argument("--val_interval", type=int, default=VAL_INTERVAL, help=f"Validation interval in epochs (default: {VAL_INTERVAL})")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL, help=f"Periodic checkpoint save interval (default: {CHECKPOINT_INTERVAL})")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to dataset directory (default: auto-detect)")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Base directory for checkpoints (default: auto-detect)")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Output directory (default: auto-detect)")
    parser.add_argument("--resume_checkpoint", type=str, default=RESUME_CHECKPOINT, help="Path to predictor checkpoint to resume from (default: None)")
    parser.add_argument("--reset_epoch", action="store_true", default=False, help="Reset epoch counter to 1 and reinitialize cosine LR scheduler for fine-tuning")
    parser.add_argument("--lr", type=float, default=None, help="Override fine-tuning learning rate (default: keep from optimizer/config)")
    parser.add_argument("--decoder_checkpoint", type=str, default=DECODER_CHECKPOINT, help="Path to pretrained/frozen decoder checkpoint (default: auto-detect)")
    parser.add_argument("--pretrain_checkpoint", type=str, default=PRETRAIN_CHECKPOINT, help="Path to pretrain encoder checkpoint (default: auto-detect)")
    parser.add_argument("--device", type=str, default=DEVICE, help=f"Compute device (default: {DEVICE})")
    parser.add_argument("--wandb", action="store_true", default=USE_WANDB, help="Enable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    train_decoder = args.train_decoder
    train_predictor = args.train_predictor

    print("=" * 70)
    print("      Phase 3 Motion Generation Training (Background Script)")
    print("=" * 70)
    print(f"Start Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device           : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Train Decoder    : {'ON (Training)' if train_decoder else 'OFF (Frozen/Evaluating)'}")
    print(f"Train Predictor  : {'ON (Training)' if train_predictor else 'OFF (Paused)'}")
    print(f"Target Epochs    : {args.epochs}")
    print(f"Batch Size       : {args.batch_size} (Effective: {args.effective_batch_size})")
    print(f"Horizon          : {args.horizon} frames ({args.horizon / 20.0:.1f}s)")
    print(f"History Length   : {args.history_length} frames ({args.history_length / 20.0:.1f}s)")
    print(f"Latent Manifold  : {args.latent_dim}D (Pure Latent Flow Matching)")
    print(f"CFG Dropout      : {args.cfg_dropout}")
    print(f"History Dropout  : {args.history_dropout}")
    print(f"L_vel Weight     : {args.loss_vel_weight}")
    print(f"Val Interval     : Every {args.val_interval} epochs")
    print(f"Periodic Save    : Every {args.checkpoint_interval} epochs")
    print("=" * 70)

    # 1. Resolve dataset and checkpoint paths
    dataset_path = Path(args.dataset_path) if args.dataset_path else (
        SRC_DIR / "dataset" / "humanml3d-subset" if (SRC_DIR / "dataset" / "humanml3d-subset").exists()
        else ROOT_DIR / "dataset" / "humanml3d-subset"
    )
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        SRC_DIR / "checkpoints" / "phase3"
    )
    output_path = Path(args.output_path) if args.output_path else (
        SRC_DIR / "output" / "phase3"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # 2. Setup Configuration
    config = Config()
    config.device = device
    config.dataset_path = dataset_path
    config.checkpoint_dir = checkpoint_dir
    config.output_path = output_path
    config.batch_size = args.batch_size
    config.effective_batch_size = args.effective_batch_size
    config._num_epochs = args.epochs
    config.curriculum = None
    config.horizon = args.horizon
    config.history_length = args.history_length
    config.t_sampling_power = T_SAMPLING_POWER
    config.cfg_dropout = args.cfg_dropout
    config.history_dropout = args.history_dropout
    config.predictor_config.history_dropout = args.history_dropout
    config.val_interval = args.val_interval
    config.checkpoint_interval = args.checkpoint_interval
    config.use_prefix_history = True
    config.loss_vel_weight = args.loss_vel_weight
    config.gradient_clip = args.gradient_clip

    # 64D Manifold Configuration
    config.latent_dim = args.latent_dim
    config.encoder_config.latent_dim = args.latent_dim
    config.decoder_config.latent_dim = args.latent_dim
    config.predictor_config.latent_dim = args.latent_dim

    # 3. Create DataLoaders
    print(f"\n[1/5] Loading datasets from {dataset_path}...")
    train_loader, normalizer = create_dataloader(config, "train", shuffle=True)
    val_loader, _ = create_dataloader(config, "val", shuffle=False)
    train_loader.dataset.set_horizon(config.horizon)
    val_loader.dataset.set_horizon(config.horizon)
    print(f"      Train Batches: {len(train_loader)}  |  Val Batches: {len(val_loader)}")

    # 4. Find pretrained encoder checkpoint
    if args.pretrain_checkpoint:
        pretrain_ckpt_path = Path(args.pretrain_checkpoint)
    else:
        pretrain_candidates = [
            SRC_DIR / "checkpoints" / "pretrain",
            ROOT_DIR / "checkpoints" / "pretrain",
            Path("./checkpoints/pretrain"),
        ]
        pretrain_ckpt_path = None
        for cand in pretrain_candidates:
            if cand.exists():
                try:
                    pretrain_ckpt_path = _find_latest_checkpoint(cand, prefix="pretrain_best_val")
                    break
                except Exception:
                    pass
        if pretrain_ckpt_path is None:
            pretrain_ckpt_path = SRC_DIR / "checkpoints" / "pretrain" / "pretrain_best_val_20261017_005458_425.pt"

    print(f"\n[2/5] Pretrained Encoder Checkpoint: {pretrain_ckpt_path}")

    # 5. Initialize Decoder Trainer
    decoder_config = Config()
    decoder_config.device = device
    decoder_config.dataset_path = config.dataset_path
    decoder_config.checkpoint_dir = config.checkpoint_dir / "decoder"
    decoder_config.output_path = config.output_path / "decoder"
    decoder_config.batch_size = config.batch_size
    decoder_config.effective_batch_size = config.effective_batch_size
    decoder_config._num_epochs = config._num_epochs
    decoder_config.curriculum = config.curriculum
    decoder_config.horizon = config.horizon
    decoder_config.history_length = config.history_length
    decoder_config.checkpoint_interval = config.checkpoint_interval

    # 64D Manifold Configuration
    decoder_config.latent_dim = args.latent_dim
    decoder_config.encoder_config.latent_dim = args.latent_dim
    decoder_config.decoder_config.latent_dim = args.latent_dim
    decoder_config.predictor_config.latent_dim = args.latent_dim

    decoder_trainer = FinetuneTrainer(
        config=decoder_config,
        pretrained_checkpoint_path=pretrain_ckpt_path,
        wandb_project="phase3_decoder" if args.wandb else None,
    )

    # Load frozen Decoder checkpoint (Phase 2 output)
    if args.decoder_checkpoint:
        dec_ckpt_path = Path(args.decoder_checkpoint)
        if not dec_ckpt_path.exists():
            print(f"      Warning: Specified decoder checkpoint not found at {dec_ckpt_path}.")
            dec_ckpt_path = None
    else:
        dec_cand_dirs = [
            SRC_DIR / "checkpoints" / "phase3" / "decoder",
            ROOT_DIR / "checkpoints" / "phase3" / "decoder",
            Path("./checkpoints/phase3/decoder"),
            SRC_DIR / "checkpoints" / "finetune",
            ROOT_DIR / "checkpoints" / "finetune",
            Path("./checkpoints/finetune"),
            Path("/kaggle/input/3mogenfinetune"),
        ]
        dec_ckpt_path = None
        for cand in dec_cand_dirs:
            if cand.exists():
                try:
                    dec_ckpt_path = _find_latest_checkpoint(cand, prefix="finetune_best_val")
                    if dec_ckpt_path is None:
                        dec_ckpt_path = _find_latest_checkpoint(cand, prefix="phase3_decoder_best_val")
                    if dec_ckpt_path is None:
                        dec_ckpt_path = _find_latest_checkpoint(cand, prefix="finetune_latest")
                    if dec_ckpt_path is not None:
                        break
                except Exception:
                    pass

        if dec_ckpt_path is None:
            fallback = SRC_DIR / "checkpoints" / "phase3" / "decoder" / "finetune_epoch_500_20261017_011538.pt"
            if fallback.exists():
                dec_ckpt_path = fallback

    if dec_ckpt_path is not None and dec_ckpt_path.exists():
        dec_ckpt = torch.load(dec_ckpt_path, map_location=device, weights_only=False)
        if "decoder" in dec_ckpt:
            decoder_trainer.decoder.load_state_dict(dec_ckpt["decoder"])
        if "decoder_ema" in dec_ckpt:
            decoder_trainer.ema_decoder.model.load_state_dict(dec_ckpt["decoder_ema"])
        print(f"      Loaded LatentDecoder from {dec_ckpt_path.name}")
    else:
        print(f"      Warning: No Phase 2 Decoder checkpoint found. Using initial weights.")

    # 6. Initialize Predictor Trainer
    print("\n[3/5] Initializing FlowMatchingPredictor Trainer...")
    predictor_config = Config()
    predictor_config.device = device
    predictor_config.dataset_path = config.dataset_path
    predictor_config.checkpoint_dir = config.checkpoint_dir / "predictor"
    predictor_config.output_path = config.output_path / "predictor"
    predictor_config.batch_size = config.batch_size
    predictor_config.effective_batch_size = config.effective_batch_size
    predictor_config._num_epochs = config._num_epochs
    predictor_config.curriculum = config.curriculum
    predictor_config.horizon = config.horizon
    predictor_config.history_length = config.history_length
    predictor_config.t_sampling_power = config.t_sampling_power
    predictor_config.cfg_dropout = config.cfg_dropout
    predictor_config.checkpoint_interval = config.checkpoint_interval
    predictor_config.use_prefix_history = True
    predictor_config.loss_vel_weight = config.loss_vel_weight

    # 64D Manifold Configuration
    predictor_config.latent_dim = args.latent_dim
    predictor_config.encoder_config.latent_dim = args.latent_dim
    predictor_config.decoder_config.latent_dim = args.latent_dim
    predictor_config.predictor_config.latent_dim = args.latent_dim

    predictor_trainer = FlowMatchingTrainer(
        config=predictor_config,
        pretrained_checkpoint_path=pretrain_ckpt_path,
        wandb_project="phase3_predictor" if args.wandb else None,
    )

    pred_ckpt_dir = predictor_config.checkpoint_dir
    pred_ckpt_dir.mkdir(parents=True, exist_ok=True)
    dec_ckpt_dir = decoder_config.checkpoint_dir
    dec_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 7. Setup optimization variables
    acc_steps = max(config.effective_batch_size // config.batch_size, 1)
    gradient_clip = float(config.gradient_clip)
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    steps_per_epoch = max(len(train_loader) // acc_steps, 1)
    total_steps = max(args.epochs * steps_per_epoch, 1)

    # Predictor optimizer & scaler
    pred_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    pred_optimizer = predictor_trainer.optimizer
    pred_scheduler = predictor_trainer.lr_scheduler

    # Decoder optimizer & scaler
    dec_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    dec_optimizer = decoder_trainer.optimizer
    dec_scheduler = decoder_trainer.lr_scheduler

    target_encoder = predictor_trainer.ema_encoder.model
    target_encoder.eval()

    # 8. Checkpoint loading or fresh start
    start_epoch = 1
    best_pred_val_loss = float("inf")
    best_dec_val_loss = float("inf")

    if args.resume_checkpoint:
        resume_p = Path(args.resume_checkpoint)
        if resume_p.exists():
            print(f"      Resuming Predictor checkpoint: {resume_p}")
            predictor_ckpt = torch.load(resume_p, map_location=device, weights_only=False)
            predictor_trainer.predictor.load_state_dict(predictor_ckpt["predictor"], strict=False)
            predictor_trainer.ema_predictor.model.load_state_dict(predictor_ckpt["predictor_ema"], strict=False)
            predictor_trainer.null_text_embedding.load_state_dict(predictor_ckpt["null_text_embedding"], strict=False)

            if "optimizer" in predictor_ckpt and pred_optimizer is not None:
                try:
                    pred_optimizer.load_state_dict(predictor_ckpt["optimizer"])
                except Exception as e:
                    print(f"      Warning: Could not load optimizer state: {e}")
            if "scaler" in predictor_ckpt and predictor_ckpt["scaler"] is not None and use_amp:
                try:
                    pred_scaler.load_state_dict(predictor_ckpt["scaler"])
                except Exception as e:
                    print(f"      Warning: Could not load scaler state: {e}")
            latent_stats_ckpt = predictor_ckpt.get("latent_stats")
            if latent_stats_ckpt is not None:
                if hasattr(latent_stats_ckpt, "state_dict"):
                    predictor_trainer.latent_stats.load_state_dict(latent_stats_ckpt.state_dict())
                else:
                    predictor_trainer.latent_stats.load_state_dict(latent_stats_ckpt)
                print("      Loaded latent normalization statistics from checkpoint.")
            else:
                print("      Checkpoint lacked latent_stats; computing from dataset...")
                predictor_trainer.latent_stats.initialized.copy_(torch.tensor(False, device=device))
                predictor_trainer._compute_latent_statistics()

            if args.reset_epoch:
                start_epoch = 1
                best_pred_val_loss = predictor_ckpt.get("best_val_loss", float("inf"))
                print(f"      [Fine-tune Mode] Resetting epoch to 1 for {args.epochs} fine-tuning epochs (Resumed Best Val: {best_pred_val_loss:.4f})")
                if args.lr is not None:
                    for g in pred_optimizer.param_groups:
                        g["lr"] = float(args.lr)
                    print(f"      Overriding optimizer learning rate to {args.lr}")
                total_steps = max(args.epochs * steps_per_epoch, 1)
                pred_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    pred_optimizer,
                    T_max=total_steps,
                    eta_min=1e-5,
                )
                predictor_trainer.lr_scheduler = pred_scheduler
            else:
                start_epoch = predictor_ckpt.get("epoch", 0) + 1
                best_pred_val_loss = predictor_ckpt.get("best_val_loss", float("inf"))
                print(f"      Resuming from epoch {start_epoch} (Previous Best Val: {best_pred_val_loss:.4f})")

                if "scheduler" in predictor_ckpt and pred_scheduler is not None:
                    old_num_epochs = predictor_ckpt.get("config", {}).get("_num_epochs", start_epoch) if isinstance(predictor_ckpt.get("config"), dict) else getattr(predictor_ckpt.get("config", None), "_num_epochs", start_epoch)
                    if args.epochs > old_num_epochs:
                        remaining_steps = max((args.epochs - start_epoch + 1) * steps_per_epoch, 1)
                        pred_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            pred_optimizer,
                            T_max=remaining_steps,
                            eta_min=1e-5,
                        )
                        predictor_trainer.lr_scheduler = pred_scheduler
                        print(f"      Extended training schedule: Re-anchored LR scheduler for {args.epochs - start_epoch + 1} remaining epochs ({remaining_steps} steps).")
                    else:
                        try:
                            pred_scheduler.load_state_dict(predictor_ckpt["scheduler"])
                        except Exception as e:
                            print(f"      Warning: Could not load scheduler state: {e}")
    else:
        print("      Training FlowMatchingPredictor from scratch with calibrated Track B architecture.")

    # Graceful interrupt handler: save checkpoint on Ctrl+C or kill
    def handle_interrupt(sig, frame):
        print(f"\n[!] Signal {sig} received. Saving emergency checkpoint before exit...")
        if train_predictor:
            emergency_pred_ckpt = pred_ckpt_dir / f"phase3_predictor_interrupted_{predictor_trainer.session_id}.pt"
            torch.save({
                "encoder": predictor_trainer.encoder.state_dict(),
                "encoder_ema": predictor_trainer.ema_encoder.state_dict(),
                "predictor": predictor_trainer.predictor.state_dict(),
                "predictor_ema": predictor_trainer.ema_predictor.state_dict(),
                "null_text_embedding": predictor_trainer.null_text_embedding.state_dict(),
                "latent_stats": predictor_trainer.latent_stats.state_dict(),
                "optimizer": pred_optimizer.state_dict(),
                "scaler": pred_scaler.state_dict() if use_amp else None,
                "scheduler": pred_scheduler.state_dict(),
                "epoch": current_epoch,
                "best_val_loss": best_pred_val_loss,
                "config": predictor_config.to_dict(),
            }, emergency_pred_ckpt)
            print(f"Saved emergency predictor checkpoint: {emergency_pred_ckpt}")

        if train_decoder:
            emergency_dec_ckpt = dec_ckpt_dir / f"phase3_decoder_interrupted_{decoder_trainer.session_id}.pt"
            torch.save({
                "encoder": decoder_trainer.encoder.state_dict(),
                "encoder_ema": decoder_trainer.ema_encoder.state_dict(),
                "decoder": decoder_trainer.decoder.state_dict(),
                "decoder_ema": decoder_trainer.ema_decoder.state_dict(),
                "optimizer": dec_optimizer.state_dict(),
                "scaler": dec_scaler.state_dict() if use_amp else None,
                "scheduler": dec_scheduler.state_dict(),
                "epoch": current_epoch,
                "best_val_loss": best_dec_val_loss,
                "config": decoder_config.to_dict(),
            }, emergency_dec_ckpt)
            print(f"Saved emergency decoder checkpoint: {emergency_dec_ckpt}")

        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    print(f"\n[4/5] Training Schedule:")
    print(f"      Accumulation Steps: {acc_steps}  |  Steps/Epoch: {steps_per_epoch}  |  Total Optimizer Steps: {total_steps}")
    print(f"      Mixed Precision   : {'Enabled (' + str(amp_dtype) + ')' if use_amp else 'Disabled'}")
    print(f"      Val Interval      : Every {config.val_interval} epochs")
    print(f"      Checkpoint Saves  : {pred_ckpt_dir}")

    # 9. Main Training Loop
    print("\n[5/5] Launching Training Loop...\n" + "-" * 70)
    current_epoch = start_epoch
    save_checkpoint_start_epoch = max(1, args.epochs - 100)

    for epoch in range(start_epoch, args.epochs + 1):
        current_epoch = epoch
        t_epoch_start = time.time()

        if train_decoder:
            decoder_trainer.decoder.train()
        else:
            decoder_trainer.decoder.eval()

        if train_predictor:
            predictor_trainer.predictor.train()
            predictor_trainer.null_text_embedding.train()
        else:
            predictor_trainer.predictor.eval()

        epoch_dec_loss = 0.0
        epoch_pred_loss = 0.0
        num_batches_dec = 0
        num_batches_pred = 0

        dec_optimizer.zero_grad(set_to_none=True)
        pred_optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            motion_raw = batch["motion"].to(device)
            history_raw = batch["history_motion"].to(device)
            text_raw = batch["text_clip"].to(device) if batch.get("text_clip") is not None else None
            joints = batch["joints"].to(device) if "joints" in batch else None
            B = motion_raw.shape[0]

            target_motion = normalizer.normalize(motion_raw)
            history_motion = normalizer.normalize(history_raw)
            text_emb = text_raw[:, -1, :] if (text_raw is not None and text_raw.ndim == 3) else text_raw

            with torch.no_grad():
                zeros_target = torch.zeros(B, 512, device=device, dtype=target_motion.dtype)
                zeros_hist = torch.zeros(B, 512, device=device, dtype=target_motion.dtype)

                raw_target = target_encoder(target_motion, zeros_target, mask=None, return_layer_outputs=True)
                hidden_target = raw_target[:, 1:, -1, :]
                if hasattr(target_encoder, "get_latent"):
                    z1 = target_encoder.get_latent(hidden_target).detach()
                else:
                    z1 = hidden_target.detach()

                raw_hist = target_encoder(history_motion, zeros_hist, mask=None, return_layer_outputs=False)
                if hasattr(target_encoder, "get_latent"):
                    track_features = target_encoder.get_latent(raw_hist).detach()
                else:
                    track_features = raw_hist.detach()

            # ── STEP 1: Decoder Training (if enabled) ──
            if train_decoder and joints is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    decoded = decoder_trainer.decoder(z1)
                    dec_loss = compute_decoder_loss(decoded, target_motion, joints, normalizer)

                dec_scaler.scale(dec_loss / acc_steps).backward()

                if (step + 1) % acc_steps == 0:
                    dec_scaler.unscale_(dec_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        decoder_trainer.decoder.parameters(),
                        max_norm=gradient_clip,
                    )
                    dec_scaler.step(dec_optimizer)
                    dec_scaler.update()
                    dec_optimizer.zero_grad(set_to_none=True)
                    decoder_trainer.ema_decoder.update(decoder_trainer.decoder)
                    dec_scheduler.step()

                epoch_dec_loss += dec_loss.item()
                num_batches_dec += 1

            # ── STEP 2: Predictor Training (if enabled) ──
            if train_predictor:
                if "captions" in batch:
                    text_clip = predictor_trainer.clip_encoder.encode_sequence(batch["captions"])
                elif text_raw is not None:
                    text_clip = text_raw.to(device)
                else:
                    text_clip = None

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred_loss, _ = predictor_trainer.compute_flow_loss(
                        target_motion=target_motion,
                        history_motion=history_motion,
                        text_emb=text_clip,
                        is_training=True,
                        z1=z1,
                        track_features=track_features,
                        history_valid_length=batch["history_valid_length"].to(device) if "history_valid_length" in batch and isinstance(batch["history_valid_length"], torch.Tensor) else None,
                    )

                pred_scaler.scale(pred_loss / acc_steps).backward()

                if (step + 1) % acc_steps == 0:
                    pred_scaler.unscale_(pred_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(predictor_trainer.predictor.parameters()) +
                        list(predictor_trainer.null_text_embedding.parameters()),
                        max_norm=gradient_clip,
                    )
                    pred_scaler.step(pred_optimizer)
                    pred_scaler.update()
                    pred_optimizer.zero_grad(set_to_none=True)
                    predictor_trainer.ema_predictor.update(predictor_trainer.predictor)
                    pred_scheduler.step()

                epoch_pred_loss += pred_loss.item()
                num_batches_pred += 1

        avg_dec = epoch_dec_loss / max(num_batches_dec, 1) if num_batches_dec > 0 else 0.0
        avg_pred = epoch_pred_loss / max(num_batches_pred, 1) if num_batches_pred > 0 else 0.0

        dec_lr_str = f"{dec_scheduler.get_last_lr()[0]:.2e}" if train_decoder else "PAUSED"
        pred_lr_str = f"{pred_scheduler.get_last_lr()[0]:.2e}" if train_predictor else "PAUSED"
        sec_per_epoch = time.time() - t_epoch_start

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"dec_loss={avg_dec:.4f} | pred_loss={avg_pred:.4f} | "
            f"dec_lr={dec_lr_str} | pred_lr={pred_lr_str} | "
            f"time={sec_per_epoch:.1f}s"
        )

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % config.val_interval == 0:
            decoder_trainer.decoder.eval()
            decoder_trainer.ema_decoder.model.eval()
            predictor_trainer.ema_predictor.model.eval()

            val_pred_loss = 0.0
            val_dec_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for vbatch in val_loader:
                    vmotion_raw = vbatch["motion"].to(device)
                    vhistory_raw = vbatch["history_motion"].to(device)
                    vtext_raw = vbatch.get("text_clip", None)
                    vjoints = vbatch.get("joints", None)

                    vtarget = normalizer.normalize(vmotion_raw)
                    vhistory = normalizer.normalize(vhistory_raw)
                    vB = vtarget.shape[0]

                    vz_text = torch.zeros(vB, 512, device=device, dtype=vtarget.dtype)
                    vh_text = torch.zeros(vB, 512, device=device, dtype=vtarget.dtype)

                    vraw_target = target_encoder(vtarget, vz_text, mask=None, return_layer_outputs=True)
                    vhidden_target = vraw_target[:, 1:, -1, :]
                    if hasattr(target_encoder, "get_latent"):
                        vz1 = target_encoder.get_latent(vhidden_target).detach()
                    else:
                        vz1 = vhidden_target.detach()

                    vraw_hist = target_encoder(vhistory, vh_text, mask=None, return_layer_outputs=False)
                    if hasattr(target_encoder, "get_latent"):
                        vtrack_features = target_encoder.get_latent(vraw_hist).detach()
                    else:
                        vtrack_features = vraw_hist.detach()

                    # Decoder validation loss
                    if vjoints is not None:
                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            vdecoded = decoder_trainer.ema_decoder.model(vz1)
                            vdec_l = compute_decoder_loss(vdecoded, vtarget, vjoints.to(device), normalizer)
                        val_dec_loss += vdec_l.item()

                    # Predictor validation loss
                    if "captions" in vbatch:
                        vtext_clip = predictor_trainer.clip_encoder.encode_sequence(vbatch["captions"])
                    elif vtext_raw is not None:
                        vtext_clip = vtext_raw.to(device)
                    else:
                        vtext_clip = None

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        vpred_l, _ = predictor_trainer.compute_flow_loss(
                            vtarget, vhistory, vtext_clip,
                            is_training=False,
                            z1=vz1,
                            track_features=vtrack_features,
                            history_valid_length=vbatch["history_valid_length"].to(device) if "history_valid_length" in vbatch and isinstance(vbatch["history_valid_length"], torch.Tensor) else None,
                        )
                    val_pred_loss += vpred_l.item()

                    val_batches += 1
                    if val_batches >= config.val_batches:
                        break

            avg_vdec = val_dec_loss / max(val_batches, 1)
            avg_vpred = val_pred_loss / max(val_batches, 1)
            print(f"          Val | dec_val={avg_vdec:.4f} | pred_val={avg_vpred:.4f}")

            # Save best decoder checkpoint ONLY if training AND during final 100 epochs
            if avg_vdec < best_dec_val_loss:
                best_dec_val_loss = avg_vdec
                if train_decoder and epoch >= save_checkpoint_start_epoch:
                    dec_ckpt = dec_ckpt_dir / f"phase3_decoder_best_val_{decoder_trainer.session_id}.pt"
                    torch.save({
                        "encoder": decoder_trainer.encoder.state_dict(),
                        "encoder_ema": decoder_trainer.ema_encoder.state_dict(),
                        "decoder": decoder_trainer.decoder.state_dict(),
                        "decoder_ema": decoder_trainer.ema_decoder.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_dec_val_loss,
                        "config": decoder_config.to_dict(),
                    }, dec_ckpt)
                    print(f"          ✓ New best decoder checkpoint: {dec_ckpt.name}")

            # Save best predictor checkpoint
            if avg_vpred < best_pred_val_loss:
                best_pred_val_loss = avg_vpred
                if train_predictor:
                    best_ckpt_path = pred_ckpt_dir / f"phase3_predictor_best_val_{predictor_trainer.session_id}.pt"
                    torch.save({
                        "encoder": predictor_trainer.encoder.state_dict(),
                        "encoder_ema": predictor_trainer.ema_encoder.state_dict(),
                        "predictor": predictor_trainer.predictor.state_dict(),
                        "predictor_ema": predictor_trainer.ema_predictor.state_dict(),
                        "null_text_embedding": predictor_trainer.null_text_embedding.state_dict(),
                        "latent_stats": predictor_trainer.latent_stats.state_dict(),
                        "optimizer": pred_optimizer.state_dict(),
                        "scaler": pred_scaler.state_dict() if use_amp else None,
                        "scheduler": pred_scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_pred_val_loss,
                        "config": predictor_config.to_dict(),
                    }, best_ckpt_path)
                    print(f"          ✓ New best predictor checkpoint: {best_ckpt_path.name}")

        # Periodic checkpoint save
        if train_predictor and (epoch % config.checkpoint_interval == 0):
            periodic_ckpt = pred_ckpt_dir / f"phase3_predictor_epoch_{epoch}_{predictor_trainer.session_id}.pt"
            torch.save({
                "encoder": predictor_trainer.encoder.state_dict(),
                "encoder_ema": predictor_trainer.ema_encoder.state_dict(),
                "predictor": predictor_trainer.predictor.state_dict(),
                "predictor_ema": predictor_trainer.ema_predictor.state_dict(),
                "null_text_embedding": predictor_trainer.null_text_embedding.state_dict(),
                "latent_stats": predictor_trainer.latent_stats.state_dict(),
                "optimizer": pred_optimizer.state_dict(),
                "scaler": pred_scaler.state_dict() if use_amp else None,
                "scheduler": pred_scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_pred_val_loss,
                "config": predictor_config.to_dict(),
            }, periodic_ckpt)
            print(f"          ✓ Periodic checkpoint saved: {periodic_ckpt.name}")

        # Always update latest checkpoint
        if train_predictor:
            latest_pred_ckpt = pred_ckpt_dir / f"phase3_predictor_latest_{predictor_trainer.session_id}.pt"
            torch.save({
                "encoder": predictor_trainer.encoder.state_dict(),
                "encoder_ema": predictor_trainer.ema_encoder.state_dict(),
                "predictor": predictor_trainer.predictor.state_dict(),
                "predictor_ema": predictor_trainer.ema_predictor.state_dict(),
                "null_text_embedding": predictor_trainer.null_text_embedding.state_dict(),
                "latent_stats": predictor_trainer.latent_stats.state_dict(),
                "optimizer": pred_optimizer.state_dict(),
                "scaler": pred_scaler.state_dict() if use_amp else None,
                "scheduler": pred_scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_pred_val_loss,
                "config": predictor_config.to_dict(),
            }, latest_pred_ckpt)

        if train_decoder:
            latest_dec_ckpt = dec_ckpt_dir / f"phase3_decoder_latest_{decoder_trainer.session_id}.pt"
            torch.save({
                "encoder": decoder_trainer.encoder.state_dict(),
                "encoder_ema": decoder_trainer.ema_encoder.state_dict(),
                "decoder": decoder_trainer.decoder.state_dict(),
                "decoder_ema": decoder_trainer.ema_decoder.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_dec_val_loss,
                "config": decoder_config.to_dict(),
            }, latest_dec_ckpt)

    print("\n" + "=" * 70)
    print("Phase 3 Training Complete!")
    print(f"Best Decoder   Val Loss: {best_dec_val_loss:.4f}")
    print(f"Best Predictor Val Loss: {best_pred_val_loss:.4f}")
    print(f"Predictor Checkpoints   : {pred_ckpt_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
