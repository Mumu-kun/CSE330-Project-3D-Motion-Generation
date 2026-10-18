#!/usr/bin/env python3
"""Phase 2 LatentDecoder Training Script (64D Manifold -> 72D Kinematic Bijection).

Extracts and standalone-executes the Phase 2 LatentDecoder training loop.
Loads a pretrained MotionHistoryEncoder (64D manifold), freezes it completely,
and trains a fresh LatentDecoder to reconstruct 3D joint positions and kinematics
from ground-truth 64D manifold latents.

Provides:
  - Transparent, native PyTorch training and validation loops with rich progress metrics
  - Oracle MPJPE (mm) computation on every validation split
  - 64D Geodesic manifold compatibility with get_latent projection
  - Gradient accumulation, Mixed Precision (AMP), gradient clipping, and EMA tracking
  - Emergency checkpoint preservation on SIGINT / SIGTERM signals
  - Seamless resume capability from any checkpoint (--resume_checkpoint)

Usage:
    # 1. Edit the CONFIGURATION & HYPERPARAMETERS section below.
    # 2. Run in foreground on GPU:
        python train_decoder.py

    # 3. Or run in background with logging:
        nohup python train_decoder.py > decoder_training.log 2>&1 &
        tail -f decoder_training.log

    # 4. Or override parameters via CLI:
        python train_decoder.py --epochs 500 --batch_size 128 --effective_batch_size 256 --latent_dim 64
"""

# =============================================================================
# OPTIONABLE VARIABLES & HYPERPARAMETERS
# (Edit values here directly, or override via CLI arguments)
# =============================================================================

# --- Training & Optimization ---
NUM_EPOCHS: int = 500                # Total decoder training epochs
BATCH_SIZE: int = 128                # Micro-batch size per forward/backward step
EFFECTIVE_BATCH_SIZE: int = 256      # Effective batch size via gradient accumulation (256 // 128 = 2 steps)
LEARNING_RATE: float = 1e-4          # AdamW learning rate
WEIGHT_DECAY: float = 1e-5           # Weight decay for AdamW
GRADIENT_CLIP: float = 30.0          # Max gradient norm clipping threshold
EMA_DECAY: float = 0.999             # Exponential moving average decay for LatentDecoder

# --- Motion Sequences & Context ---
HORIZON: int = 80                    # Target motion window length (frames at 20fps = 4.0s)
HISTORY_LENGTH: int = 40             # Historical context length (frames at 20fps = 2.0s)

# --- 64D Latent Manifold ---
LATENT_DIM: int = 64                 # Latent manifold dimensionality (exact physical kinematic rank)

# --- Validation & Checkpoint Frequencies ---
VAL_INTERVAL: int = 25               # Evaluate on validation split every N epochs
CHECKPOINT_INTERVAL: int = 50        # Save periodic checkpoints every N epochs

# --- Checkpoints & Resume ---
# Set RESUME_CHECKPOINT to a .pt path to resume training (e.g. "./checkpoints/phase3/decoder/finetune_latest_*.pt")
RESUME_CHECKPOINT: str | None = None

# Explicit path to Phase 1 encoder checkpoint (leave None for automatic discovery in ./checkpoints/pretrain)
PRETRAIN_CHECKPOINT: str | None = None

# --- Paths & Logging ---
# Paths can be relative (to project root or src) or absolute. Auto-detected if None.
DATASET_PATH: str | None = None       # Defaults to ./dataset/humanml3d-subset or src/dataset/humanml3d-subset
CHECKPOINT_DIR: str | None = None     # Defaults to ./checkpoints/phase3/decoder
OUTPUT_PATH: str | None = None        # Defaults to ./output/phase3/decoder
LOG_FILE: str | None = "decoder_training.log"  # Log file for background logging

# --- Hardware & Tracking ---
DEVICE: str = "cuda"                  # "cuda" or "cpu"
USE_WANDB: bool = False               # Set True to enable Weights & Biases logging
WANDB_PROJECT: str = "phase2_decoder" # W&B project name if enabled

# =============================================================================
# END OF USER OPTIONS
# =============================================================================

import argparse
from datetime import datetime, timezone, timedelta
import gc
import math
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
from utils.models import _find_latest_checkpoint, EMAModel
from utils.models.motion_history_encoder import MotionHistoryEncoder
from utils.models.flow_matching_predictor import LatentDecoder
from utils.motion_utils import (
    FeatureNormalizer,
    Features,
    x271_to_x72,
    x72_to_positions,
)


def compute_decoder_loss_and_metrics(
    decoded: torch.Tensor,
    motion: torch.Tensor,
    joints: torch.Tensor,
    normalizer: FeatureNormalizer,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruction loss for 72D LatentDecoder with direct RIC position prediction.

    Computes:
      - Root vertical velocity (loss_root_y)
      - Root horizontal velocities (loss_root_xz)
      - Root rotational angle velocity / yaw (loss_yaw)
      - Relative-to-root joint 3D positions RIC (loss_ric)
      - Foot-ground contact logits (loss_contact)
      - 3D joint positions MSE (loss_joint)
      - Oracle MPJPE in millimeters (mpjpe_mm)
    """
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
    target_positions = joints[:, 1:, :].flatten(0, 1)
    loss_joint = F.mse_loss(dec_positions, target_positions)

    # Oracle MPJPE calculation (in millimeters)
    pos_pred = dec_positions.view(-1, 22, 3)
    pos_tgt = target_positions.view(-1, 22, 3)
    mpjpe_mm = torch.norm(pos_pred - pos_tgt, dim=-1).mean().item() * 1000.0

    total_loss = (
        1.0 * loss_root_y
        + 1.0 * loss_root_xz
        + 1.0 * loss_yaw
        + 1.0 * loss_ric
        + 0.5 * loss_contact
        + 1.0 * loss_joint
    )

    metrics = {
        "loss": total_loss.item(),
        "root_y": loss_root_y.item(),
        "root_xz": loss_root_xz.item(),
        "yaw": loss_yaw.item(),
        "ric": loss_ric.item(),
        "contact": loss_contact.item(),
        "joint": loss_joint.item(),
        "mpjpe_mm": mpjpe_mm,
    }
    return total_loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 LatentDecoder Training Script (64D Manifold -> 72D Bijection)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--effective_batch_size", type=int, default=EFFECTIVE_BATCH_SIZE, help=f"Effective batch size (default: {EFFECTIVE_BATCH_SIZE})")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help=f"Weight decay (default: {WEIGHT_DECAY})")
    parser.add_argument("--gradient_clip", type=float, default=GRADIENT_CLIP, help=f"Gradient clipping max norm (default: {GRADIENT_CLIP})")
    parser.add_argument("--horizon", type=int, default=HORIZON, help=f"Motion horizon in frames (default: {HORIZON})")
    parser.add_argument("--history_length", type=int, default=HISTORY_LENGTH, help=f"History length in frames (default: {HISTORY_LENGTH})")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM, help=f"Latent manifold dimensionality (default: {LATENT_DIM})")
    parser.add_argument("--val_interval", type=int, default=VAL_INTERVAL, help=f"Validation interval in epochs (default: {VAL_INTERVAL})")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL, help=f"Periodic checkpoint save interval (default: {CHECKPOINT_INTERVAL})")
    parser.add_argument("--resume_checkpoint", type=str, default=RESUME_CHECKPOINT, help="Path to decoder checkpoint to resume from (default: None)")
    parser.add_argument("--pretrain_checkpoint", type=str, default=PRETRAIN_CHECKPOINT, help="Path to pretrained encoder checkpoint (default: auto-detect)")
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
    session_id = datetime.now(timezone(timedelta(hours=6))).strftime("%Y%m%d_%H%M%S")

    print("=" * 75)
    print("      Phase 2 LatentDecoder Training (64D Manifold -> 72D Bijection)")
    print("=" * 75)
    print(f"Start Time         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Target Epochs      : {args.epochs}")
    print(f"Batch Size         : {args.batch_size} (Effective: {args.effective_batch_size})")
    print(f"Learning Rate      : {args.learning_rate}")
    print(f"Motion Horizon     : {args.horizon} frames ({args.horizon / 20.0:.1f}s)")
    print(f"History Context    : {args.history_length} frames ({args.history_length / 20.0:.1f}s)")
    print(f"Latent Manifold    : {args.latent_dim}D -> 72D Kinematic Bijection")
    print(f"Val Interval       : Every {args.val_interval} epochs")
    print(f"Periodic Save      : Every {args.checkpoint_interval} epochs")
    print("=" * 75)

    # 1. Resolve dataset and checkpoint paths
    dataset_candidates = [
        Path(args.dataset_path) if args.dataset_path else None,
        SRC_DIR / "dataset" / "humanml3d-subset",
        ROOT_DIR / "dataset" / "humanml3d-subset",
        Path("./dataset/humanml3d-subset"),
        Path("/kaggle/input/humanml3d-subset"),
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
        SRC_DIR / "checkpoints" / "phase3" / "decoder" if (SRC_DIR / "checkpoints").exists()
        else ROOT_DIR / "checkpoints" / "phase3" / "decoder"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output_path) if args.output_path else (
        SRC_DIR / "output" / "phase3" / "decoder" if (SRC_DIR / "output").exists()
        else ROOT_DIR / "output" / "phase3" / "decoder"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Find Pretrained Encoder Checkpoint
    if args.pretrain_checkpoint:
        pretrain_ckpt_path = Path(args.pretrain_checkpoint)
        if not pretrain_ckpt_path.exists():
            raise FileNotFoundError(f"Specified pretrain checkpoint does not exist: {pretrain_ckpt_path}")
    else:
        pretrain_candidates = [
            SRC_DIR / "checkpoints" / "pretrain",
            ROOT_DIR / "checkpoints" / "pretrain",
            Path("./checkpoints/pretrain"),
            Path("/kaggle/input/3mogenpretrain"),
        ]
        pretrain_ckpt_path = None
        for cand in pretrain_candidates:
            if cand.exists():
                try:
                    pretrain_ckpt_path = _find_latest_checkpoint(cand, prefix="pretrain_best_val")
                    if pretrain_ckpt_path is not None:
                        break
                except Exception:
                    pass

        if pretrain_ckpt_path is None:
            pretrain_ckpt_path = SRC_DIR / "checkpoints" / "pretrain" / "pretrain_best_val_20260620_205707_6630.pt"
            if not pretrain_ckpt_path.exists():
                raise FileNotFoundError(
                    "Could not find any pretrained encoder checkpoint matching 'pretrain_best_val_*.pt'. "
                    "Please run Phase 1 pretraining (train_pretrain.py) first, or specify --pretrain_checkpoint."
                )

    print(f"\n[1/5] Dataset location   : {dataset_path}")
    print(f"      Encoder checkpoint : {pretrain_ckpt_path}")
    print(f"      Checkpoints dir    : {checkpoint_dir}")
    print(f"      Outputs dir        : {output_path}")

    # 3. Setup Configuration
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
    config.history_length = args.history_length
    config.val_interval = args.val_interval
    config.checkpoint_interval = args.checkpoint_interval

    # 64D Manifold Configuration
    config.latent_dim = args.latent_dim
    config.encoder_config.latent_dim = args.latent_dim
    config.decoder_config.latent_dim = args.latent_dim
    config.predictor_config.latent_dim = args.latent_dim

    # 4. Create DataLoaders
    print(f"\n[2/5] Creating DataLoaders (horizon={args.horizon})...")
    train_loader, normalizer = create_dataloader(config, "train", shuffle=True)
    val_loader, _ = create_dataloader(config, "val", shuffle=False)
    train_loader.dataset.set_horizon(config.horizon)
    val_loader.dataset.set_horizon(config.horizon)
    print(f"      Train Batches: {len(train_loader)}  |  Val Batches: {len(val_loader)}")

    # 5. Clean VRAM and Build Models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[3/5] Loading and freezing 64D MotionHistoryEncoder...")
    encoder = MotionHistoryEncoder(config).to(device)
    enc_ckpt = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
    enc_state = enc_ckpt.get("encoder_ema") or enc_ckpt.get("encoder")
    if enc_state is None:
        raise KeyError(f"Checkpoint at {pretrain_ckpt_path} contains neither 'encoder_ema' nor 'encoder' keys.")
    encoder.load_state_dict(enc_state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"      Encoder frozen successfully ({sum(p.numel() for p in encoder.parameters()):,} parameters).")

    print(f"\n      Initializing fresh LatentDecoder (64D -> 72D)...")
    decoder = LatentDecoder(config).to(device)
    ema_decoder = EMAModel(decoder, decay=EMA_DECAY).to(device)
    print(f"      Decoder parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad):,} trainable.")

    # 6. Setup Optimizer, Scaler, and Scheduler
    acc_steps = max(config.effective_batch_size // config.batch_size, 1)
    gradient_clip = float(config.gradient_clip)
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    steps_per_epoch = max(len(train_loader) // acc_steps, 1)
    total_optimizer_steps = max(args.epochs * steps_per_epoch, 1)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_optimizer_steps,
        eta_min=float(args.learning_rate) * 0.01,
    )

    # 7. Checkpoint Resuming or Fresh Start
    start_epoch = 1
    best_dec_val_loss = float("inf")
    best_mpjpe_mm = float("inf")

    if args.resume_checkpoint:
        resume_p = Path(args.resume_checkpoint)
        if resume_p.exists():
            print(f"\n      Resuming Decoder from checkpoint: {resume_p}")
            resume_ckpt = torch.load(resume_p, map_location=device, weights_only=False)
            if "decoder" in resume_ckpt:
                decoder.load_state_dict(resume_ckpt["decoder"])
            if "decoder_ema" in resume_ckpt:
                ema_decoder.model.load_state_dict(resume_ckpt["decoder_ema"])
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            if "scaler" in resume_ckpt and use_amp:
                scaler.load_state_dict(resume_ckpt["scaler"])
            if "scheduler" in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            start_epoch = resume_ckpt.get("epoch", 0) + 1
            best_dec_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
            best_mpjpe_mm = resume_ckpt.get("best_mpjpe", float("inf"))
            print(f"      Resuming from epoch {start_epoch} (Best Val: {best_dec_val_loss:.4f}, Best MPJPE: {best_mpjpe_mm:.1f} mm)")
        else:
            print(f"      Warning: Resume checkpoint {resume_p} not found. Starting from scratch.")
    else:
        print("      Training LatentDecoder from scratch.")

    # 8. Optional W&B
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"decoder_64d_{session_id}",
                config=vars(args),
            )
        except Exception as e:
            print(f"      Warning: Failed to initialize W&B: {e}")

    # 9. Emergency Signal Handler
    def handle_interrupt(sig, frame):
        print(f"\n[!] Signal {sig} received. Saving emergency checkpoint before exit...")
        emergency_ckpt = checkpoint_dir / f"finetune_interrupted_{session_id}.pt"
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "decoder_ema": ema_decoder.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "scheduler": scheduler.state_dict(),
            "epoch": current_epoch,
            "best_val_loss": best_dec_val_loss,
            "best_mpjpe": best_mpjpe_mm,
            "config": config.to_dict() if hasattr(config, "to_dict") else vars(config),
        }, emergency_ckpt)
        print(f"      Saved emergency checkpoint: {emergency_ckpt}")
        if wandb_run:
            wandb_run.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    print(f"\n[4/5] Training Schedule:")
    print(f"      Accumulation Steps: {acc_steps}  |  Steps/Epoch: {steps_per_epoch}  |  Total Optimizer Steps: {total_optimizer_steps}")
    print(f"      Mixed Precision   : {'Enabled (' + str(amp_dtype) + ')' if use_amp else 'Disabled'}")
    print(f"      Val Interval      : Every {config.val_interval} epochs")
    print(f"      Checkpoints Dir   : {checkpoint_dir}")

    # 10. Main Training Loop
    print("\n[5/5] Launching LatentDecoder Training Loop...\n" + "-" * 75)
    current_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs + 1):
        current_epoch = epoch
        t_epoch_start = time.time()
        decoder.train()

        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            motion_raw = batch["motion"].to(device)
            joints = batch["joints"].to(device) if "joints" in batch else None
            B = motion_raw.shape[0]

            if joints is None:
                continue

            target_motion = normalizer.normalize(motion_raw)

            # Extract 64D manifold latents through frozen encoder
            with torch.no_grad():
                zeros_text = torch.zeros(B, 512, device=device, dtype=target_motion.dtype)
                raw_target = encoder(target_motion, zeros_text, mask=None, return_layer_outputs=True)
                hidden_target = raw_target[:, 1:, -1, :]
                if hasattr(encoder, "get_latent"):
                    z1 = encoder.get_latent(hidden_target).detach()
                else:
                    z1 = hidden_target.detach()

            # Decoder Forward & Loss
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                decoded = decoder(z1)
                dec_loss, _ = compute_decoder_loss_and_metrics(decoded, target_motion, joints, normalizer)

            scaler.scale(dec_loss / acc_steps).backward()

            if (step + 1) % acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(),
                    max_norm=gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_decoder.update(decoder)
                scheduler.step()

            epoch_loss += dec_loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        epoch_sec = time.time() - t_epoch_start
        current_lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate

        # Print training line
        print(f"Epoch [{epoch:4d}/{args.epochs}] ({epoch_sec:4.1f}s) | Loss: {avg_train_loss:.4f} | lr: {current_lr:.2e}")

        # Periodic Validation
        if epoch % config.val_interval == 0 or epoch == args.epochs:
            ema_decoder.model.eval()
            val_loss_acc = 0.0
            val_mpjpe_acc = 0.0
            val_batches = 0

            with torch.no_grad():
                for vbatch in val_loader:
                    vmotion_raw = vbatch["motion"].to(device)
                    vjoints = vbatch.get("joints", None)
                    if vjoints is None:
                        continue

                    vtarget = normalizer.normalize(vmotion_raw)
                    vB = vtarget.shape[0]

                    vz_text = torch.zeros(vB, 512, device=device, dtype=vtarget.dtype)
                    vraw_target = encoder(vtarget, vz_text, mask=None, return_layer_outputs=True)
                    vhidden = vraw_target[:, 1:, -1, :]
                    if hasattr(encoder, "get_latent"):
                        vz1 = encoder.get_latent(vhidden).detach()
                    else:
                        vz1 = vhidden.detach()

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        vdecoded = ema_decoder.model(vz1)
                        v_loss, v_metrics = compute_decoder_loss_and_metrics(vdecoded, vtarget, vjoints.to(device), normalizer)

                    val_loss_acc += v_metrics["loss"]
                    val_mpjpe_acc += v_metrics["mpjpe_mm"]
                    val_batches += 1

            avg_val_loss = val_loss_acc / max(val_batches, 1)
            avg_val_mpjpe = val_mpjpe_acc / max(val_batches, 1)

            print(f"          Val | dec_loss={avg_val_loss:.4f} | Oracle MPJPE: {avg_val_mpjpe:.1f} mm | Best Loss: {best_dec_val_loss:.4f} | Best MPJPE: {best_mpjpe_mm:.1f} mm")

            # Save best validation checkpoint (overwriting single best checkpoint)
            if avg_val_loss < best_dec_val_loss:
                best_dec_val_loss = avg_val_loss
                best_mpjpe_mm = min(best_mpjpe_mm, avg_val_mpjpe)
                best_ckpt_file = checkpoint_dir / f"finetune_best_val_{session_id}.pt"
                torch.save({
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "decoder_ema": ema_decoder.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_dec_val_loss,
                    "best_mpjpe": best_mpjpe_mm,
                    "config": config.to_dict() if hasattr(config, "to_dict") else vars(config),
                }, best_ckpt_file)
                print(f"          ✓ New best decoder checkpoint: {best_ckpt_file.name}")

            if wandb_run:
                wandb_run.log({
                    "train/loss": avg_train_loss,
                    "train/lr": current_lr,
                    "val/loss": avg_val_loss,
                    "val/oracle_mpjpe_mm": avg_val_mpjpe,
                    "epoch": epoch,
                })

        # Always update latest checkpoint
        latest_file = checkpoint_dir / f"finetune_latest_{session_id}.pt"
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "decoder_ema": ema_decoder.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_dec_val_loss,
            "best_mpjpe": best_mpjpe_mm,
            "config": config.to_dict() if hasattr(config, "to_dict") else vars(config),
        }, latest_file)

    print("\n" + "=" * 75)
    print(f"Decoder training complete across {args.epochs} epochs!")
    print(f"Best Validation Loss : {best_dec_val_loss:.4f}")
    print(f"Best Oracle MPJPE    : {best_mpjpe_mm:.1f} mm (Target: < 42.0 mm)")
    print(f"Checkpoints saved in : {checkpoint_dir}")
    print("=" * 75)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
