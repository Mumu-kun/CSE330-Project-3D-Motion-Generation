"""
Training utilities for Motion History Encoder and Flow Matching Predictor.

Implements a progressive horizon curriculum training approach:
- Stage 1: 16 frames
- Stage 2: 32 frames
- Stage 3: 64 frames
- Stage 4: 128 frames (optional)

Key features:
- Full teacher forcing (no scheduled sampling)
- Fixed learning rate (no scheduling)
- EMA for validation and checkpointing
- CFG dropout for conditional generation
"""

import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union, Tuple
from tqdm import tqdm

from utils.wandb_logger import WandbLogger
from utils.motion_utils import (
    FeatureNormalizer,
    RootPositionTracker,
    flow_output_to_271d,
)


# =============================================================================
# Feature Extraction Helpers
# =============================================================================


def extract_prev_frame_features(frame: torch.Tensor) -> torch.Tensor:
    """
    Extract 261D prev_frame_features from 271D frame.

    271D Input Format:
        [0:3]     Root height Y, Root velocity X, Root velocity Z
        [3:69]    RIC positions (22 x 3)
        [69:201]  6D rotations (22 x 6)
        [201:267] Local velocities (22 x 3)
        [267:271] Foot contacts (4D)

    261D Output Format:
        [0:9]     Root features: height(1) + velocity(2) + rotation_6d(6)
        [9:261]   Joint features: 21 joints x 12D = 252D
                  Per joint: RIC(3) + rotation_6d(6) + velocity(3) = 12D

    Args:
        frame: (B, 271) single frame

    Returns:
        prev_features: (B, 261)
    """
    B = frame.shape[0]

    # Root features (9D)
    root_height = frame[:, 0:1]  # height_y
    root_vel = frame[:, 1:3]  # vel_x, vel_z
    root_rot6d = frame[:, 69:75]  # root rotation_6d
    prev_root = torch.cat([root_height, root_vel, root_rot6d], dim=-1)  # (B, 9)

    # Joint features (252D): 21 non-root joints
    # RIC positions: [6:69] for 21 joints (skip root at [3:6])
    joint_ric = frame[:, 6:69]  # 21 x 3 = 63D
    # Rotations: [75:201] for 21 joints (skip root at [69:75])
    joint_rot6d = frame[:, 75:201]  # 21 x 6 = 126D
    # Velocities: [204:267] for 21 joints (skip root at [201:204])
    joint_vel = frame[:, 204:267]  # 21 x 3 = 63D
    prev_joints = torch.cat([joint_ric, joint_rot6d, joint_vel], dim=-1)  # (B, 252)

    return torch.cat([prev_root, prev_joints], dim=-1)  # (B, 261)


def extract_clean_target(frame: torch.Tensor) -> torch.Tensor:
    """
    Extract 72D clean target from 271D frame.

    72D Output Format:
        [0:9]     Root features: height(1) + velocity(2) + rotation_6d(6)
        [9:72]    Joint RIC positions: 21 joints x 3D = 63D

    Args:
        frame: (B, 271) single frame

    Returns:
        target: (B, 72)
    """
    # Root features (9D)
    root_height = frame[:, 0:1]
    root_vel = frame[:, 1:3]
    root_rot6d = frame[:, 69:75]
    root_features = torch.cat([root_height, root_vel, root_rot6d], dim=-1)  # (B, 9)

    # Joint RIC positions (63D)
    joint_ric = frame[:, 6:69]  # 21 joints x 3D

    return torch.cat([root_features, joint_ric], dim=-1)  # (B, 72)


# =============================================================================
# EMA Model Management
# =============================================================================


class EMAModel:
    """
    Exponential Moving Average model wrapper.

    Maintains an EMA copy of a model for more stable evaluation.
    EMA is only used for validation, sampling, and checkpointing.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA model.

        Args:
            model: The model to create EMA copy of
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def update(self, model: nn.Module) -> None:
        """
        Update EMA weights.

        Args:
            model: The source model to update from
        """
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device: str) -> "EMAModel":
        """Move EMA model to device."""
        self.model.to(device)
        return self


# =============================================================================
# Training Function
# =============================================================================


def train(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    save_dir: str,
    horizon: int = 16,
    device: str = "cuda",
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    ema_decay: float = 0.999,
    cfg_dropout: float = 0.1,
    clip_encoder: Optional[nn.Module] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_from: Optional[str] = None,
    normalizer: Optional["FeatureNormalizer"] = None,
) -> Tuple[EMAModel, EMAModel]:
    """
    Training loop with progressive horizon curriculum.

    Key features:
    - Full teacher forcing (no scheduled sampling)
    - Fixed learning rate (no scheduling)
    - EMA for validation and checkpointing
    - CFG dropout for classifier-free guidance capability

    Args:
        encoder: MotionHistoryEncoder model
        predictor: FlowMatchingPredictor model
        dataloader: DataLoader providing HumanML3D batches (RAW features)
        num_epochs: Total training epochs
        save_dir: Directory to save checkpoints
        horizon: Current stage horizon (sequence length)
        device: Device to train on
        lr: Learning rate (fixed)
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        ema_decay: EMA decay rate
        cfg_dropout: Dropout probability for classifier-free guidance
        clip_encoder: Optional CLIPEncoder for encoding raw captions
        wandb_project: W&B project name (optional, enables logging if provided)
        wandb_run_name: W&B run name (optional)
        resume_from: Path to checkpoint to resume from (optional)
        normalizer: Optional FeatureNormalizer for normalizing raw features

    Returns:
        Tuple of (encoder_ema, predictor_ema)
    """
    os.makedirs(save_dir, exist_ok=True)
    encoder.to(device)
    predictor.to(device)

    # Initialize W&B logger if project is specified
    wandb_logger = None
    if wandb_project:
        config = {
            "lr": lr,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "ema_decay": ema_decay,
            "num_epochs": num_epochs,
            "horizon": horizon,
            "cfg_dropout": cfg_dropout,
            "batch_size": dataloader.batch_size,
            "encoder_params": sum(p.numel() for p in encoder.parameters()),
            "predictor_params": sum(p.numel() for p in predictor.parameters()),
        }
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
        )

    # EMA setup
    encoder_ema = EMAModel(encoder, decay=ema_decay).to(device)
    predictor_ema = EMAModel(predictor, decay=ema_decay).to(device)

    # Optimizer (single optimizer for both models)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Mixed precision training (CPU-safe)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training state
    global_step = 0
    best_loss = float("inf")
    best_epoch = -1
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        encoder_ema.model.load_state_dict(checkpoint["encoder_ema"])
        predictor_ema.model.load_state_dict(checkpoint["predictor_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", -1)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    print(f"Training for {num_epochs} epochs with horizon={horizon}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    encoder.train()
    predictor.train()

    def save_checkpoint(filename: str, loss: float, epoch: int):
        """Save training checkpoint."""
        path = os.path.join(save_dir, filename)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "encoder_ema": encoder_ema.model.state_dict(),
                "predictor_ema": predictor_ema.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "loss": loss,
                "horizon": horizon,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    try:
        use_amp = device.startswith("cuda")
        amp_dtype = torch.float32  # Default for CPU
        if use_amp:
            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        for epoch in tqdm(
            range(start_epoch, num_epochs), desc="Training", unit="epoch"
        ):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch")
            batch_start_time = time.time()

            for batch in pbar:
                # 1. Unpack batch - RAW features from dataset
                motion_raw = batch["motion"].to(device)  # (B, T, 271) - RAW
                B, T, _ = motion_raw.shape

                # Normalize raw features if normalizer is provided
                if normalizer is not None:
                    motion = normalizer.normalize(
                        motion_raw
                    )  # (B, T, 271) - normalized
                else:
                    motion = motion_raw

                # Handle text: Use pre-encoded embeddings or encode raw captions with CLIP
                if "text_clip" in batch:
                    text = batch["text_clip"].to(device)
                elif "captions" in batch and clip_encoder is not None:
                    # Encode raw captions using CLIP
                    captions = batch["captions"]
                    with torch.no_grad():
                        text = clip_encoder(captions)  # (B, 77, 512)
                elif "captions" in batch:
                    raise ValueError(
                        "Raw captions provided but no clip_encoder. "
                        "Pass clip_encoder to train() or provide pre-encoded 'text_clip' in dataset."
                    )
                else:
                    raise ValueError(
                        "No text input found. Batch must contain 'text_clip' or 'captions'."
                    )

                # 2. Sample window based on horizon
                # Use actual sequence lengths from batch to avoid padding issues
                lengths = batch.get(
                    "lengths", torch.full((B,), T, device=device, dtype=torch.long)
                )
                min_length = int(lengths.min().item())

                horizon = min(horizon, min_length - 1)

                max_start = max(1, min_length - horizon - 1)
                start_idx = torch.randint(0, max_start, (1,)).item()
                end_idx = start_idx + horizon

                # 3. Extract history and target (already normalized)
                hist = motion[:, start_idx:end_idx]  # (B, T_hist, 271) - normalized
                target_frames = motion[
                    :, start_idx + 1 : end_idx + 1
                ]  # (B, T_hist, 271) - normalized

                # 5. CFG dropout
                text_input = text if torch.rand(1).item() > cfg_dropout else None

                optimizer.zero_grad(set_to_none=True)

                # 6. Forward pass with mixed precision (CPU-safe)

                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    ## TF training

                    # Encode context - pass normalized features with normalize=False
                    contexts = encoder(
                        text=text_input,
                        input_features=hist,
                        batch_size=B,
                        normalize=False,  # Features already normalized
                    )

                    num_pred_frames = min(contexts.shape[1], target_frames.shape[1])

                    # Split into history contexts and prediction contexts
                    # context[t] has seen frames[0:t], predicts frame[t+1]
                    pred_contexts = contexts[:, -num_pred_frames:]  # (B, N, 22, D)

                    # Previous frames and targets
                    prev_frames = hist[:, -num_pred_frames:]
                    target_frames_tf = target_frames[:, -num_pred_frames:]

                    # Flatten for predictor
                    B, N, J, D = pred_contexts.shape
                    contexts_flat = pred_contexts.reshape(B * N, J, D)
                    prev_flat = prev_frames.reshape(B * N, 271)
                    targets_flat = target_frames_tf.reshape(B * N, 271)

                    # Extract features
                    prev_features = extract_prev_frame_features(prev_flat)  # (B*N, 261)
                    clean_targets = extract_clean_target(targets_flat)

                    # Flow matching: sample t and create noisy target
                    t = torch.rand(B * N, device=device)
                    noise = torch.randn_like(clean_targets)
                    x_t = (
                        t.view(B * N, 1) * clean_targets
                        + (1 - t.view(B * N, 1)) * noise
                    )

                    # Predict velocity field - pass normalized features with normalize=False
                    pred = predictor(
                        history_features=contexts_flat,
                        noise_level=t,
                        noisy_target=x_t,
                        prev_frame_features=prev_features,
                        normalize=False,  # Features already normalized
                    )

                    # Loss: velocity field prediction
                    target_v = clean_targets - noise
                    loss_tf = F.mse_loss(pred, target_v)

                    # ## Rollout training
                    # loss_ar = torch.tensor(0.0, device=device)
                    # rollout_steps = min(4, target_frames.shape[1] - 1)
                    # start_ar = target_frames.shape[1] - rollout_steps

                    # current_history = hist[:, :start_ar]
                    # target_frames_ar = target_frames[
                    #     :, start_ar - 1 : start_ar - 1 + rollout_steps
                    # ]
                    # root_tracker = RootPositionTracker.from_history(current_history)

                    # for step in range(rollout_steps):

                    #     # ---------------------------------
                    #     # 1. Encode current history
                    #     # ---------------------------------
                    #     contexts_roll = encoder(
                    #         text=text_input,
                    #         input_features=current_history,
                    #         batch_size=B,
                    #         normalize=False,  # Features already normalized
                    #     )

                    #     context_last = contexts_roll[:, -1]  # (B, 22, D)
                    #     prev_frame = current_history[:, -1]  # (B, 271)

                    #     # ---------------------------------
                    #     # 2. Prepare flow inputs
                    #     # ---------------------------------
                    #     prev_features = extract_prev_frame_features(prev_frame)

                    #     clean_target = extract_clean_target(target_frames_ar[:, step])

                    #     t = torch.rand(B, device=device)
                    #     noise = torch.randn_like(clean_target)

                    #     x_t = t.view(B, 1) * clean_target + (1 - t.view(B, 1)) * noise

                    #     # ---------------------------------
                    #     # 3. Predict velocity
                    #     # ---------------------------------
                    #     pred_roll = predictor(
                    #         history_features=context_last,
                    #         noise_level=t,
                    #         noisy_target=x_t,
                    #         prev_frame_features=prev_features,
                    #         normalize=False,  # Features already normalized
                    #     )

                    #     target_v_roll = clean_target - noise
                    #     step_loss = F.mse_loss(pred_roll, target_v_roll)

                    #     loss_ar = loss_ar + step_loss

                    #     # ---------------------------------
                    #     # 4. Generate predicted next frame
                    #     # ---------------------------------
                    #     # ---------------------------------
                    #     # Run mini ODE sampling
                    #     # ---------------------------------
                    #     x_sample = torch.randn_like(clean_target)

                    #     num_steps = 10  # small, cheaper than inference
                    #     dt = 1.0 / num_steps

                    #     for s in range(num_steps):
                    #         t_step = torch.full((B,), s * dt, device=device)

                    #         v = predictor(
                    #             history_features=context_last,
                    #             noise_level=t_step,
                    #             noisy_target=x_sample,
                    #             prev_frame_features=prev_features,
                    #             normalize=False,  # Features already normalized
                    #         )

                    #         x_sample = x_sample + v * dt

                    #     clean_pred = x_sample.detach()

                    #     prev_root_pos = root_tracker.get()
                    #     pred_frame, new_root_pos = flow_output_to_271d(
                    #         clean_pred,
                    #         prev_frame,
                    #         prev_root_pos,
                    #     )

                    #     root_tracker.root_pos = new_root_pos

                    #     # ---------------------------------
                    #     # 5. Append predicted frame
                    #     # ---------------------------------
                    #     current_history = torch.cat(
                    #         [current_history, pred_frame.unsqueeze(1)], dim=1
                    #     )

                    # if rollout_steps > 0:
                    #     loss_ar = loss_ar / rollout_steps
                    # else:
                    #     loss_ar = torch.tensor(0.0, device=device)
                loss_ar = 0
                lambda_ar = 0.5
                loss = loss_tf + lambda_ar * loss_ar

                # 7. Backward pass
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                # 8. EMA update
                encoder_ema.update(encoder)
                predictor_ema.update(predictor)

                # 9. Timing and memory metrics
                batch_time = time.time() - batch_start_time
                batch_start_time = time.time()

                gpu_memory_allocated = 0.0
                gpu_memory_reserved = 0.0
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9

                # 10. Logging
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "train/loss_tf": loss_tf.item(),
                            # "train/loss_ar": loss_ar.item(),
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/epoch": epoch,
                            "train/grad_norm": (
                                grad_norm.item()
                                if hasattr(grad_norm, "item")
                                else grad_norm
                            ),
                            "train/batch_time": batch_time,
                            "train/samples_per_sec": (
                                B / batch_time if batch_time > 0 else 0
                            ),
                            "train/horizon": horizon,
                            "train/num_pred_frames": num_pred_frames,
                            "system/gpu_memory_allocated_gb": gpu_memory_allocated,
                            "system/gpu_memory_reserved_gb": gpu_memory_reserved,
                        },
                        step=global_step,
                    )

                if global_step % 100 == 0:
                    tqdm.write(
                        f"[Epoch {epoch}] [Step {global_step}] loss={loss.item():.6f} lr={lr:.2e}"
                    )

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

            # End of Epoch
            pbar.close()
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            tqdm.write(f"==> End of Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}")

            # W&B epoch-level logging
            if wandb_logger is not None:
                wandb_logger.log(
                    {
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/num": epoch,
                    },
                    step=global_step,
                )

            # Checkpointing
            # Save Latest (Always)
            save_checkpoint("latest.pt", avg_epoch_loss, epoch)

            # Save Best
            if avg_epoch_loss < best_loss:
                tqdm.write(
                    f"New best model! (Loss: {best_loss:.6f} -> {avg_epoch_loss:.6f})"
                )
                best_loss = avg_epoch_loss
                best_epoch = epoch
                save_checkpoint("best.pt", avg_epoch_loss, epoch)

    except KeyboardInterrupt:
        tqdm.write("Training interrupted. Saving emergency checkpoint...")
        save_checkpoint("latest_interrupted.pt", 0.0, epoch)
        tqdm.write("Done.")

    # Finish W&B run
    if wandb_logger is not None:
        wandb_logger.log_summary({"best_loss": best_loss, "best_epoch": best_epoch})
        wandb_logger.finish()

    return encoder_ema, predictor_ema


# =============================================================================
# Validation Function
# =============================================================================


def validate(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader,
    horizon: int,
    device: str = "cuda",
    num_batches: int = 10,
    clip_encoder: Optional[nn.Module] = None,
) -> dict:
    """
    Validation with teacher-forced reconstruction.

    Args:
        encoder: MotionHistoryEncoder model (should be EMA)
        predictor: FlowMatchingPredictor model (should be EMA)
        dataloader: Validation DataLoader
        horizon: Current horizon for validation
        device: Device to validate on
        num_batches: Number of batches to validate
        clip_encoder: Optional CLIPEncoder for encoding raw captions

    Returns:
        Dictionary with validation metrics
    """
    encoder.eval()
    predictor.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            motion = batch["motion"].to(device)
            B, T, _ = motion.shape

            # Handle text: Use pre-encoded embeddings or encode raw captions with CLIP
            if "text_clip" in batch:
                text = batch["text_clip"].to(device)
            elif "captions" in batch and clip_encoder is not None:
                captions = batch["captions"]
                text = clip_encoder(captions)
            elif "captions" in batch:
                raise ValueError(
                    "Raw captions provided but no clip_encoder. "
                    "Pass clip_encoder to validate() or provide pre-encoded 'text_clip'."
                )
            else:
                raise ValueError(
                    "No text input found. Batch must contain 'text_clip' or 'captions'."
                )

            lengths = batch.get(
                "lengths", torch.full((B,), T, device=device, dtype=torch.long)
            )
            min_length = int(lengths.min().item())

            if min_length <= horizon + 1:
                continue

            # Sample random window
            max_start = max(1, min_length - horizon - 1)
            start_idx = torch.randint(0, max_start, (1,)).item()
            end_idx = min(start_idx + horizon, min_length - 1)

            hist = motion[:, start_idx:end_idx]
            target_frame = motion[:, end_idx]

            prev_features = extract_prev_frame_features(hist[:, -1])
            clean_target = extract_clean_target(target_frame)

            # Encode context
            context = encoder(text=text, input_features=hist, batch_size=B)[:, -1, :, :]

            # Flow matching
            t = torch.rand(B, device=device)
            noise = torch.randn_like(clean_target)
            x_t = t.view(B, 1) * clean_target + (1 - t.view(B, 1)) * noise
            print(context.shape)
            pred = predictor(
                history_features=context,
                noise_level=t,
                noisy_target=x_t,
                prev_frame_features=prev_features,
            )

            target_v = clean_target - noise
            loss = F.mse_loss(pred, target_v)

            total_loss += loss.item() * B
            total_samples += B

    encoder.train()
    predictor.train()

    return {
        "val_loss": total_loss / max(1, total_samples),
    }
