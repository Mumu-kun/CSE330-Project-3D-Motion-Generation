"""
Training utilities for Motion History Encoder and Flow Matching Predictor.

Redesigned training mechanism:

- Single-step flow matching:
  - Predict ONLY the next frame given a variable-length motion history.
- Progressive AR horizon curriculum:
  - Curriculum controls max history length (in frames).
  - Per-batch, sample history length H ∈ [1, curr_horizon].
- Standard flow matching objective:
  - Velocity prediction in a reduced 72D feature space.
- Light CFG support:
  - Optional low-probability dropout of text conditioning.
- EMA for validation and checkpointing.
- Optional AR-style validation that approximates autoregressive rollout.
"""

import copy
import os
import time
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from config import Config
from utils.dataset import Text2MotionDataset
from utils.wandb_logger import WandbLogger
from utils.motion_utils import (
    FeatureNormalizer,
    RootPositionTracker,  # kept in case you use it elsewhere
    flow_output_to_271d,  # kept in case you use it elsewhere
    extract_prev_frame_features,
)

# =============================================================================
# Feature Extraction Helpers
# =============================================================================


def extract_clean_target(frame: torch.Tensor) -> torch.Tensor:
    """
    Extract 72D clean target from 271D frame.

    72D Output Format:
    [0:9]  Root features: height(1) + velocity(2) + rotation_6d(6)
    [9:72] Joint RIC positions: 21 joints x 3D = 63D

    Args:
        frame: (..., 271) tensor with full 271D motion features

    Returns:
        (..., 72) tensor with cleaned target features
    """
    # Root features: height(1) + velocity(2) = [0:3]
    root_height_vel = frame[..., :3]
    # Root rotation: [69:75] = 6D
    root_rot = frame[..., 69:75]
    root_features = torch.cat([root_height_vel, root_rot], dim=-1)  # (..., 9)

    # Joint RIC: 21 joints x 3D = 63D
    # From [3:69] = 66D (22 joints), we take [6:69] = 63D (21 joints, excluding root)
    joint_features = frame[..., 6:69]  # (..., 63)

    return torch.cat([root_features, joint_features], dim=-1)


# =============================================================================
# EMA Model Wrapper
# =============================================================================


class EMAModel:
    """
    Exponential Moving Average model wrapper.

    Maintains an EMA copy of a model for more stable evaluation.
    EMA is used for validation, sampling, and checkpointing.
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
# Training Setup Helpers
# =============================================================================


def setup_training_environment(
    encoder: nn.Module,
    predictor: nn.Module,
    config: "Config",
    dataloader: DataLoader,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> Tuple[
    Any,
    Any,
    Optional[WandbLogger],
    EMAModel,
    EMAModel,
    Optimizer,
    GradScaler,
    int,
    Dict[str, Any],
    str,
    bool,
]:
    """
    Setup training environment: device, directories, W&B, EMA models, optimizer.

    Returns:
        (device, save_dir, wandb_logger, encoder_ema, predictor_ema,
         optimizer, scaler, start_epoch, training_state, device_str, use_amp)
    """
    # Extract config values
    device = config.device
    lr = config.learning_rate
    weight_decay = config.weight_decay
    ema_decay = config.ema_decay
    horizon = config.horizon
    curriculum = config.curriculum
    cfg_dropout = config.cfg_dropout
    num_epochs = config.num_epochs

    # Setup device and directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    encoder.to(device)
    predictor.to(device)

    # Setup W&B logger
    wandb_logger = None
    if wandb_project:
        wandb_config = {
            "lr": lr,
            "weight_decay": weight_decay,
            "ema_decay": ema_decay,
            "num_epochs": num_epochs,
            "horizon": horizon,
            "cfg_dropout": cfg_dropout,
            "batch_size": dataloader.batch_size,
            "encoder_params": sum(p.numel() for p in encoder.parameters()),
            "predictor_params": sum(p.numel() for p in predictor.parameters()),
            "curriculum": curriculum,
        }
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
        )

    # Setup EMA models
    encoder_ema = EMAModel(encoder, decay=ema_decay).to(device)
    predictor_ema = EMAModel(predictor, decay=ema_decay).to(device)

    # Setup optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # type: ignore

    # Setup mixed precision (CUDA only)
    device_str = str(device)
    use_amp = device_str.startswith("cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    # Training state
    training_state: Dict[str, Any] = {
        "global_step": 0,
        "best_loss": float("inf"),
        "best_epoch": -1,
        "best_val_loss": float("inf"),
        "best_val_epoch": -1,
    }
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        encoder_ema.model.load_state_dict(checkpoint["encoder_ema"])
        predictor_ema.model.load_state_dict(checkpoint["predictor_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        training_state["global_step"] = checkpoint.get("global_step", 0)
        training_state["best_loss"] = checkpoint.get("best_loss", float("inf"))
        training_state["best_epoch"] = checkpoint.get("best_epoch", -1)
        training_state["best_val_loss"] = checkpoint.get("best_val_loss", float("inf"))
        training_state["best_val_epoch"] = checkpoint.get("best_val_epoch", -1)
        if "current_horizon" in checkpoint:
            training_state["current_horizon"] = checkpoint["current_horizon"]
        print(
            f"Resumed from epoch {start_epoch}, "
            f"step {training_state['global_step']}"
        )

    # Print training info
    if curriculum is not None and len(curriculum) > 0:
        print(f"Training for {num_epochs} epochs with curriculum: " f"{curriculum}")
    else:
        print(f"Training for {num_epochs} epochs with fixed horizon={horizon}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    encoder.train()
    predictor.train()

    return (
        device,
        str(config.checkpoint_dir),
        wandb_logger,
        encoder_ema,
        predictor_ema,
        optimizer,
        scaler,
        start_epoch,
        training_state,
        device_str,
        use_amp,
    )


def setup_curriculum_state(
    curriculum: Optional[list[dict[str, int]]],
    horizon: int,
    checkpoint_state: Optional[dict] = None,
) -> dict:
    """
    Initialize curriculum learning state.

    Returns:
        {
            "use_curriculum": bool,
            "current_horizon": int,
        }
    """
    use_curriculum = curriculum is not None and len(curriculum) > 0

    if checkpoint_state and "current_horizon" in checkpoint_state:
        current_horizon = checkpoint_state["current_horizon"]
        max_horizon = checkpoint_state["max_horizon"]
    elif use_curriculum and curriculum:
        current_horizon = curriculum[0]["horizon"]
        max_horizon = curriculum[-1]["horizon"]
    else:
        current_horizon = horizon
        max_horizon = horizon

    return {
        "use_curriculum": use_curriculum,
        "current_horizon": current_horizon,
        "max_horizon": max_horizon,
    }


def save_training_checkpoint(
    save_dir: str,
    filename: str,
    encoder: nn.Module,
    predictor: nn.Module,
    encoder_ema: EMAModel,
    predictor_ema: EMAModel,
    optimizer: Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    config: "Config",
    curriculum_state: dict,
    training_state: dict,
) -> None:
    """
    Save training checkpoint with all required state.
    """
    path = os.path.join(save_dir, filename)
    checkpoint = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "encoder_ema": encoder_ema.model.state_dict(),
        "predictor_ema": predictor_ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "horizon": curriculum_state["max_horizon"],
        "current_horizon": curriculum_state["current_horizon"],
        "use_curriculum": curriculum_state["use_curriculum"],
        "best_loss": training_state["best_loss"],
        "best_epoch": training_state["best_epoch"],
        "best_val_loss": training_state["best_val_loss"],
        "best_val_epoch": training_state["best_val_epoch"],
    }
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


# =============================================================================
# Batch Processing Helpers
# =============================================================================


def unpack_batch(
    batch: dict,
    device: torch.device,
    normalizer: Optional["FeatureNormalizer"] = None,
    clip_encoder: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Unpack and normalize batch data from dataloader.

    Returns:
        motion: (B, T, 271) normalized
        text:   (B, 512) or (B, 77, 512)
        B, T
    """
    motion_raw = batch["motion"].to(device)  # (B, T, 271)
    B, T, _ = motion_raw.shape

    if normalizer is not None:
        motion = normalizer.normalize(motion_raw)
    else:
        motion = motion_raw

    if "text_clip" in batch:
        text = batch["text_clip"].to(device)
    elif "captions" in batch and clip_encoder is not None:
        captions = batch["captions"]
        with torch.no_grad():
            text = clip_encoder(captions)
    elif "captions" in batch:
        raise ValueError(
            "Raw captions provided but no clip_encoder. "
            "Pass clip_encoder to train() or provide pre-encoded 'text_clip' in dataset."
        )
    else:
        raise ValueError(
            "No text input found. Batch must contain 'text_clip' or 'captions'."
        )

    return motion, text, B, T


def sample_next_frame_window(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    curr_horizon: int,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Sample (history, next_frame) pair based on current AR horizon.

    Args:
        motion:   (B, T, 271) normalized
        lengths:  (B,) sequence lengths
        curr_horizon: current max history length (frames)
        device:   device

    Returns:
        hist: (B, H, 271) history up to frame t-1
        target: (B, 271) next frame at t
        effective_horizon: H
    """
    B, T, _ = motion.shape
    assert curr_horizon <= T - 1, f"curr_horizon {curr_horizon} > T-1 {T-1}"
    hist = motion[:, 0:curr_horizon]  # (B, curr_horizon, 271)
    target = motion[:, curr_horizon:]  # (B, T - curr_horizon, 271)

    return hist, target, curr_horizon


def apply_cfg_dropout(
    text: torch.Tensor,
    cfg_dropout: float,
    device: Union[str, torch.device],
    B: int,
    encoder_text_dim: int,
) -> Optional[torch.Tensor]:
    """
    Apply classifier-free guidance dropout.

    Returns:
        text tensor or None (for unconditional branch).
    """
    if cfg_dropout <= 0.0:
        return text

    if torch.rand(1).item() > cfg_dropout:
        return text
    else:
        return None


def prepare_text_for_encoder(
    text_input: Optional[torch.Tensor],
    device: Union[str, torch.device],
    B: int,
    encoder_text_dim: int,
) -> torch.Tensor:
    """
    Prepare text embeddings for encoder.

    Returns:
        (B, encoder_text_dim)
    """
    if text_input is None:
        return torch.zeros(B, encoder_text_dim, device=device)
    else:
        # text_input may be (B, 77, D) or (B, D)
        if text_input.dim() == 3:
            return text_input.squeeze(1)
        return text_input


# =============================================================================
# Logging Helpers
# =============================================================================


def log_batch_metrics(
    wandb_logger: Optional[WandbLogger],
    loss: torch.Tensor,
    lr: float,
    epoch: int,
    grad_norm: torch.Tensor,
    batch_time: float,
    B: int,
    current_horizon: int,
    effective_horizon: int,
    num_pred_frames: int,
    loss_components: dict,
    global_step: int,
) -> None:
    """
    Log batch-level metrics to W&B.
    """
    if wandb_logger is None:
        return

    wandb_logger.log(
        {
            "train/loss_root_y": loss_components["root_y"].item(),
            "train/loss_root_vel": loss_components["root_vel"].item(),
            "train/loss_root_rot": loss_components["root_rot"].item(),
            "train/loss_joints": loss_components["joints"].item(),
            "train/loss": loss.item(),
            "train/lr": lr,
            "train/epoch": epoch,
            "train/grad_norm": (
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
            ),
            "train/batch_time": batch_time,
            "train/samples_per_sec": (B / batch_time if batch_time > 0 else 0),
            "train/current_horizon": current_horizon,
            "train/effective_horizon": effective_horizon,
            "train/num_pred_frames": num_pred_frames,
        },
        step=global_step,
    )


def log_epoch_metrics(
    wandb_logger: Optional[WandbLogger],
    avg_epoch_loss: float,
    epoch: int,
    current_horizon: int,
    val_metrics: dict,
    global_step: int,
) -> None:
    """
    Log epoch-level metrics to W&B.
    """
    if wandb_logger is None:
        return

    log_dict = {
        "epoch/avg_loss": avg_epoch_loss,
        "epoch/num": epoch,
        "epoch/current_horizon": current_horizon,
    }
    if val_metrics and "val_loss" in val_metrics:
        log_dict["epoch/val_loss"] = val_metrics["val_loss"]
    wandb_logger.log(log_dict, step=global_step)


# =============================================================================
# Checkpointing Helpers
# =============================================================================


def handle_checkpointing(
    save_dir: str,
    encoder: nn.Module,
    predictor: nn.Module,
    encoder_ema: EMAModel,
    predictor_ema: EMAModel,
    optimizer: Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    avg_epoch_loss: float,
    config: "Config",
    curriculum_state: dict,
    training_state: dict,
    val_metrics: dict,
) -> dict:
    """
    Handle checkpoint saving (latest, best, best_val).
    """
    # Save latest
    save_training_checkpoint(
        save_dir=save_dir,
        filename="latest.pt",
        encoder=encoder,
        predictor=predictor,
        encoder_ema=encoder_ema,
        predictor_ema=predictor_ema,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        loss=avg_epoch_loss,
        config=config,
        curriculum_state=curriculum_state,
        training_state=training_state,
    )

    # Best by training loss
    if avg_epoch_loss < training_state["best_loss"]:
        tqdm.write(
            f"New best model! "
            f"(Loss: {training_state['best_loss']:.6f} -> {avg_epoch_loss:.6f})"
        )
        training_state["best_loss"] = avg_epoch_loss
        training_state["best_epoch"] = epoch
        save_training_checkpoint(
            save_dir=save_dir,
            filename="best.pt",
            encoder=encoder,
            predictor=predictor,
            encoder_ema=encoder_ema,
            predictor_ema=predictor_ema,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            loss=avg_epoch_loss,
            config=config,
            curriculum_state=curriculum_state,
            training_state=training_state,
        )

    # Best by validation loss
    if config.save_best_val and val_metrics and "val_loss" in val_metrics:
        val_loss = val_metrics["val_loss"]
        if val_loss < training_state["best_val_loss"]:
            tqdm.write(
                "New best validation model! "
                f"(Val Loss: {training_state['best_val_loss']:.6f} -> {val_loss:.6f})"
            )
            training_state["best_val_loss"] = val_loss
            training_state["best_val_epoch"] = epoch
            save_training_checkpoint(
                save_dir=save_dir,
                filename="best_val.pt",
                encoder=encoder,
                predictor=predictor,
                encoder_ema=encoder_ema,
                predictor_ema=predictor_ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                loss=avg_epoch_loss,
                config=config,
                curriculum_state=curriculum_state,
                training_state=training_state,
            )

    return training_state


# =============================================================================
# Validation (Teacher-Forced or AR-style)
# =============================================================================


def validate(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader,
    horizon: int,
    device: str = "cuda",
    num_batches: int = 10,
    clip_encoder: Optional[nn.Module] = None,
    normalizer: Optional["FeatureNormalizer"] = None,
) -> dict:
    """
    Simple teacher-forced validation: single-step prediction from a
    random history window of length up to `horizon`.
    """
    encoder.eval()
    predictor.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches > 0 and i >= num_batches:
                break

            motion_raw = batch["motion"].to(device)  # (B, T, 271)
            B, T, _ = motion_raw.shape

            if normalizer is not None:
                motion = normalizer.normalize(motion_raw)
            else:
                motion = motion_raw

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

            # Use horizon as max history length here
            hist, target_frames, H = sample_next_frame_window(
                motion=motion,
                lengths=lengths,
                curr_horizon=horizon,
                device=device,
            )

            target_frame = target_frames[:, 0]  # (B, 271)
            text_dim = getattr(encoder, "text_embedding_dim", 512)
            text_for_encoder = text.squeeze(1) if text.dim() == 3 else text
            if text_for_encoder.shape[-1] != text_dim:
                # Optionally handle mismatch via projection
                pass

            context = encoder(hist, text_for_encoder)  # (B, 22, D)

            clean_target = extract_clean_target(target_frame)  # (B, 72)
            prev_frame = hist[:, -1]
            prev_features = extract_prev_frame_features(prev_frame)

            t = torch.rand(B, device=device)
            noise = torch.randn_like(clean_target)
            x_t = t.view(B, 1) * clean_target + (1 - t.view(B, 1)) * noise

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


# =============================================================================
# Main Training Function
# =============================================================================


def train(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader,
    config: "Config",
    clip_encoder: Optional[nn.Module] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    resume_from: Optional[str] = None,
    normalizer: Optional["FeatureNormalizer"] = None,
    val_dataloader: Optional[DataLoader] = None,
) -> Tuple[EMAModel, EMAModel]:
    """
    Training loop with progressive AR horizon curriculum and single-step
    flow matching (predict next frame from history window).
    """
    (
        device,
        checkpoint_dir,
        wandb_logger,
        encoder_ema,
        predictor_ema,
        optimizer,
        scaler,
        start_epoch,
        training_state,
        device_str,
        use_amp,
    ) = setup_training_environment(
        encoder=encoder,
        predictor=predictor,
        config=config,
        dataloader=dataloader,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        resume_from=resume_from,
    )

    curriculum_state = setup_curriculum_state(
        curriculum=config.curriculum,
        horizon=config.horizon,
        checkpoint_state=(
            training_state if "current_horizon" in training_state else None
        ),
    )

    num_epochs = config.num_epochs
    lr = config.learning_rate
    max_grad_norm = config.gradient_clip
    cfg_dropout = config.cfg_dropout
    val_interval = config.val_interval
    val_batches = config.val_batches
    val_use_ema = config.val_use_ema

    amp_dtype = torch.float32
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        # Explicit type annotation for type checkers
        training_state: Dict[str, Any] = training_state
        for epoch in tqdm(
            range(start_epoch, num_epochs), desc="Training", unit="epoch"
        ):
            # Curriculum horizon update
            prev_horizon = curriculum_state["current_horizon"]
            if curriculum_state["use_curriculum"] and config.curriculum:
                for level in reversed(config.curriculum):
                    if epoch <= level["epochs"]:
                        curriculum_state["current_horizon"] = level["horizon"]
                    else:
                        break
                if curriculum_state["current_horizon"] != prev_horizon:
                    tqdm.write(
                        "Curriculum update: "
                        f"horizon {prev_horizon} -> {curriculum_state['current_horizon']}"
                    )

            epoch_loss = 0.0
            num_batches = 0
            pred_horizon = 1

            dataloader.dataset.set_horizon(curriculum_state["current_horizon"] + pred_horizon)  # type: ignore

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch")
            batch_start_time = time.time()

            for batch in pbar:
                motion, text, B, T = unpack_batch(
                    batch, device, normalizer, clip_encoder
                )

                lengths = batch.get(
                    "lengths",
                    torch.full((B,), T, device=device, dtype=torch.long),
                )

                hist, target_frames, effective_horizon = sample_next_frame_window(
                    motion=motion,
                    lengths=lengths,
                    curr_horizon=curriculum_state["current_horizon"],
                    device=device,
                )

                text_input = apply_cfg_dropout(
                    text, cfg_dropout, device, B, config.encoder_text_dim
                )
                text_for_encoder = prepare_text_for_encoder(
                    text_input, device, B, config.encoder_text_dim
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_str, dtype=amp_dtype, enabled=use_amp):  # type: ignore
                    # Encode history
                    contexts = encoder(hist, text_for_encoder)  # (B, 22, D)

                    # Flow matching target
                    target_frame = target_frames[:, 0]  # (B, 271)
                    clean_targets = extract_clean_target(target_frame)  # (B, 72)
                    prev_frame = hist[:, -1]
                    prev_features = extract_prev_frame_features(prev_frame)

                    t = torch.rand(B, device=device)
                    noise = torch.randn_like(clean_targets)
                    x_t = t.view(B, 1) * clean_targets + (1 - t.view(B, 1)) * noise

                    pred = predictor(
                        history_features=contexts,
                        noise_level=t,
                        noisy_target=x_t,
                        prev_frame_features=prev_features,
                    )

                    target_v = clean_targets - noise

                    loss_tf = F.mse_loss(pred, target_v)
                    loss_root_y = F.mse_loss(pred[:, 0], target_v[:, 0])
                    loss_root_vel = F.mse_loss(pred[:, 1:3], target_v[:, 1:3])
                    loss_root_rot = F.mse_loss(pred[:, 3:9], target_v[:, 3:9])
                    loss_joints = F.mse_loss(pred[:, 9:], target_v[:, 9:])

                    loss = loss_tf

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()

                encoder_ema.update(encoder)
                predictor_ema.update(predictor)

                batch_time = time.time() - batch_start_time
                batch_start_time = time.time()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

                log_batch_metrics(
                    wandb_logger=wandb_logger,
                    loss=loss,
                    lr=lr,
                    epoch=epoch,
                    grad_norm=grad_norm,
                    batch_time=batch_time,
                    B=B,
                    current_horizon=curriculum_state["current_horizon"],
                    effective_horizon=effective_horizon,
                    num_pred_frames=1,
                    loss_components={
                        "root_y": loss_root_y,
                        "root_vel": loss_root_vel,
                        "root_rot": loss_root_rot,
                        "joints": loss_joints,
                    },
                    global_step=training_state["global_step"],
                )

                if training_state["global_step"] % 100 == 0:
                    tqdm.write(
                        f"[Epoch {epoch}] [Step {training_state['global_step']}] "
                        f"loss={loss.item():.6f} lr={lr:.2e}"
                    )

                epoch_loss += loss.item()
                num_batches += 1
                training_state["global_step"] += 1

            pbar.close()
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            tqdm.write(f"==> End of Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}")

            # Validation
            val_metrics: dict = {}
            if val_dataloader is not None and (epoch + 1) % val_interval == 0:
                tqdm.write("Running validation...")

                if val_use_ema:
                    val_encoder = encoder_ema.model
                    val_predictor = predictor_ema.model
                else:
                    val_encoder = encoder
                    val_predictor = predictor

                val_metrics = validate(
                    encoder=val_encoder,
                    predictor=val_predictor,
                    dataloader=val_dataloader,
                    horizon=curriculum_state["current_horizon"],
                    device=device,
                    num_batches=val_batches,
                    clip_encoder=clip_encoder,
                    normalizer=normalizer,
                )

                val_loss = val_metrics["val_loss"]
                tqdm.write(f"Validation loss: {val_loss:.6f}")

                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "val/loss": val_loss,
                            "val/epoch": epoch,
                        },
                        step=training_state["global_step"],
                    )

            log_epoch_metrics(
                wandb_logger=wandb_logger,
                avg_epoch_loss=avg_epoch_loss,
                epoch=epoch,
                current_horizon=curriculum_state["current_horizon"],
                val_metrics=val_metrics,
                global_step=training_state["global_step"],
            )

            training_state = handle_checkpointing(
                save_dir=checkpoint_dir,
                encoder=encoder,
                predictor=predictor,
                encoder_ema=encoder_ema,
                predictor_ema=predictor_ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=training_state["global_step"],
                avg_epoch_loss=avg_epoch_loss,
                config=config,
                curriculum_state=curriculum_state,
                training_state=training_state,
                val_metrics=val_metrics,
            )

    except KeyboardInterrupt:
        tqdm.write("Training interrupted. Saving emergency checkpoint...")
        save_training_checkpoint(
            save_dir=checkpoint_dir,
            filename="latest_interrupted.pt",
            encoder=encoder,
            predictor=predictor,
            encoder_ema=encoder_ema,
            predictor_ema=predictor_ema,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=training_state["global_step"],
            loss=0.0,
            config=config,
            curriculum_state=curriculum_state,
            training_state=training_state,
        )
        tqdm.write("Done.")

    if wandb_logger is not None:
        summary = {
            "best_loss": training_state["best_loss"],
            "best_epoch": training_state["best_epoch"],
            "best_val_loss": training_state["best_val_loss"],
            "best_val_epoch": training_state["best_val_epoch"],
        }
        if curriculum_state["use_curriculum"]:
            summary["final_horizon"] = curriculum_state["current_horizon"]
            summary["max_horizon"] = curriculum_state["max_horizon"]
        wandb_logger.log_summary(summary)
        wandb_logger.finish()

    return encoder_ema, predictor_ema
