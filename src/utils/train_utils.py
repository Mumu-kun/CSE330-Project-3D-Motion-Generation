import math
import copy
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, List, Union, Tuple
from tqdm import tqdm


def build_prev_and_clean_diffs(
    hist: torch.Tensor, future: torch.Tensor, joint_count: int = 22
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts spatial features for the previous frame and the target displacement (velocities).

    Feature Layout (Standard HumanML3D 263D):
    - [4:67]    RIC Positions (21 joints, root is 0)
    - [67:193]  RIC Rotations (21 joints, 6D)
    - [193:259] Local Velocities (22 joints, 3D)

    Args:
        hist:   History frames (B, T_hist, 263)
        future: Target frame (B, 1, 263)

    Returns:
        prev_pos:   (B, 22, 3)
        prev_rot6d: (B, 22, 6)
        prev_v:     (B, 22, 3)
        clean_v:    (B, 22, 3) - The "Velocity" target for Flow Matching
    """
    B, T_hist, _ = hist.shape
    last_frame = hist[:, -1]
    target_frame = future[:, 0]

    # 1. RIC Position extraction
    def get_pos(frame):
        # RIC is for 21 non-root joints. Root is at index 0 and is (0,0,0) in RIC space.
        ric_21 = frame[:, 4:67].reshape(B, 21, 3)
        root_pos = torch.zeros((B, 1, 3), device=frame.device, dtype=frame.dtype)
        return torch.cat([root_pos, ric_21], dim=1)  # (B, 22, 3)

    prev_pos = get_pos(last_frame)

    # 2. RIC Rotation extraction (6D)
    def get_rot(frame):
        # 21 joints. Root is identity in 6D: [1, 0, 0, 0, 1, 0]
        rot_21 = frame[:, 67:193].reshape(B, 21, 6)
        root_rot = torch.zeros((B, 1, 6), device=frame.device, dtype=frame.dtype)
        root_rot[:, 0, 0] = 1.0  # [1, 0, 0, ...]
        root_rot[:, 0, 4] = 1.0  # [..., 1, 0]
        return torch.cat([root_rot, rot_21], dim=1)  # (B, 22, 6)

    prev_rot6d = get_rot(last_frame)

    # 3. Velocity extraction
    prev_v = last_frame[:, 193:259].contiguous().view(B, 22, 3)
    clean_v = target_frame[:, 193:259].contiguous().view(B, 22, 3)

    return prev_pos, prev_rot6d, prev_v, clean_v


def train(
    motion_history_encoder: torch.nn.Module,
    flow_predictor: torch.nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    save_dir: str,
    device: str = "cuda",
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    ema_decay: float = 0.9999,
):
    """
    Trains the motion generation models using Flow Matching.

    Args:
        motion_history_encoder:  Context encoder (Dual-MLP GRU)
        flow_predictor:          Flow predictor (Spatial Transformer)
        dataloader:              DataLoader providing HumanML3D batches
        num_epochs:              Total training epochs
        save_dir:                Directory to save checkpoints
        ...
    """
    os.makedirs(save_dir, exist_ok=True)
    motion_history_encoder.to(device)
    flow_predictor.to(device)

    # EMA setup (Exponential Moving Average)
    def copy_model(m):
        ema = copy.deepcopy(m)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    ema_mhe = copy_model(motion_history_encoder)
    ema_fmp = copy_model(flow_predictor)

    params = list(motion_history_encoder.parameters()) + list(
        flow_predictor.parameters()
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # type: ignore
    scaler = GradScaler()

    # Calculate total steps for scheduler
    total_steps = num_epochs * len(dataloader)
    print(f"Training for {num_epochs} epochs, total {total_steps} steps.")

    # LR schedule: warmup + cosine decay
    def lr_lambda(step):
        warmup = max(1, int(0.02 * total_steps))
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_loss = float("inf")

    motion_history_encoder.train()
    flow_predictor.train()

    def save_checkpoint(filename: str, loss: float):
        path = os.path.join(save_dir, filename)
        torch.save(
            {
                "motion_history_encoder": motion_history_encoder.state_dict(),
                "flow_predictor": flow_predictor.state_dict(),
                "ema_mhe": ema_mhe.state_dict(),
                "ema_fmp": ema_fmp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "loss": loss,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    try:
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch")
            for batch in pbar:
                # 1. Unpack batch
                motion = batch["motion"].to(device)  # [B, T, 263]
                B, T, _ = motion.shape

                # Handle Text: Prefer pre-encoded embeddings, fallback to raw captions
                text_embeddings = batch.get("text_clip", batch.get("captions"))
                if hasattr(text_embeddings, "to"):
                    text_embeddings = text_embeddings.to(device)

                # Handle Duration: Prefer normalized duration, fallback to lengths
                if "duration" in batch:
                    duration = batch["duration"].to(device)
                else:
                    # Normalize lengths by max_motion_length (default 200)
                    lengths = batch["lengths"].to(device).float()
                    duration = (lengths / 200.0).unsqueeze(-1)  # (B, 1)

                # 2. Classifier-Free Guidance (CFG) Dropout Implementation
                global_dropout_prob = 0.05  # 5% chance to drop everything
                cond_dropout_prob = 0.1  # 10% chance to drop individual signals

                is_global_uncond = torch.rand(1) < global_dropout_prob

                # 3. Zero-Shot Logic

                # 10% chance to train for zero-shot initial pose generation (frame 0)
                is_zero_shot = (torch.rand(1) < 0.1) and (T > 0)

                # CRITICAL: Initialize these BEFORE any branches
                t_prog = None
                prev_features = None
                clean_diffs = None
                text_input = None
                dur_input = None
                hist_input_for_encoder = None
                future = None

                if is_zero_shot:
                    # ===== ZERO-SHOT BRANCH =====
                    # Target is the very first frame
                    future = motion[:, 0:1]  # The first frame

                    text_input = (
                        None
                        if (is_global_uncond or torch.rand(1) < cond_dropout_prob)
                        else text_embeddings
                    )
                    dur_input = (
                        None
                        if (is_global_uncond or torch.rand(1) < cond_dropout_prob)
                        else duration
                    )
                    hist_input_for_encoder = None

                    # CRITICAL: Set t_prog to None for zero-shot
                    t_prog = None

                    # Initialize prev_features and clean_diffs for zero-shot
                    prev_features = None
                    clean_diffs = future[:, 0, 193:259].contiguous().view(B, 22, 3)

                else:
                    # ===== STANDARD WINDOW SAMPLING BRANCH =====
                    # Select a random window for training
                    idx_limit = min(
                        int(batch["lengths"].min().item()) if "lengths" in batch else T,
                        T,
                    )

                    end_idx = torch.randint(1, idx_limit - 1, (1,)).item()
                    start_idx = torch.randint(0, end_idx, (1,)).item()  # type: ignore

                    # Extract actual slices
                    hist_actual_slice = motion[:, start_idx:end_idx]
                    future = motion[:, start_idx : end_idx + 1]

                    # Standard Dropout Logic
                    text_input = (
                        None
                        if (is_global_uncond or torch.rand(1) < cond_dropout_prob)
                        else text_embeddings
                    )
                    dur_input = (
                        None
                        if (is_global_uncond or torch.rand(1) < cond_dropout_prob)
                        else duration
                    )
                    hist_input_for_encoder = (
                        None
                        if (is_global_uncond or torch.rand(1) < cond_dropout_prob)
                        else hist_actual_slice
                    )

                    # CRITICAL: Build spatial features and compute t_prog in this branch
                    prev_pos, prev_rot6d, prev_v, clean_diffs = (
                        build_prev_and_clean_diffs(hist_actual_slice, future)
                    )
                    prev_features = torch.cat([prev_pos, prev_rot6d, prev_v], dim=-1)

                    # Compute temporal progress: normalized position in sequence
                    prog = float(end_idx) / float(T)
                    t_prog = torch.full((B,), prog, device=device, dtype=torch.float32)

                # SANITY CHECK: Verify all variables are initialized
                assert future is not None, "future not initialized!"
                assert clean_diffs is not None, "clean_diffs not initialized!"
                assert (
                    text_input is None or text_input.shape[0] == B
                ), f"text_input batch size mismatch: {text_input.shape[0]} != {B}"
                assert (
                    dur_input is None or dur_input.shape[0] == B
                ), f"dur_input batch size mismatch: {dur_input.shape[0]} != {B}"
                assert (
                    t_prog is None or t_prog.shape[0] == B
                ), f"t_prog batch size mismatch: {t_prog.shape[0]} != {B}"

                optimizer.zero_grad(set_to_none=True)

                # 4. Training Loop Step
                with autocast(
                    dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                ):
                    # A) Encode motion history
                    history_context = motion_history_encoder(
                        text=text_input,
                        input_features=hist_input_for_encoder,
                        # total_duration=dur_input,
                        batch_size=B,
                    )

                    # C) Sample noise and flow time t
                    eps = torch.randn_like(clean_diffs)  # [B, 22, 3]
                    t = torch.rand(B, device=device)  # [B]
                    t_b = t.view(B, 1, 1)  # [B, 1, 1]

                    # Flow Matching: Interpolate clean -> noisy as t: 0 -> 1
                    noisy_target_diffs = t_b * clean_diffs + (1.0 - t_b) * eps

                    # D) Forward through flow predictor
                    pred_eps = flow_predictor(
                        history_features=history_context,
                        noise_level=t,
                        noisy_target_diffs=noisy_target_diffs,
                        prev_frame_features=prev_features,
                        # temporal_progress=t_prog,
                    )

                    flow_target = clean_diffs - eps

                    # E) Loss: Prediction of the velocity field/noise
                    loss = F.mse_loss(pred_eps, flow_target)

                # 4. Optimizer Step
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # 5. EMA update
                with torch.no_grad():
                    for ema_p, p in zip(
                        ema_mhe.parameters(), motion_history_encoder.parameters()
                    ):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                    for ema_p, p in zip(
                        ema_fmp.parameters(), flow_predictor.parameters()
                    ):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

                # 6. Logging
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                if global_step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    tqdm.write(
                        f"[Epoch {epoch}] [Step {global_step}] loss={loss.item():.6f} lr={current_lr:.2e}"
                    )

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

            # End of Epoch
            pbar.close()
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            tqdm.write(f"==> End of Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}")

            # Checkpointing
            # Save Latest (Always)
            save_checkpoint("latest.pt", avg_epoch_loss)

            # Save Best
            if avg_epoch_loss < best_loss:
                tqdm.write(
                    f"New best model! (Loss: {best_loss:.6f} -> {avg_epoch_loss:.6f})"
                )
                best_loss = avg_epoch_loss
                save_checkpoint("best.pt", avg_epoch_loss)

    except KeyboardInterrupt:
        tqdm.write("Training interrupted. Saving emergency checkpoint...")
        save_checkpoint("latest_interrupted.pt", 0.0)
        tqdm.write("Done.")

    return ema_mhe, ema_fmp
