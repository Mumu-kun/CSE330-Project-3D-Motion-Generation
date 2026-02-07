import math
import copy
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, List, Union, Tuple


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
    dataloader: torch.utils.data.DataLoader,
    num_steps: int,
    device: str = "cuda",
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    ema_decay: float = 0.999,
):
    """
    Trains the motion generation models using Flow Matching.

    Args:
        motion_history_encoder: Context encoder (Dual-MLP GRU)
        flow_predictor:          Flow predictor (Spatial Transformer)
        dataloader:              DataLoader providing HumanML3D batches
        num_steps:               Total training steps
        ...
    """
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
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()

    # LR schedule: warmup + cosine decay
    def lr_lambda(step):
        warmup = max(1, int(0.02 * num_steps))
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(max(1, num_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    motion_history_encoder.train()
    flow_predictor.train()

    # Continuous data iterator
    def get_data_iter():
        while True:
            for batch in dataloader:
                yield batch

    data_iter = get_data_iter()

    while step < num_steps:
        batch = next(data_iter)

        # 1. Unpack batch (Support both snippet keys and project dataloader keys)
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
        # Randomly set conditioning signals to None to train learnable null tokens
        global_dropout_prob = 0.05  # 5% chance to drop everything
        cond_dropout_prob = 0.1  # 10% chance to drop individual signals

        is_global_uncond = torch.rand(1) < global_dropout_prob

        # 3. Window Sampling & Zero-Shot Logic
        T_hist = 15  # History window size

        # 10% chance to train for zero-shot initial pose generation (frame 0)
        is_zero_shot = (torch.rand(1) < 0.1) and (T > 0)

        if is_zero_shot:
            # Target is the very first frame
            start_idx = 0
            hist_actual_slice = None  # No physical history
            future = motion[:, 0:1]  # The first frame

            # For zero-shot, we usually WANT text to be present, but still allow dropout
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

        else:
            # Standard window sampling logic
            if T <= T_hist:
                continue

            # Select a random window for training
            idx_limit = min(
                int(batch["lengths"].min().item()) if "lengths" in batch else T, T
            )
            if idx_limit <= T_hist:
                continue

            start_idx = torch.randint(0, idx_limit - T_hist, (1,)).item()

            # Extract actual slices
            hist_actual_slice = motion[:, start_idx : start_idx + T_hist]
            future = motion[:, start_idx + T_hist : start_idx + T_hist + 1]

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

        optimizer.zero_grad(set_to_none=True)

        # 4. Training Loop Step
        with autocast(
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ):
            # A) Encode motion history
            history_context = motion_history_encoder(
                text=text_input,
                input_features=hist_input_for_encoder,
                total_duration=dur_input,
            )

            # B) Build spatial features and clean target
            if hist_actual_slice is not None:
                prev_pos, prev_rot6d, prev_v, clean_diffs = build_prev_and_clean_diffs(
                    hist_actual_slice, future
                )
                prev_features = torch.cat([prev_pos, prev_rot6d, prev_v], dim=-1)
                prog = float(start_idx + T_hist) / T
            else:
                # Zero-shot case: no previous frame
                prev_features = None
                # For simplicity, target_frame velocity is used as the target displacement
                clean_diffs = future[:, 0, 193:259].contiguous().view(B, 22, 3)
                prog = None

            if prog is None:
                t_prog = None
            else:
                t_prog = torch.full((B,), prog, device=device)

            # C) Sample noise and flow time t
            eps = torch.randn_like(clean_diffs)  # [B, 22, 3]
            t = torch.rand(B, device=device)  # [B]
            t_b = t.view(B, 1, 1)  # [B, 1, 1]

            # Flow Matching: Interpolate clean -> noisy as t: 0 -> 1
            noisy_target_diffs = (1.0 - t_b) * clean_diffs + t_b * eps

            # D) Forward through flow predictor
            pred_eps = flow_predictor(
                history_features=history_context,
                noise_level=t,
                noisy_target_diffs=noisy_target_diffs,
                prev_frame_features=prev_features,
                temporal_progress=t_prog,
            )

            # E) Loss: Prediction of the velocity field/noise
            loss = F.mse_loss(pred_eps, eps)

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
            for ema_p, p in zip(ema_fmp.parameters(), flow_predictor.parameters()):
                ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        # 6. Logging
        if step % 100 == 0:
            print(
                f"[step {step:5d}] loss={loss.item():.6f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        step += 1

    return ema_mhe, ema_fmp
