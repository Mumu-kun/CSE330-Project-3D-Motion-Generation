"""
Inference pipeline for Text-to-3D Motion generation using FlowMatchingPredictor and MotionHistoryEncoder.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from utils.motion_utils import (
    FeatureNormalizer,
    Features,
    enforce_rigid_bone_lengths,
    positions_to_x271,
    root_rot6d_to_yaw,
    sample_initial_noise,
    x68_to_positions,
    x271_to_positions,
)


def temporal_gaussian_smooth(positions: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """Smooth joint positions along the temporal dimension using a 1D Gaussian filter.

    Args:
        positions: (horizon, 22, 3) tensor of joint positions.
        sigma: Standard deviation of Gaussian filter in frames.

    Returns:
        smoothed_positions: (horizon, 22, 3) tensor of smoothed joint positions.
    """
    if sigma <= 0.0 or positions.shape[0] < 5:
        return positions
    radius = int(round(2.0 * sigma))
    kernel_size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=positions.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    T, J, D = positions.shape
    # Reshape to (1, J*D, T) for 1D depthwise convolution
    feat = positions.permute(1, 2, 0).reshape(1, J * D, T)
    weight = kernel.view(1, 1, kernel_size).expand(J * D, 1, -1)
    pad_feat = torch.nn.functional.pad(feat, (radius, radius), mode="replicate")
    smoothed = torch.nn.functional.conv1d(pad_feat, weight, groups=J * D)
    return smoothed.view(J, D, T).permute(2, 0, 1)


def apply_latent_filter(
    z: torch.Tensor,
    filter_type: Optional[str] = "savgol",
    sigma: float = 1.0,
    window_size: int = 5,
    polyorder: int = 2,
) -> torch.Tensor:
    """Applies temporal filtering along the sequence dimension (dim=1) of a latent tensor (B, T, D).

    Args:
        z: (B, T, D) latent tensor.
        filter_type: "savgol" | "gauss" | None.
        sigma: Standard deviation for Gaussian filter.
        window_size: Window length (odd integer, e.g. 5).
        polyorder: Polynomial order for Savitzky-Golay (default 2).

    Returns:
        Filtered latent tensor of shape (B, T, D).
    """
    if filter_type is None or filter_type.lower() == "none":
        return z

    filter_type = filter_type.lower()
    B, T, D = z.shape
    if T < window_size:
        return z

    pad = window_size // 2
    # Transpose to (B, D, T) for 1D convolution along temporal dimension
    z_t = z.transpose(1, 2)
    z_pad = torch.nn.functional.pad(z_t, (pad, pad), mode="replicate")

    if filter_type == "savgol":
        # Exact analytical Savitzky-Golay coefficients for quadratic polynomial (poly=2)
        if window_size == 5:
            weights = torch.tensor([-3.0, 12.0, 17.0, 12.0, -3.0], device=z.device, dtype=z.dtype) / 35.0
        elif window_size == 7:
            weights = torch.tensor([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], device=z.device, dtype=z.dtype) / 21.0
        elif window_size == 9:
            weights = torch.tensor([-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], device=z.device, dtype=z.dtype) / 231.0
        else:
            weights = torch.tensor([-3.0, 12.0, 17.0, 12.0, -3.0], device=z.device, dtype=z.dtype) / 35.0
        kernel = weights.view(1, 1, -1).repeat(D, 1, 1)
        z_smooth = torch.nn.functional.conv1d(z_pad, kernel, groups=D)
    elif filter_type == "gauss":
        coords = torch.arange(window_size, device=z.device, dtype=z.dtype) - pad
        weights = torch.exp(-0.5 * (coords / sigma) ** 2)
        weights = weights / weights.sum()
        kernel = weights.view(1, 1, -1).repeat(D, 1, 1)
        z_smooth = torch.nn.functional.conv1d(z_pad, kernel, groups=D)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}. Choose 'savgol', 'gauss', or None.")

    return z_smooth.transpose(1, 2)


@torch.no_grad()
def generate_motion_from_prompt(
    text_prompt: str,
    history_motion: torch.Tensor,     # (1, history_length, 271) raw
    initial_joints: torch.Tensor,     # (1, 22, 3) raw - seed positions
    initial_frame: torch.Tensor,      # (1, 271) raw - seed frame
    predictor_trainer: Any,
    decoder_trainer: Any,
    normalizer: FeatureNormalizer,
    clip_encoder: Any,
    device: torch.device | str = "cuda",
    num_inference_steps: int = 20,
    time_schedule_power: float = 1.0,
    guidance_scale: float = 2.0,
    guidance_min: float = 1.5,
    dynamic_cfg: bool = True,
    horizon: int = 80,
    history_valid_length: Optional[Union[int, torch.Tensor]] = None,
    solver: str = "midpoint",
    velocity_scale: float = 1.0,
    latent_filter: Optional[str] = "savgol",
    latent_filter_sigma: float = 1.0,
    latent_filter_window: int = 5,
    smooth_output: bool = True,
    smooth_sigma: float = 0.5,
    enforce_rigid_bones: bool = True,
    use_prefix_history: Optional[bool] = None,
    current_yaw: Optional[torch.Tensor] = None,
    ref_skeleton: Optional[torch.Tensor] = None,
    return_state: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Generate joint positions by integrating flow matching ODE in latent space.

    Args:
        text_prompt: Text description of the motion to generate.
        history_motion: Historical motion sequence (1, history_length, 271) RAW features.
        initial_joints: Seed joint positions (1, 22, 3) RAW features for initial frame.
        initial_frame: Seed frame features (1, 271) RAW features for frame 0.
        predictor_trainer: FlowMatchingTrainer instance.
        decoder_trainer: LatentDecoder trainer instance (or object with .ema_decoder).
        normalizer: FeatureNormalizer instance for raw feature normalization.
        clip_encoder: CLIP text sequence encoder.
        device: Device to execute generation on.
        num_inference_steps: Number of integration steps.
        time_schedule_power: Power parameter for ODE time discretization schedule (1.0 = Uniform/Linear).
        guidance_scale: Classifier-Free Guidance (CFG) scale (2.5 recommended).
        horizon: Target horizon sequence length (default 80 frames).
        history_valid_length: Exact count of non-padded history frames. Derived if None.
        solver: ODE solver to use ('midpoint' 2nd-order RK2 or 'euler' 1st-order).
        velocity_scale: Scaling factor for velocity vector to restore ground truth kinematic energy (default 1.15).
        smooth_output: Whether to apply temporal Gaussian smoothing to joint positions.
        smooth_sigma: Gaussian smoothing standard deviation in frames (default 1.2).

    Returns:
        generated_positions: (horizon, 22, 3) numpy array of generated joint positions.
    """
    target_encoder = decoder_trainer.ema_encoder.model
    target_encoder.eval()
    predictor = predictor_trainer.ema_predictor.model
    predictor.eval()
    decoder = decoder_trainer.ema_decoder.model
    decoder.eval()

    device = torch.device(device) if isinstance(device, str) else device

    # Step 1: Clear KV cache and reference state in predictor
    predictor.clear_cache()

    # Ensure single-sample batch dimension (1, ...) for all seed inputs
    if history_motion.ndim == 2:
        history_motion = history_motion.unsqueeze(0)
    elif history_motion.shape[0] > 1:
        history_motion = history_motion[:1]

    if initial_joints.ndim == 2:
        initial_joints = initial_joints.unsqueeze(0)
    elif initial_joints.shape[0] > 1:
        initial_joints = initial_joints[:1]

    if initial_frame.ndim == 1:
        initial_frame = initial_frame.unsqueeze(0)
    elif initial_frame.shape[0] > 1:
        initial_frame = initial_frame[:1]

    # Step 2: Encode text prompt to sequence and pooled sentence embedding
    text_seq = clip_encoder.encode_sequence([text_prompt]).to(device)  # (1, S, 512)
    # Match flow_matching_trainer.py:L516: text_pooled is sequence mean, NOT pooler_output
    text_pooled = text_seq.mean(dim=1)  # (1, 512)

    # Step 3: Compute history conditioning context
    zeros_hist = torch.zeros(1, text_pooled.shape[1], device=device, dtype=history_motion.dtype)
    history_norm = normalizer.normalize(history_motion.to(device))
    raw_hist = target_encoder(
        history_norm, zeros_hist, mask=None, return_layer_outputs=False
    ).detach()
    if hasattr(target_encoder, "get_latent"):
        track_features = target_encoder.get_latent(raw_hist).detach()  # (1, T_hist, latent_dim)
    else:
        track_features = raw_hist.detach()  # (1, T_hist, H_enc)

    T_hist = track_features.shape[1]
    S_text = text_seq.shape[1]
    S_cond = S_text + T_hist

    # Determine valid history length and construct boolean key-padding mask
    if history_valid_length is None:
        # Check if history_motion is all zeros (e.g. generation from scratch)
        is_all_zero = (history_motion == 0).all().item()
        val_len = 0 if is_all_zero else T_hist
    elif isinstance(history_valid_length, torch.Tensor):
        val_len = int(history_valid_length.item())
    else:
        val_len = int(history_valid_length)

    hist_idx = torch.arange(T_hist, device=device).unsqueeze(0)  # (1, T_hist)
    valid_start = T_hist - val_len
    hist_mask = hist_idx >= valid_start  # (1, T_hist) bool
    text_mask = torch.ones((1, S_text), dtype=torch.bool, device=device)  # (1, S_text) bool
    cond_mask = torch.cat([text_mask, hist_mask], dim=1)  # (1, S_cond) bool
    key_padding_mask = cond_mask.unsqueeze(1).unsqueeze(2)  # (1, 1, 1, S_cond) bool

    # Step 4: Sample noise for target sequence latents (T_target = horizon - 1 frames)
    T_target = horizon - 1
    latent_dim = getattr(
        target_encoder,
        "latent_dim",
        getattr(target_encoder.config, "latent_dim", getattr(getattr(predictor, "config", None), "predictor_config", None) and getattr(predictor.config.predictor_config, "latent_dim", 64) or 64),
    )
    pred_cfg_obj = getattr(getattr(predictor, "config", None), "predictor_config", None)
    use_brownian = getattr(pred_cfg_obj, "use_brownian_noise", True) if pred_cfg_obj is not None else True
    brownian_sigma = getattr(pred_cfg_obj, "brownian_sigma", 1.2) if pred_cfg_obj is not None else 1.2

    z_t = sample_initial_noise(
        (1, T_target, latent_dim),
        device=device,
        dtype=torch.float32,
        use_brownian=bool(use_brownian),
        sigma=float(brownian_sigma),
    )

    # Step 5: Integrate flow ODE from t=0 to t=1
    s = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    tau = s if time_schedule_power == 1.0 else 1.0 - (1.0 - s).pow(time_schedule_power)

    if callable(predictor_trainer.null_text_embedding):
        try:
            null_emb = predictor_trainer.null_text_embedding(1).to(dtype=text_seq.dtype)
        except TypeError:
            null_emb = predictor_trainer.null_text_embedding(torch.zeros(1, dtype=torch.long, device=device)).to(dtype=text_seq.dtype)
    elif isinstance(predictor_trainer.null_text_embedding, torch.Tensor):
        null_emb = predictor_trainer.null_text_embedding.to(dtype=text_seq.dtype)
    else:
        null_emb = torch.zeros(1, text_seq.shape[-1], device=device, dtype=text_seq.dtype)
    if null_emb.dim() == 1:
        null_emb = null_emb.unsqueeze(0)
    null_seq = null_emb.unsqueeze(1).expand(-1, text_seq.shape[1], -1)  # (1, S, 512)
    null_pooled = null_seq.mean(dim=1)  # (1, 512)

    if use_prefix_history is None:
        pred_cfg = getattr(getattr(predictor, "config", None), "predictor_config", None)
        if isinstance(pred_cfg, dict):
            use_prefix_history = pred_cfg.get("use_prefix_history", True)
        elif pred_cfg is not None:
            use_prefix_history = getattr(pred_cfg, "use_prefix_history", True)
        else:
            use_prefix_history = True

    if use_prefix_history:
        # Prefix in Self-Attention: cross-attention receives pure text sequence (1, S, 512)
        cond_cross = text_seq
        uncond_cross = null_seq
        hist_input = predictor_trainer.normalize_latent(track_features) if hasattr(predictor_trainer, "normalize_latent") else track_features
        cross_mask = None

        # Build prefix self-attention mask to mask out padded history frames
        device = z_t.device
        N = z_t.shape[1]
        hist_idx = torch.arange(T_hist, device=device).unsqueeze(0)
        valid_start = T_hist - val_len
        hist_valid = hist_idx >= valid_start  # (1, T_hist) bool
        target_valid = torch.ones((1, N), dtype=torch.bool, device=device)
        self_attn_mask = torch.cat([hist_valid, target_valid], dim=1).unsqueeze(1).unsqueeze(2)
    else:
        # Legacy concatenated cross-attention: cross-attention receives text + history
        combined_cond = torch.cat([text_seq, track_features], dim=1)  # (1, S + T_hist, D)
        combined_uncond = torch.cat([null_seq, track_features], dim=1)  # (1, S + T_hist, D)
        cond_cross = combined_cond
        uncond_cross = combined_uncond
        hist_input = None
        cross_mask = key_padding_mask
        self_attn_mask = None

    def eval_velocity(z_curr: torch.Tensor, t_curr: torch.Tensor) -> torch.Tensor:
        v_c, _, _ = predictor(
            noisy_states=z_curr,
            timesteps=t_curr,
            track_features=cond_cross,
            text_embedding=text_pooled,
            key_padding_mask=cross_mask,
            history_states=hist_input,
            self_attn_mask=self_attn_mask,
        )
        t_val = float(t_curr.mean().item())
        eff_cfg = guidance_scale - (guidance_scale - guidance_min) * t_val if dynamic_cfg and guidance_scale > 1.0 else guidance_scale
        if eff_cfg > 1.0:
            v_u, _, _ = predictor(
                noisy_states=z_curr,
                timesteps=t_curr,
                track_features=uncond_cross,
                text_embedding=null_pooled,
                key_padding_mask=cross_mask,
                history_states=hist_input,
                self_attn_mask=self_attn_mask,
            )
            v_eff = v_u + eff_cfg * (v_c - v_u)
        else:
            v_eff = v_c
        return v_eff * velocity_scale

    for step in range(num_inference_steps):
        t_start = tau[step].expand(1)
        dt = (tau[step + 1] - tau[step]).item()

        if solver == "midpoint":
            v1 = eval_velocity(z_t, t_start)
            t_mid = (tau[step] + 0.5 * dt).expand(1)
            z_mid = z_t + 0.5 * dt * v1
            v_mid = eval_velocity(z_mid, t_mid)
            z_t = z_t + v_mid * dt
        else:
            v1 = eval_velocity(z_t, t_start)
            z_t = z_t + v1 * dt

    # Final integrated latent in normalized space
    z1_pred_norm = z_t  # (1, T_target, H_enc)

    # Step 6: Denormalize latent back to encoder's native latent scale before passing to decoder
    z1_pred = predictor_trainer.denormalize_latent(z1_pred_norm)

    # Apply Latent-Space Spline Filter ("savgol" / "gauss" / None)
    z1_pred = apply_latent_filter(
        z1_pred,
        filter_type=latent_filter,
        sigma=latent_filter_sigma,
        window_size=latent_filter_window,
    )

    decoded_68d = decoder(z1_pred)  # (1, T_target, 68)

    # Step 7: Step-by-step autoregressive reconstruction of joint positions
    current_pos = initial_joints.to(device)  # (1, 22, 3)
    current_frame = normalizer.normalize(initial_frame.to(device))  # (1, 271)
    if ref_skeleton is None:
        ref_skeleton = initial_joints.to(device)
    else:
        ref_skeleton = ref_skeleton.to(device)

    prev_x271_raw = normalizer.denormalize(current_frame)
    if current_yaw is None:
        current_yaw = root_rot6d_to_yaw(prev_x271_raw[:, Features.ROOT_ROT6D])

    # Check whether the loaded x68_to_positions function supports monotonic yaw tracking
    supports_yaw = hasattr(x68_to_positions, "__code__") and "current_yaw" in x68_to_positions.__code__.co_varnames

    generated_positions = [current_pos[0].cpu()]
    all_raw_frames = [prev_x271_raw]

    for frame_idx in range(T_target):
        pred_68d = decoded_68d[:, frame_idx]  # (1, 68)

        if supports_yaw:
            new_pos, current_yaw = x68_to_positions(
                pred_68d,
                normalizer,
                current_frame,
                current_pos,
                current_yaw=current_yaw,
                ref_skeleton=ref_skeleton,
                return_yaw=True,
            )  # (1, 22, 3)
        else:
            new_pos = x68_to_positions(
                pred_68d,
                normalizer,
                current_frame,
                current_pos,
            )

        new_frame, _ = positions_to_x271(new_pos, current_pos, normalizer)

        current_pos = new_pos
        current_frame = new_frame
        generated_positions.append(new_pos[0].cpu())
        all_raw_frames.append(normalizer.denormalize(new_frame))

    positions_tensor = torch.stack(generated_positions, dim=0)  # (horizon, 22, 3)
    if enforce_rigid_bones:
        positions_tensor = enforce_rigid_bone_lengths(positions_tensor, ref_skeleton[0].cpu())
    if smooth_output:
        positions_tensor = temporal_gaussian_smooth(positions_tensor, sigma=smooth_sigma)

    if return_state:
        raw_seq = torch.cat(all_raw_frames, dim=0)  # (horizon, 271)
        if raw_seq.shape[0] >= 40:
            hist_40 = raw_seq[-40:].unsqueeze(0)
            valid_hist_len = 40
        else:
            pad_len = 40 - raw_seq.shape[0]
            pad_frames = raw_seq[0:1].expand(pad_len, -1)
            hist_40 = torch.cat([pad_frames, raw_seq], dim=0).unsqueeze(0)
            valid_hist_len = raw_seq.shape[0]

        last_state = {
            "history_motion": hist_40,
            "initial_joints": current_pos,
            "initial_frame": normalizer.denormalize(current_frame),
            "current_yaw": current_yaw,
            "ref_skeleton": ref_skeleton,
            "history_valid_length": valid_hist_len,
        }
        return positions_tensor.cpu().numpy(), last_state

    return positions_tensor.cpu().numpy()


def get_neutral_standing_history(
    normalizer: FeatureNormalizer,
    history_length: int = 40,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Constructs neutral standing pose history frames (1, history_length, 271) raw from normalizer mean.
    
    Using dataset mean root and resting joint angles avoids artificial 'dead stop' momentum shocks
    when generating motions from rest.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    mean_frame = normalizer.mean.to(dev).view(1, 1, -1)
    return mean_frame.expand(1, history_length, -1).clone()


def get_neutral_standing_seed(
    normalizer: FeatureNormalizer,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs a consistent (seed_joints, seed_frame) pair from dataset mean.
    
    Guarantees that seed_joints (1, 22, 3) and seed_frame (1, 271) are derived
    from the exact same anatomical resting pose, eliminating first-frame teleportation
    and bone-length distortions.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    norm_mean = torch.zeros(1, 271, device=dev)
    seed_joints = x271_to_positions(norm_mean, normalizer)  # (1, 22, 3)
    seed_frame = normalizer.mean.to(dev).view(1, -1)        # (1, 271)
    return seed_joints, seed_frame


@torch.no_grad()
def generate_chained_motion(
    stages: List[Tuple[str, int]],
    predictor_trainer: Any,
    decoder_trainer: Any,
    normalizer: FeatureNormalizer,
    clip_encoder: Any,
    device: torch.device | str = "cuda",
    num_inference_steps: int = 20,
    time_schedule_power: float = 1.0,
    guidance_scale: float = 2.5,
    guidance_min: float = 1.2,
    dynamic_cfg: bool = True,
    solver: str = "midpoint",
    velocity_scale: float = 1.0,
    latent_filter: Optional[str] = "savgol",
    latent_filter_sigma: float = 1.0,
    latent_filter_window: int = 5,
    smooth_output: bool = True,
    smooth_sigma: float = 0.5,
    enforce_rigid_bones: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Generate multi-stage narrative continuous motion by chaining sequential flow matching inferences.

    Args:
        stages: List of (prompt, horizon) tuples defining sequential actions.
        predictor_trainer: FlowMatchingTrainer instance.
        decoder_trainer: LatentDecoder trainer instance.
        normalizer: FeatureNormalizer instance.
        clip_encoder: CLIP text sequence encoder.
        device: Device to execute generation on.
        num_inference_steps: Number of integration steps per stage.
        time_schedule_power: Discretization schedule power.
        guidance_scale: Initial CFG scale.
        guidance_min: Minimum CFG scale at t=1.
        dynamic_cfg: Whether to anneal CFG linearly.
        solver: ODE solver ('midpoint' or 'euler').
        velocity_scale: Velocity multiplier.
        latent_filter: Latent smoothing filter ('savgol' or 'gauss').
        latent_filter_sigma: Smoothing sigma.
        latent_filter_window: Savgol filter window size.
        smooth_output: Apply Gaussian smoothing across the entire trajectory.
        smooth_sigma: Global Gaussian smoothing standard deviation.
        enforce_rigid_bones: Enforce rigid anatomical bone lengths across the sequence.
        verbose: Print progress per stage.

    Returns:
        concatenated_positions: (total_horizon, 22, 3) numpy array of seamless chained joint positions.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    seed_joints, seed_frame = get_neutral_standing_seed(normalizer, device=dev)
    neutral_history = get_neutral_standing_history(normalizer, history_length=40, device=dev)

    state = {
        "history_motion": neutral_history,
        "initial_joints": seed_joints,
        "initial_frame": seed_frame,
        "current_yaw": None,
        "ref_skeleton": seed_joints,
        "history_valid_length": 0,  # Cold start from rest for stage 1
    }

    all_stage_positions = []

    for stage_idx, (prompt, stage_horizon) in enumerate(stages):
        if verbose:
            print(f"[Stage {stage_idx + 1}/{len(stages)}] Generating {stage_horizon} frames: \"{prompt}\" (hist_len={state['history_valid_length']})...")

        pos_np, next_state = generate_motion_from_prompt(
            text_prompt=prompt,
            history_motion=state["history_motion"],
            initial_joints=state["initial_joints"],
            initial_frame=state["initial_frame"],
            predictor_trainer=predictor_trainer,
            decoder_trainer=decoder_trainer,
            normalizer=normalizer,
            clip_encoder=clip_encoder,
            device=dev,
            num_inference_steps=num_inference_steps,
            time_schedule_power=time_schedule_power,
            guidance_scale=guidance_scale,
            guidance_min=guidance_min,
            dynamic_cfg=dynamic_cfg,
            horizon=stage_horizon,
            history_valid_length=state["history_valid_length"],
            solver=solver,
            velocity_scale=velocity_scale,
            latent_filter=latent_filter,
            latent_filter_sigma=latent_filter_sigma,
            latent_filter_window=latent_filter_window,
            smooth_output=False,       # Smooth globally at the end
            enforce_rigid_bones=False, # Enforce globally at the end
            current_yaw=state["current_yaw"],
            ref_skeleton=seed_joints,
            return_state=True,
        )

        if stage_idx == 0:
            all_stage_positions.append(pos_np)
        else:
            # pos_np[0] == previous_stage[-1], so skip frame 0 for seamless handoff
            all_stage_positions.append(pos_np[1:])

        state = next_state
        state["ref_skeleton"] = seed_joints  # Keep consistent anatomical base

    # Concatenate all stages into a continuous trajectory
    stitched_positions = np.concatenate(all_stage_positions, axis=0)  # (total_T, 22, 3)
    pos_tensor = torch.from_numpy(stitched_positions).float()

    if enforce_rigid_bones:
        pos_tensor = enforce_rigid_bone_lengths(pos_tensor, seed_joints[0].cpu())
    if smooth_output:
        pos_tensor = temporal_gaussian_smooth(pos_tensor, sigma=smooth_sigma)

    if verbose:
        print(f"[Done] Total generated chained motion: {pos_tensor.shape[0]} frames ({pos_tensor.shape[0] / 20.0:.2f} seconds)")

    return pos_tensor.cpu().numpy()

