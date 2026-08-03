"""
Inference pipeline for Text-to-3D Motion generation using FlowMatchingPredictor and MotionHistoryEncoder.
"""

from typing import Any, Optional, Union

import numpy as np
import torch

from utils.motion_utils import FeatureNormalizer, positions_to_x271, x68_to_positions


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
    time_schedule_power: float = 3.0,
    guidance_scale: float = 2.5,
    horizon: int = 80,
    history_valid_length: Optional[Union[int, torch.Tensor]] = None,
) -> np.ndarray:
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
        num_inference_steps: Number of Euler integration steps.
        time_schedule_power: Power parameter for ODE time discretization schedule.
        guidance_scale: Classifier-Free Guidance (CFG) scale.
        horizon: Target horizon sequence length (default 80 frames).
        history_valid_length: Exact count of non-padded history frames. Derived if None.

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

    # Step 2: Encode text prompt to sequence embedding
    text_seq = clip_encoder.encode_sequence([text_prompt]).to(device)  # (1, S, 512)
    text_pooled_ref = text_seq.mean(dim=1)  # for zeros sizing

    # Step 3: Compute history conditioning context
    zeros_hist = torch.zeros(1, text_pooled_ref.shape[1], device=device, dtype=history_motion.dtype)
    history_norm = normalizer.normalize(history_motion.to(device))
    track_features = target_encoder(
        history_norm, zeros_hist, mask=None, return_layer_outputs=False
    ).detach()  # (1, T_hist, H_enc)

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
    z_t = torch.randn(1, T_target, target_encoder.config.hidden_size, device=device)

    # Step 5: Integrate flow ODE from t=0 to t=1
    s = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    tau = 1.0 - (1.0 - s).pow(time_schedule_power)

    for step in range(num_inference_steps):
        t_start = tau[step].expand(1)
        t_end = tau[step + 1]
        dt = t_end - t_start

        # Conditioned branch: Concatenate text sequence and history context
        combined_cond = torch.cat([text_seq, track_features], dim=1)  # (1, S + T_hist, 512)
        text_pooled = text_seq.mean(dim=1)  # (1, 512)

        v_cond, _, _ = predictor(
            noisy_states=z_t,
            timesteps=t_start,
            track_features=combined_cond,
            text_embedding=text_pooled,
            key_padding_mask=key_padding_mask,
        )

        # Unconditioned branch: Concatenate null sequence and history context
        null_emb = predictor_trainer.null_text_embedding(1).to(dtype=text_seq.dtype)  # (1, 512)
        null_seq = null_emb.unsqueeze(1).expand(-1, text_seq.shape[1], -1)  # (1, S, 512)
        combined_uncond = torch.cat([null_seq, track_features], dim=1)  # (1, S + T_hist, 512)
        null_pooled = null_seq.mean(dim=1)  # (1, 512)

        v_uncond, _, _ = predictor(
            noisy_states=z_t,
            timesteps=t_start,
            track_features=combined_uncond,
            text_embedding=null_pooled,
            key_padding_mask=key_padding_mask,
        )

        v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
        z_t = z_t + v_t * dt

    # Final integrated latent in normalized space
    z1_pred_norm = z_t  # (1, T_target, H_enc)

    # Step 6: Denormalize latent back to encoder's native latent scale before passing to decoder
    z1_pred = predictor_trainer.denormalize_latent(z1_pred_norm)
    decoded_68d = decoder(z1_pred)  # (1, T_target, 68)

    # Step 7: Step-by-step autoregressive reconstruction of joint positions
    current_pos = initial_joints.to(device)  # (1, 22, 3)
    current_frame = normalizer.normalize(initial_frame.to(device))  # (1, 271)

    generated_positions = [current_pos[0].cpu()]

    for frame_idx in range(T_target):
        pred_68d = decoded_68d[:, frame_idx]  # (1, 68)

        new_pos = x68_to_positions(
            pred_68d,
            normalizer,
            current_frame,
            current_pos,
        )  # (1, 22, 3)

        new_frame, _ = positions_to_x271(new_pos, current_pos, normalizer)

        current_pos = new_pos
        current_frame = new_frame
        generated_positions.append(new_pos[0].cpu())

    return torch.stack(generated_positions, dim=0).numpy()
