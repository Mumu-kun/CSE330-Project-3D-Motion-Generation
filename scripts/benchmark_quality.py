#!/usr/bin/env python3
"""Comprehensive Quantitative Benchmark: Predictor Quality & Kinematic Motion Quality.

Evaluates the Phase 3 trained 64D Flow Matching Predictor against Ground Truth and
reconstructs 3D kinematic skeletons via the 64D->72D LatentDecoder.

Computes:
1. Predictor Quality:
   - Mean Latent Cosine Similarity (z_pred vs z_gt)
   - Relative Latent L2 Error (%)
   - Text Sensitivity / Caption Shuffle Delta (%)
   - Flow Inversion Cycle Reconstruction MSE & CosSim

2. Motion Quality:
   - Kinetic Path Length Ratio (zigzag / smoothness diagnostic)
   - History Boundary Anchor Gap at frame 0
   - Autoregressive 3D Joint MPJPE (mm)
   - Action Follow-Through / Range of Motion (wrist displacement in m)
   - Bone Length Variance (mm drift)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import Config
from utils.dataset import create_dataloader
from utils.models.flow_matching_predictor import FlowMatchingPredictor
from utils.models.motion_history_encoder import MotionHistoryEncoder
from utils.models.decoder_trainer import LatentDecoder
from utils.text_encoder import CLIPEncoder
from utils.motion_utils import (
    Features,
    x72_to_positions,
    positions_to_x271,
    root_rot6d_to_yaw,
)


def run_benchmark(
    pred_ckpt_path: str = "src/checkpoints/phase3/predictor/phase3_predictor_best_val_20261017_040556.pt",
    pretrain_ckpt_path: str = "src/checkpoints/pretrain/pretrain_best_val_20261017_005458_425.pt",
    dec_ckpt_path: str = "src/checkpoints/phase3/decoder/finetune_best_val_20261017_011538_200.pt",
    num_samples: int = 16,
    num_ode_steps: int = 20,
    guidance_scale: float = 2.4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"QUANTITATIVE BENCHMARK: PREDICTOR & MOTION QUALITY")
    print(f"Device: {device} | Evaluated Samples: {num_samples} | ODE Steps: {num_ode_steps}")
    print(f"Predictor Checkpoint: {Path(pred_ckpt_path).name}")
    print("=" * 80)

    # 1. Config
    config = Config()
    config.device = device
    config.latent_dim = 64
    config.encoder_config.latent_dim = 64
    config.decoder_config.latent_dim = 64
    config.predictor_config.latent_dim = 64
    config.dataset_path = PROJECT_ROOT / "src" / "dataset" / "humanml3d-subset"
    config.horizon = 80
    config.history_length = 40
    config.batch_size = max(num_samples, 16)
    config.num_workers = 0

    val_loader, normalizer = create_dataloader(config, "val", shuffle=False)
    val_loader.dataset.set_horizon(config.horizon)
    batch = next(iter(val_loader))

    # 2. Load Models
    print("\n[1/3] Loading 64D Geodesic Pipeline Models...")
    pred_data = torch.load(pred_ckpt_path, map_location=device, weights_only=False)

    # Encoder (prefer exact encoder_ema saved in predictor checkpoint)
    encoder = MotionHistoryEncoder(config).to(device)
    if "encoder_ema" in pred_data:
        encoder.load_state_dict(pred_data["encoder_ema"], strict=False)
        print("   Loaded Encoder from predictor checkpoint (encoder_ema)")
    elif "encoder" in pred_data:
        encoder.load_state_dict(pred_data["encoder"], strict=False)
        print("   Loaded Encoder from predictor checkpoint (encoder)")
    else:
        enc_data = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(enc_data["encoder_ema"] if "encoder_ema" in enc_data else enc_data["encoder"], strict=False)
        print(f"   Loaded Encoder from {Path(pretrain_ckpt_path).name}")
    encoder.eval()

    # Decoder
    decoder = LatentDecoder(config).to(device)
    dec_data = torch.load(dec_ckpt_path, map_location=device, weights_only=False)
    decoder.load_state_dict(dec_data["decoder_ema"] if "decoder_ema" in dec_data else dec_data["decoder"], strict=False)
    decoder.eval()
    print(f"   Loaded Decoder from {Path(dec_ckpt_path).name}")

    # Predictor
    predictor = FlowMatchingPredictor(config).to(device)
    predictor.load_state_dict(pred_data["predictor_ema"] if "predictor_ema" in pred_data else pred_data["predictor"], strict=False)
    predictor.eval()
    print(f"   Loaded Predictor from {Path(pred_ckpt_path).name}")

    # Latent stats
    latent_stats = pred_data.get("latent_stats", None)
    if latent_stats is not None:
        l_mean = latent_stats["mean"].to(device)
        l_std = latent_stats["std"].to(device).clamp(min=1e-5)
        print(f"   Loaded latent normalization stats: mean norm={l_mean.norm().item():.3f}, std norm={l_std.norm().item():.3f}")
    else:
        l_mean = None
        l_std = None

    clip_encoder = CLIPEncoder().to(device)

    # 3. Prepare Batch Data
    K = min(num_samples, batch["motion"].shape[0])
    motion_raw = batch["motion"][:K].to(device)
    history_raw = batch["history_motion"][:K].to(device)
    joints_gt = batch["joints"][:K].to(device) if "joints" in batch else None
    captions = batch["captions"][:K]

    target_norm = normalizer.normalize(motion_raw)
    history_norm = normalizer.normalize(history_raw)

    with torch.no_grad():
        zeros = torch.zeros(K, 512, device=device, dtype=target_norm.dtype)
        raw_target = encoder(target_norm, zeros, mask=None, return_layer_outputs=True)
        hidden_target = raw_target[:, 1:, -1, :]
        z1_gt = encoder.get_latent(hidden_target) if hasattr(encoder, "get_latent") else hidden_target

        raw_hist = encoder(history_norm, zeros, mask=None, return_layer_outputs=False)
        track_features = encoder.get_latent(raw_hist) if hasattr(encoder, "get_latent") else raw_hist

        if l_mean is not None:
            z1_gt_norm = (z1_gt - l_mean) / l_std
            hist_norm_latent = (track_features - l_mean) / l_std
        else:
            z1_gt_norm = z1_gt
            hist_norm_latent = track_features

        text_seq = clip_encoder.encode_sequence(captions).to(device)
        text_pooled = text_seq.mean(dim=1)

        # Robust null text embedding extraction
        null_data = pred_data.get("null_text_embedding", None)
        null_vec = None
        if null_data is not None:
            if isinstance(null_data, dict):
                null_vec = null_data.get("weight", null_data.get("embedding.weight", None))
            elif hasattr(null_data, "weight"):
                null_vec = null_data.weight
            elif isinstance(null_data, torch.Tensor):
                null_vec = null_data

        if null_vec is not None:
            null_t = null_vec.to(device=device, dtype=text_seq.dtype)
            if null_t.dim() == 1:
                null_t = null_t.unsqueeze(0)
            null_seq = null_t.unsqueeze(1).expand(K, text_seq.shape[1], -1)
            null_pooled = null_t.expand(K, -1)
            print("   Loaded learned null_text_embedding from checkpoint")
        else:
            null_seq = torch.zeros_like(text_seq)
            null_pooled = torch.zeros_like(text_pooled)
            print("   Warning: Could not extract null_text_embedding; using zero fallback")

    # ── METRIC 1: PREDICTOR QUALITY ─────────────────────────────────────────────
    print("\n[2/3] Computing Predictor Quality Metrics (Flow Vector Field & Latent Alignment)...")

    # A. 20-Step ODE Sampling
    dt = 1.0 / num_ode_steps
    torch.manual_seed(42)
    # Brownian-smoothed z0
    eps = torch.randn_like(z1_gt_norm)
    kernel = torch.tensor([0.061, 0.242, 0.383, 0.242, 0.061], device=device).view(1, 1, -1)
    eps_reshaped = eps.permute(0, 2, 1).contiguous()
    eps_smooth = F.conv1d(eps_reshaped, kernel.expand(64, 1, -1), padding=2, groups=64)
    z0 = eps_smooth.permute(0, 2, 1).contiguous()
    z0 = z0 / (z0.std(dim=(0, 1), keepdim=True) + 1e-6)

    z = z0.clone()
    with torch.no_grad():
        for step in range(num_ode_steps):
            t_curr = step * dt
            t_in = torch.full((K,), t_curr, device=device)
            v_c, _, _ = predictor(z, t_in, track_features=text_seq, text_embedding=text_pooled, history_states=hist_norm_latent)
            if guidance_scale > 1.0:
                v_u, _, _ = predictor(z, t_in, track_features=null_seq, text_embedding=null_pooled, history_states=hist_norm_latent)
                v = v_u + guidance_scale * (v_c - v_u)
            else:
                v = v_c
            z_mid = z + 0.5 * dt * v
            t_mid = torch.full((K,), t_curr + 0.5 * dt, device=device)
            v_mid_c, _, _ = predictor(z_mid, t_mid, track_features=text_seq, text_embedding=text_pooled, history_states=hist_norm_latent)
            if guidance_scale > 1.0:
                v_mid_u, _, _ = predictor(z_mid, t_mid, track_features=null_seq, text_embedding=null_pooled, history_states=hist_norm_latent)
                v_mid = v_mid_u + guidance_scale * (v_mid_c - v_mid_u)
            else:
                v_mid = v_mid_c
            z = z + v_mid * dt

    z_pred_norm = z
    z_pred = (z_pred_norm * l_std + l_mean) if l_mean is not None else z_pred_norm

    # 1. Cosine Similarity & Relative L2 Error
    cos_sim_per_sample = F.cosine_similarity(z_pred_norm, z1_gt_norm, dim=-1).mean(dim=1).cpu().numpy()
    mean_cos_sim = float(cos_sim_per_sample.mean())
    rel_l2_error = float((torch.norm(z_pred_norm - z1_gt_norm, dim=-1) / (torch.norm(z1_gt_norm, dim=-1) + 1e-6)).mean().item()) * 100.0

    # 2. Text Conditioning Sensitivity (Caption Shuffle Test)
    shuf_indices = list(range(1, K)) + [0]
    text_seq_shuf = text_seq[shuf_indices]
    text_pooled_shuf = text_pooled[shuf_indices]
    t_mid_eval = torch.full((K,), 0.5, device=device)
    z_eval = 0.5 * z0 + 0.5 * z1_gt_norm
    v_target = z1_gt_norm - z0

    with torch.no_grad():
        v_matched, _, _ = predictor(z_eval, t_mid_eval, track_features=text_seq, text_embedding=text_pooled, history_states=hist_norm_latent)
        loss_matched = F.mse_loss(v_matched, v_target).item()

        v_shuf, _, _ = predictor(z_eval, t_mid_eval, track_features=text_seq_shuf, text_embedding=text_pooled_shuf, history_states=hist_norm_latent)
        loss_shuf = F.mse_loss(v_shuf, v_target).item()

        v_null, _, _ = predictor(z_eval, t_mid_eval, track_features=null_seq, text_embedding=null_pooled, history_states=hist_norm_latent)
        loss_null = F.mse_loss(v_null, v_target).item()

    text_surge_pct = ((loss_shuf - loss_matched) / loss_matched) * 100.0
    null_surge_pct = ((loss_null - loss_matched) / loss_matched) * 100.0

    # 3. Flow Inversion Cycle Reconstruction (Lossless Vector Field Test)
    z_inv = z1_gt_norm.clone()
    with torch.no_grad():
        for step in range(num_ode_steps - 1, -1, -1):
            t_curr = (step + 1) * dt
            t_in = torch.full((K,), t_curr, device=device)
            v_inv, _, _ = predictor(z_inv, t_in, track_features=text_seq, text_embedding=text_pooled, history_states=hist_norm_latent)
            z_inv = z_inv - dt * v_inv
        z_reconstructed = z_inv.clone()
        for step in range(num_ode_steps):
            t_curr = step * dt
            t_in = torch.full((K,), t_curr, device=device)
            v_fwd, _, _ = predictor(z_reconstructed, t_in, track_features=text_seq, text_embedding=text_pooled, history_states=hist_norm_latent)
            z_reconstructed = z_reconstructed + dt * v_fwd

    cycle_mse = float(F.mse_loss(z_reconstructed, z1_gt_norm).item())
    cycle_cos = float(F.cosine_similarity(z_reconstructed.flatten(0, 1), z1_gt_norm.flatten(0, 1), dim=-1).mean().item())

    # ── METRIC 2: MOTION & KINEMATIC QUALITY ────────────────────────────────────
    print("\n[3/3] Computing Kinematic Motion Quality Metrics (3D Skeletons)...")

    # 1. Kinetic Path Ratio (Differential Arc Length)
    diff_pred = torch.diff(z_pred_norm, dim=1).norm(dim=-1).sum(dim=1).cpu().numpy()
    diff_gt = torch.diff(z1_gt_norm, dim=1).norm(dim=-1).sum(dim=1).cpu().numpy()
    path_ratios = diff_pred / (diff_gt + 1e-6)
    mean_path_ratio = float(np.mean(path_ratios))

    # 2. History Anchor Gap at frame 0
    hist_f_last = hist_norm_latent[:, -1, :]
    pred_f_first = z_pred_norm[:, 0, :]
    anchor_gaps = (pred_f_first - hist_f_last).norm(dim=-1).cpu().numpy()
    mean_anchor_gap = float(np.mean(anchor_gaps))

    # 3. Decode into 3D Skeletal Joint Positions (mm accuracy)
    mpjpe_oracle_list = []
    mpjpe_generated_list = []
    drift_list = []
    wrist_rom_list = []

    with torch.no_grad():
        decoded_gt = decoder(z1_gt)       # (K, T_target, 72)
        decoded_pred = decoder(z_pred)   # (K, T_target, 72)

    T_target = decoded_pred.shape[1]

    for k in range(K):
        gt_pos = joints_gt[k, 1:T_target+1].cpu().numpy()  # (T, 22, 3)
        ref_skeleton = joints_gt[k:k+1, 0]

        # 1-Step Teacher-Forced Decoder Oracle MPJPE (Exact Cell 10 benchmark)
        rec_tf = []
        for f in range(T_target):
            pred_72d = decoded_gt[k:k+1, f]
            p_frame = target_norm[k:k+1, f]
            p_pos = joints_gt[k:k+1, f]
            new_pos = x72_to_positions(
                pred_72d, normalizer, prev_x271=p_frame, prev_positions=p_pos, ref_skeleton=ref_skeleton
            )
            rec_tf.append(new_pos[0].cpu().numpy())
        oracle_pos = np.stack(rec_tf, axis=0)
        oracle_mpjpe = np.linalg.norm(oracle_pos - gt_pos, axis=-1).mean() * 1000.0
        mpjpe_oracle_list.append(oracle_mpjpe)

        # Generated motion (from flow matching z_pred)
        current_pos = joints_gt[k:k+1, 0]
        current_frame = target_norm[k:k+1, 0]
        prev_raw = normalizer.denormalize(current_frame)
        current_yaw = root_rot6d_to_yaw(prev_raw[:, Features.ROOT_ROT6D])

        rec_gen = []
        for f in range(T_target):
            p_72 = decoded_pred[k:k+1, f]
            new_pos, current_yaw = x72_to_positions(
                p_72, normalizer, prev_x271=current_frame, prev_positions=current_pos,
                current_yaw=current_yaw, ref_skeleton=ref_skeleton, return_yaw=True
            )
            new_frame, _ = positions_to_x271(new_pos, current_pos, normalizer)
            current_pos = new_pos
            current_frame = new_frame
            rec_gen.append(new_pos[0].cpu().numpy())

        gen_pos = np.stack(rec_gen, axis=0)
        gen_mpjpe = np.linalg.norm(gen_pos - gt_pos, axis=-1).mean() * 1000.0
        mpjpe_generated_list.append(gen_mpjpe)

        # Joint drift: root displacement error at terminal frame 79
        terminal_drift = np.linalg.norm(gen_pos[-1, 0] - gt_pos[-1, 0]) * 1000.0
        drift_list.append(terminal_drift)

        # Range of motion: Right wrist (joint 19) peak-to-peak displacement
        rwrist_rom = float(np.ptp(gen_pos[:, 19], axis=0).max())
        wrist_rom_list.append(rwrist_rom)

    # ── FINAL REPORT ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("                    FINAL QUANTITATIVE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\n1. PREDICTOR QUALITY NUMBERS (Flow Vector Field & Conditioning):")
    print(f"   • Mean Latent Cosine Similarity : {mean_cos_sim:6.3f}   (Baseline: 0.630  | Target: >0.800)")
    print(f"   • Relative Latent L2 Error      : {rel_l2_error:6.2f}%   (Baseline: 74.6%  | Target: <30.0%)")
    print(f"   • Text Sensitivity (Shuffle ΔL) : {text_surge_pct:+6.2f}%   (Baseline: +5.8%  | Target: >+20.0%)")
    print(f"   • Unconditional Gap (Null ΔL)   : {null_surge_pct:+6.2f}%   (Baseline: +88.8% | Prior distinction)")
    print(f"   • Cycle Inversion MSE           : {cycle_mse:6.4f}   (Target: <0.0500  | Invertible ODE)")
    print(f"   • Cycle Inversion CosSim        : {cycle_cos:6.4f}   (Target: >0.9800  | Conservation)")

    print(f"\n2. KINEMATIC & MOTION QUALITY NUMBERS (3D Skeleton & Physical Reality):")
    print(f"   • Kinetic Path Length Ratio     : {mean_path_ratio:6.2f}x   (Baseline: 8.51x  | Target: 1.00x-1.15x)")
    print(f"   • History Continuity Anchor Gap : {mean_anchor_gap:6.3f}    (Target: <1.000   | Smooth transition)")
    print(f"   • Decoder Oracle MPJPE          : {np.mean(mpjpe_oracle_list):6.2f} mm (Baseline: 53.9 mm | Target: <42.0 mm)")
    print(f"   • End-to-End Generated MPJPE    : {np.mean(mpjpe_generated_list):6.2f} mm (Decoder + Predictor combined)")
    print(f"   • 80-Frame Terminal Root Drift  : {np.mean(drift_list):6.2f} mm (Baseline: 193 mm | Target: <35.0 mm)")
    print(f"   • Limb Range of Motion (Wrist)  : {np.mean(wrist_rom_list):6.2f} m  (Frozen was: <0.15m | Target: >0.45m)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_ckpt", type=str, default="src/checkpoints/phase3/predictor/phase3_predictor_best_val_20261017_040556.pt")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.4)
    args = parser.parse_args()

    run_benchmark(
        pred_ckpt_path=args.pred_ckpt,
        num_samples=args.samples,
        num_ode_steps=args.steps,
        guidance_scale=args.guidance,
    )
