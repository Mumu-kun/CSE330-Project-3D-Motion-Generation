import sys
import traceback
from pathlib import Path
sys.path.insert(0, 'src')
import torch
import torch.nn.functional as F
import numpy as np

try:
    from utils.config import Config
    from utils.models.flow_matching_predictor import FlowMatchingPredictor
    from utils.models.motion_history_encoder import MotionHistoryEncoder
    from utils.models.decoder_trainer import LatentDecoder
    from utils.motion_utils import enforce_rigid_bone_lengths, x72_to_positions
    from utils.dataset import create_dataloader

    device = 'cpu'
    config = Config()
    config.dataset_path = Path('src/dataset/humanml3d-subset')
    config.num_workers = 0

    ckpt = torch.load('checkpoints/phase3/predictor/phase3_predictor_best_val_20260830_003547.pt', map_location=device, weights_only=False)
    predictor = FlowMatchingPredictor(config).to(device)
    predictor.load_state_dict(ckpt['predictor_ema'] if 'predictor_ema' in ckpt else ckpt['predictor'], strict=False)
    predictor.eval()

    dec_ckpt = torch.load('checkpoints/phase3/decoder/phase3_decoder_best_val_20260730_221759.pt', map_location=device, weights_only=False)
    encoder = MotionHistoryEncoder(config).to(device)
    encoder.load_state_dict(dec_ckpt['encoder_ema'] if 'encoder_ema' in dec_ckpt else dec_ckpt['encoder'])
    encoder.eval()

    decoder = LatentDecoder(config).to(device)
    decoder.load_state_dict(dec_ckpt['decoder_ema'] if 'decoder_ema' in dec_ckpt else dec_ckpt['decoder'])
    decoder.eval()

    val_loader, normalizer = create_dataloader(config, 'val', shuffle=False)
    b = next(iter(val_loader))

    # Use first 16 samples for snappy CPU evaluation
    K = min(16, b['motion'].shape[0])
    motion_raw = b['motion'][:K].to(device)
    history_raw = b['history_motion'][:K].to(device)
    target_motion = normalizer.normalize(motion_raw)
    history_motion = normalizer.normalize(history_raw)

    with torch.no_grad():
        zeros_text = torch.zeros(K, 512, device=device)
        raw_target = encoder(target_motion, zeros_text, mask=None, return_layer_outputs=True)
        z1_gt = raw_target[:, 1:, -1, :]
        track_features = encoder(history_motion, zeros_text, mask=None, return_layer_outputs=False)

    B, T, H = z1_gt.shape
    z1_gt_norm = z1_gt / 1.15

    text_seq = b['text_clip'][:K].to(device)
    text_seq = text_seq[:, -1:, :] if text_seq.ndim == 3 else text_seq.unsqueeze(1)
    text_pooled = text_seq.mean(dim=1)
    combined_cond = torch.cat([text_seq, track_features], dim=1)

    num_steps = 20
    velocity_scale = 1.12
    tau = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    def eval_v(z, t, use_text=True):
        tp = text_pooled if use_text else torch.zeros_like(text_pooled)
        cc = combined_cond if use_text else torch.cat([torch.zeros_like(text_seq), track_features], dim=1)
        v, _, _ = predictor(z, t, track_features=cc, text_embedding=tp, history_states=track_features)
        return v * velocity_scale

    # 1. Flow Inversion & Cycle Reconstruction (Lossless Vector Field Test)
    z = z1_gt_norm.clone()
    for step in range(num_steps - 1, -1, -1):
        t_start = tau[step + 1].expand(B)
        dt = -(tau[step + 1] - tau[step]).item()
        v1 = eval_v(z, t_start)
        t_mid = (tau[step + 1] + 0.5 * dt).expand(B)
        z_mid = z + 0.5 * dt * v1
        v_mid = eval_v(z_mid, t_mid)
        z = z + v_mid * dt
    z0_inv = z

    for step in range(num_steps):
        t_start = tau[step].expand(B)
        dt = (tau[step + 1] - tau[step]).item()
        v1 = eval_v(z, t_start)
        t_mid = (tau[step] + 0.5 * dt).expand(B)
        z_mid = z + 0.5 * dt * v1
        v_mid = eval_v(z_mid, t_mid)
        z = z + v_mid * dt
    z1_recon = z

    cycle_mse = F.mse_loss(z1_recon, z1_gt_norm).item()
    cycle_cos = F.cosine_similarity(z1_recon.flatten(0, 1), z1_gt_norm.flatten(0, 1), dim=-1).mean().item()

    # 2. Text Sensitivity Test (Delta z from Text Conditioning)
    torch.manual_seed(42)
    z_noise = torch.randn(B, T, H, device=device)
    z_cond = z_noise.clone()
    z_uncond = z_noise.clone()

    for step in range(num_steps):
        t_start = tau[step].expand(B)
        dt = (tau[step + 1] - tau[step]).item()
        # Conditional
        v1_c = eval_v(z_cond, t_start, use_text=True)
        z_mid_c = z_cond + 0.5 * dt * v1_c
        t_mid = (tau[step] + 0.5 * dt).expand(B)
        v_mid_c = eval_v(z_mid_c, t_mid, use_text=True)
        z_cond = z_cond + v_mid_c * dt
        # Unconditional
        v1_u = eval_v(z_uncond, t_start, use_text=False)
        z_mid_u = z_uncond + 0.5 * dt * v1_u
        v_mid_u = eval_v(z_mid_u, t_mid, use_text=False)
        z_uncond = z_uncond + v_mid_u * dt

    text_shift = (z_cond - z_uncond).norm(dim=-1).mean().item()
    text_cos = F.cosine_similarity(z_cond.flatten(0, 1), z_uncond.flatten(0, 1), dim=-1).mean().item()

    print('=' * 70)
    print('PROPER DIAGNOSTIC TEST SUITE: FLOW MATCHING VALIDATION')
    print('=' * 70)
    print(f'1. Lossless Flow Inversion Cycle Test:')
    print(f'   Cycle Reconstruction MSE:       {cycle_mse:.6f} (Target: < 0.05, Lower = Better)')
    print(f'   Cycle Cosine Similarity:        {cycle_cos:.4f} (Target: > 0.98, Higher = Better)')
    print(f'\n2. Text Responsiveness & Conditioning Sensitivity:')
    print(f'   Text Latent Shift (norm):       {text_shift:.4f} (Target: > 1.0, Proves Model is NOT Text-Blind)')
    print(f'   Cond vs Uncond Cosine Sim:      {text_cos:.4f} (Target: < 0.85, Shows Text Drives Trajectory)')
    print(f'\n3. Latent Distribution Preservation:')
    print(f'   Ground Truth Std:               {z1_gt_norm.std().item():.4f}')
    print(f'   Generated Sample Std:           {z_cond.std().item():.4f} (Calibrated via velocity_scale=1.12)')
    print('=' * 70)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
