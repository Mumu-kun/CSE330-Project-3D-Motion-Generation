import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from utils.config import Config
from utils.models.flow_matching_predictor import FlowMatchingPredictor, LatentDecoder
from utils.motion_utils import FeatureNormalizer, x68_to_positions, positions_to_x271
from utils.text_encoder import CLIPEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = Config()
config.device = device

# Load normalizer
mean = torch.from_numpy(np.load("src/dataset/humanml3d-subset/Mean.npy")).float()
std = torch.from_numpy(np.load("src/dataset/humanml3d-subset/Std.npy")).float()
normalizer = FeatureNormalizer(mean, std)

# Load predictor
pred_path = "src/checkpoints/phase3/predictor/phase3_predictor_best_val_20260901_005718.pt"
pred_ckpt = torch.load(pred_path, map_location=device, weights_only=False)
predictor = FlowMatchingPredictor(config).to(device)
if "predictor" in pred_ckpt:
    predictor.load_state_dict(pred_ckpt["predictor"])
elif "predictor_ema" in pred_ckpt:
    predictor.load_state_dict(pred_ckpt["predictor_ema"])
predictor.eval()

# Load decoder
dec_path = "src/checkpoints/phase3/decoder/phase3_decoder_best_val_20260730_221759.pt"
dec_ckpt = torch.load(dec_path, map_location=device, weights_only=False)
decoder = LatentDecoder(config).to(device)
if "decoder" in dec_ckpt:
    decoder.load_state_dict(dec_ckpt["decoder"])
elif "decoder_ema" in dec_ckpt:
    decoder.load_state_dict(dec_ckpt["decoder_ema"])
decoder.eval()

clip_encoder = CLIPEncoder().to(device)

# Latent stats
latent_mean = pred_ckpt["latent_stats"]["mean"].to(device) if "latent_stats" in pred_ckpt else None
latent_std = pred_ckpt["latent_stats"]["std"].to(device) if "latent_stats" in pred_ckpt else None
print("latent_mean is loaded:", latent_mean is not None)

prompts = [
    "a person throws a ball with the left hand",
    "a person throws a ball with the right hand",
    "a person walks forward",
    "a person kicks with their right leg",
]

for prompt in prompts:
    print(f"\n=================== PROMPT: '{prompt}' ===================")
    text_seq = clip_encoder.encode_sequence([prompt]).to(device) # (1, S, 512)
    text_pooled = text_seq.mean(dim=1) # (1, 512)
    S_text = text_seq.shape[1]
    
    # Track features: all zeros for scratch
    T_hist = config.history_length
    track_features = torch.zeros(1, T_hist, 512, device=device)
    combined_cond = torch.cat([text_seq, track_features], dim=1)
    
    # Mask
    text_mask = torch.ones((1, S_text), dtype=torch.bool, device=device)
    hist_mask = torch.zeros((1, T_hist), dtype=torch.bool, device=device) # all false
    cond_mask = torch.cat([text_mask, hist_mask], dim=1)
    key_padding_mask = cond_mask.unsqueeze(1).unsqueeze(2)
    
    # Integration
    num_steps = 20
    s = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    z_t = torch.randn(1, 79, 512, device=device)
    
    for step in range(num_steps):
        t_start = s[step].expand(1)
        dt = (s[step+1] - s[step]).item()
        
        v_c, _, _ = predictor(
            noisy_states=z_t,
            timesteps=t_start,
            track_features=combined_cond,
            text_embedding=text_pooled,
            key_padding_mask=key_padding_mask,
        )
        z_t = z_t + v_c * dt
        
    z1_norm = z_t
    if latent_mean is not None:
        z1 = z1_norm * latent_std + latent_mean
    else:
        z1 = z1_norm
        
    decoded_72 = decoder(z1) # (1, 79, 72)
    
    # Check joint RIC ranges:
    # 72D: [root_y(1), vx(1), vz(1), sin(1), cos(1), 21 joints * 3 (63), contacts(4)]
    # In 21 non-root joints:
    # Joint 13 = left shoulder, 16 = left elbow, 18 = left wrist, 20 = left hand
    # Note in 21 non-root: joint i is index i-1.
    # So left shoulder (13) -> non-root 12 -> slice [5 + 12*3 : 5 + 13*3] = [41:44]
    # left elbow (16) -> non-root 15 -> slice [5 + 15*3 : 5 + 16*3] = [50:53]
    # left wrist (18) -> non-root 17 -> slice [5 + 17*3 : 5 + 18*3] = [56:59]
    # left hand (20) -> non-root 19 -> slice [5 + 19*3 : 5 + 20*3] = [62:65]
    # right wrist (19) -> non-root 18 -> slice [5 + 18*3 : 5 + 19*3] = [59:62]
    
    lwrist_ric = decoded_72[0, :, 56:59].detach().cpu().numpy()
    rwrist_ric = decoded_72[0, :, 59:62].detach().cpu().numpy()
    
    l_range = np.ptp(lwrist_ric, axis=0) # peak to peak
    r_range = np.ptp(rwrist_ric, axis=0)
    
    print(f"Left Wrist RIC range  (X, Y, Z): {l_range.round(3)} | Max movement: {np.linalg.norm(l_range):.3f}")
    print(f"Right Wrist RIC range (X, Y, Z): {r_range.round(3)} | Max movement: {np.linalg.norm(r_range):.3f}")
    print(f"Root velocity (vx, vz) range   : {np.ptp(decoded_72[0, :, 1:3].detach().cpu().numpy(), axis=0).round(3)}")
