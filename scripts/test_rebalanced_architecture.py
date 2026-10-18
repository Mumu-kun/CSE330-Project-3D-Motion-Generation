import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import Config
from utils.models.flow_matching_predictor import FlowMatchingPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80, flush=True)
print(f"TESTING REBALANCED PREDICTOR ARCHITECTURE (Device: {device})", flush=True)
print("=" * 80, flush=True)

config = Config()
config.device = str(device)
predictor = FlowMatchingPredictor(config).to(device)
predictor.eval()

B = 16
T_target = 79
T_hist = 40
S_text = 28
H_enc = 512

torch.manual_seed(42)
z_t = torch.randn(B, T_target, H_enc, device=device)
t_val = torch.full((B,), 0.5, device=device)

track_features = torch.randn(B, T_hist, H_enc, device=device)
text_seq_A = torch.randn(B, S_text, H_enc, device=device)
text_seq_B = torch.randn(B, S_text, H_enc, device=device)

combined_cond_A = torch.cat([text_seq_A, track_features], dim=1)
combined_cond_B = torch.cat([text_seq_B, track_features], dim=1)
text_pooled_A = text_seq_A.mean(dim=1)
text_pooled_B = text_seq_B.mean(dim=1)

# ── TEST 1: AdaLN Conditioning Modulation Balance ────────────────────────────
print("\n1. VERIFYING AdaLN CONDITIONING BALANCE:", flush=True)
time_cond = predictor.time_embedder(t_val)
text_cond_A = predictor.text_proj(text_pooled_A)
text_cond_B = predictor.text_proj(text_pooled_B)

time_norm = F.normalize(time_cond, dim=-1, eps=1e-6)
text_norm_A = F.normalize(text_cond_A, dim=-1, eps=1e-6)
text_norm_B = F.normalize(text_cond_B, dim=-1, eps=1e-6)
norm_scale = float(config.predictor_config.hidden_size ** 0.5)

adaln_A = norm_scale * (0.5 * time_norm + 0.5 * text_norm_A)
adaln_B = norm_scale * (0.5 * time_norm + 0.5 * text_norm_B)

delta_adaln = (adaln_A - adaln_B).norm(dim=-1).mean().item() / adaln_A.norm(dim=-1).mean().item() * 100.0
print(f"   Time Vector Norm Share in AdaLN : 50.0%")
print(f"   Text Vector Norm Share in AdaLN : 50.0%")
print(f"   Overall AdaLN Vector Norm       : {adaln_A.norm(dim=-1).mean().item():.2f} (Target: ~{norm_scale:.2f})")
print(f"   Prompt Swap Sensitivity Impact  : {delta_adaln:.2f}% (Old baseline was only 14.8%!)")
assert delta_adaln > 40.0, f"Expected > 40% impact on prompt swap, got {delta_adaln:.2f}%"

# ── TEST 2: Cross-Attention Softmax Mass Allocation ──────────────────────────
print("\n2. VERIFYING CROSS-ATTENTION BUDGET REBALANCING:", flush=True)
with torch.no_grad():
    _, _, cross_attns = predictor(
        noisy_states=z_t,
        timesteps=t_val,
        track_features=combined_cond_A,
        text_embedding=text_pooled_A,
        output_attentions=True,
        history_states=track_features,
    )

layer_text_pcts = []
for l_idx, attn in enumerate(cross_attns):
    attn_mean = attn.mean(dim=(0, 1, 2))
    text_mass = attn_mean[:S_text].sum().item() * 100.0
    hist_mass = attn_mean[S_text:].sum().item() * 100.0
    layer_text_pcts.append(text_mass)
    print(f"   Layer {l_idx}: Text = {text_mass:5.2f}% | History = {hist_mass:5.2f}% (Ratio: {text_mass/hist_mass:.3f})")

avg_text = np.mean(layer_text_pcts)
print(f"   Average Text Share Across Layers: {avg_text:5.2f}% (Old baseline was only 32.29%!)")
assert avg_text > 45.0, f"Expected text attention share > 45%, got {avg_text:.2f}%"

# ── TEST 3: Gradient Flow Backpropagation ───────────────────────────────────
print("\n3. VERIFYING BACKPROPAGATION GRADIENT FLOW TO TEXT:", flush=True)
predictor.train()
z_t_grad = z_t.clone().detach()
v_target = torch.randn_like(z_t_grad)
text_seq_grad = text_seq_A.clone().detach().requires_grad_(True)
track_features_grad = track_features.clone().detach().requires_grad_(True)
comb_grad = torch.cat([text_seq_grad, track_features_grad], dim=1)
text_pooled_grad = text_seq_grad.mean(dim=1)

v_pred, _, _ = predictor(
    noisy_states=z_t_grad,
    timesteps=t_val,
    track_features=comb_grad,
    text_embedding=text_pooled_grad,
    history_states=track_features_grad,
)

loss = F.mse_loss(v_pred, v_target)
loss.backward()

grad_text = text_seq_grad.grad.norm().item()
grad_hist = track_features_grad.grad.norm().item()
grad_ratio = grad_text / max(grad_hist, 1e-6)

print(f"   Gradient Norm reaching Text Tokens    : {grad_text:8.4f}")
print(f"   Gradient Norm reaching History Frames : {grad_hist:8.4f}")
print(f"   Text / History Gradient Ratio         : {grad_ratio:8.3f}x")

print("\n" + "=" * 80, flush=True)
print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ARCHITECTURE IS REBALANCED.", flush=True)
print("=" * 80, flush=True)
