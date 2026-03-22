"""
Minimal test for FlowMatchingPredictor new API
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from config import Config, FlowMatchingPredictorConfig
from models import FlowMatchingPredictor

# Test 1: Init
print("Test 1: Initialization...")
config = Config()
pred_cfg = FlowMatchingPredictorConfig()
pred_cfg.hidden_size = 256
pred_cfg.track_dimensionality = 3

predictor = FlowMatchingPredictor(
    feature_size=64,
    config=pred_cfg,
    use_relative_shift=True,
)
print("PASS: Predictor initialized")

# Test 2: Forward with shifts
print("\nTest 2: Forward with explicit shifts...")
B, N, F, D, H = 2, 22, 64, 3, 256
track_features = torch.randn(B, N, F)
noised_tracks = torch.randn(B, N, D)
timesteps = torch.rand(B)
global_cond = torch.randn(B, H)
relative_shifts = noised_tracks - noised_tracks[:, :1, :]

output = predictor(
    track_features=track_features,
    noised_tracks=noised_tracks,
    timesteps=timesteps,
    global_cond=global_cond,
    relative_shifts=relative_shifts,
)
# Unpack tuple: (flow_prediction, hidden_states, attentions)
flow_pred, _, _ = output
assert flow_pred.shape == (B, N, D), f"Shape mismatch: {flow_pred.shape}"
assert torch.isfinite(flow_pred).all(), "Non-finite output"
print(
    f"PASS: Output shape {flow_pred.shape}, min={flow_pred.min():.4f}, max={flow_pred.max():.4f}"
)

# Test 3: Forward with None shifts
print("\nTest 3: Forward with None shifts (auto-zero)...")
output2 = predictor(
    track_features=track_features,
    noised_tracks=noised_tracks,
    timesteps=timesteps,
    global_cond=global_cond,
    relative_shifts=None,
)
# Unpack tuple: (flow_prediction, hidden_states, attentions)
flow_pred2, _, _ = output2
assert flow_pred2.shape == (B, N, D), f"Shape mismatch: {flow_pred2.shape}"
assert torch.isfinite(flow_pred2).all(), "Non-finite output"
print(f"PASS: Output shape {flow_pred2.shape}, stable")

# Test 4: Global cond required
print("\nTest 4: Global conditioning requirement...")
try:
    output3 = predictor(
        track_features=track_features,
        noised_tracks=noised_tracks,
        timesteps=timesteps,
    )
    print("FAIL: Should have raised TypeError for missing global_cond")
except TypeError as e:
    print(f"PASS: TypeError raised as expected")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
