"""Script to check RIC statistics from sample data."""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import features_to_positions

# Load sample data
vecs = np.load("sample_data/000070_vec.npy")
vecs_t = torch.from_numpy(vecs).float()

# Extract RIC positions (features 3:69)
ric = vecs_t[:, 3:69].reshape(-1, 22, 3)

print("RIC statistics per joint:")
print("=" * 60)
for j in range(22):
    min_vals = ric[:, j].min(dim=0).values.tolist()
    max_vals = ric[:, j].max(dim=0).values.tolist()
    mean_vals = ric[:, j].mean(dim=0).tolist()
    print(
        f'Joint {j:2d}: min={[f"{v:.3f}" for v in min_vals]}, max={[f"{v:.3f}" for v in max_vals]}, mean={[f"{v:.3f}" for v in mean_vals]}'
    )

print("\n" + "=" * 60)
print("Suggested RIC bounds (min_x, max_x, min_y, max_y, min_z, max_z):")
print("=" * 60)
for j in range(22):
    min_vals = ric[:, j].min(dim=0).values
    max_vals = ric[:, j].max(dim=0).values
    # Add 20% margin
    margin = 0.2
    range_vals = max_vals - min_vals
    min_bound = (min_vals - margin * range_vals).tolist()
    max_bound = (max_vals + margin * range_vals).tolist()
    print(
        f"{j:2d}: ({min_bound[0]:.3f}, {max_bound[0]:.3f}, {min_bound[1]:.3f}, {max_bound[1]:.3f}, {min_bound[2]:.3f}, {max_bound[2]:.3f}),"
    )
