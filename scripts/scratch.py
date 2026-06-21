import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


file = PROJECT_ROOT / "sample_data" / "Mean263.npy"
file1 = PROJECT_ROOT / "sample_data" / "Std263.npy"
file2 = PROJECT_ROOT / "sample_data" / "000001_263.npy"

mean: np.ndarray = np.load(file)
std: np.ndarray = np.load(file1)
data: np.ndarray = np.load(file2)

print(f"Mean shape: {mean.shape}, Std shape: {std.shape}, Data shape: {data.shape}")
data2 = data.copy()
for i, d in enumerate(data2):
    data2[i] = d * std + mean


# print(mean[4:67])

file3 = PROJECT_ROOT / "sample_data" / "000000_joint.npy"
joint_data: np.ndarray = np.load(file3)
print(f"Joint data shape: {joint_data.shape}")
print(joint_data.mean(axis=0)[0])
