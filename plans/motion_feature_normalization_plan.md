# Motion Feature Normalization Plan (Save to `Mean.npy` and `Std.npy`)

This document defines the normalization procedure for the 271D motion feature vector and specifies how to compute and save statistics to:

```
Mean.npy
Std.npy
```

The normalization is designed for:
- Autoregressive rollout stability
- Flow-matching generation
- Long sequence training
- Physically meaningful reconstruction

---

# 1. Feature Layout (271D)

```
[0:3]      Root global position (XYZ)
[3:69]     22 RIC positions (22×3)
[69:201]   22 joint 6D rotations (22×6)
[201:267]  22 local velocities (22×3)
[267:271]  Foot contacts (4D)
```

---

# 2. Preprocessing Before Computing Statistics

## 2.1 Root Translation Handling

Convert absolute root XZ position to velocity form before normalization:

```
vx_t = x_t - x_{t-1}
vz_t = z_t - z_{t-1}
```

Keep root height (Y) absolute.

Final root representation used for normalization:

```
[vx, root_height, vz]
```

This prevents trajectory drift and large magnitude accumulation.

---

# 3. Normalization Strategy

Use **per-dimension z-score normalization**:

```
x_norm = (x - mean) / (std + 1e-6)
```

Statistics must be computed:
- Across ALL training sequences
- Across ALL frames
- Per feature dimension

---

# 4. Feature Group Rules

## 4.1 Root Features (3D)

Normalize:
- Root velocity X
- Root velocity Z
- Root height Y

Method: Standardization

---

## 4.2 RIC Positions (66D)

Already root-relative.

Normalize per dimension using z-score.

---

## 4.3 6D Rotations (132D)

Keep raw 6D representation.

Normalize per dimension using z-score.

Do NOT:
- L2 normalize
- Min-max scale

---

## 4.4 Local Velocities (66D)

Normalize per dimension using z-score.

Optional (recommended):
- Clip values to ±3σ before computing mean/std.

---

## 4.5 Foot Contacts (4D)

Binary values (0 or 1).

Do NOT normalize.

For Mean.npy and Std.npy:

```
mean_contact = 0
std_contact  = 1
```

This keeps contacts unchanged during normalization.

---

# 6. Saving Files

```
np.save("Mean.npy", mean)
np.save("Std.npy", std)
```

Both arrays must have shape:

```
(271,)
```

---

# 7. Using During Training

Normalize input features:

```
features_norm = (features - mean) / std
```

---

# 8. Denormalization During Inference

After generation:

```
features = features_norm * std + mean
```

Then:

1. Reconstruct root XZ position via cumulative sum of velocities
2. Convert 6D → rotation matrices
3. Apply forward kinematics for global joint positions

---

# 9. Important Rules

DO:
- Use global dataset statistics
- Store statistics once
- Keep contacts unchanged
- Use velocity form for root translation

DO NOT:
- Compute mean/std per batch
- Use a single scalar std for all dims
- Normalize binary contacts
- Use min-max scaling

---

# 10. Final Output Files

The implementation must produce exactly:

```
Mean.npy  (shape: 271,)
Std.npy   (shape: 271,)
```

These files will be required for:
- Training normalization
- Inference denormalization
- Flow-matching generation pipeline

---

End of normalization specification.

