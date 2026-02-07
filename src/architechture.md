# Human Motion Generation Architecture

This document describes the dual-stage generation pipeline: **Motion Context Encoding** and **Spatial-Temporal Flow Matching**.

---

## 1. Motion History Encoder (Contextual Intent)

The goal of this module is to map a sequence of past motion and a text description into a per-joint conditioning vector.

### Input
- **History Features**: `(B, T_hist, 263)` (HumanML3D Layout)
- **Text Conditioning**: CLIP Embeddings `(B, 512)`
- **Total Duration (Optional)**: Normalized frame count `(B, 1)`

### Process (Dual-MLP Design)
1. **Per-Joint Projection (Local Context)**:
    - Input: `RIC(3) + LocalVel(3) + Global(8) + Duration(1) + Text(32)` = 47D input per joint.
    - Architecture: `MLP(47 -> 128 -> 64)`.
    - Result: `(B, T_hist, 22, 64)`.
2. **Temporal Core**:
    - Operation: `Flatten(22 * 64) -> Bi-GRU(128)`.
    - Result: Summarizes history into a global context vector.
3. **Context Decoding**:
    - Architecture: `MLP(Summarized -> 22 * 64)`.
    - Result: `(B, 22, 64)` context latent.

---

## 2. Flow Matching Predictor (Spatial Generation)

The "engine" of the model. It predicts the velocity field required to move noisy joint positions toward the data manifold.

### Input (ARFM Conditioning)
Total 49D feature vector per joint:
- **History Context**: `(B, 22, 32 / 64)` derived from Encoder.
- **Previous Spatial State**: `Pos(3) + Rot(6) + Diffs(3)` = 12D.
- **Noisy Sample ($x_t$)**: Current noisy coordinates `(B, 22, 3)`.
- **Temporal Progress**: `(B, 22, 1)` (Normalized countdown to clip end).
- **Flow Time ($t$)**: `(B, 22, 1)` in range [0, 1].

### Internal Components
1. **Kinematic Hierarchy Encoder**:
    - Maps joints to one of 5 chains (e.g., Spine, Left Leg) and 8 depths.
    - Adds a structural bias so the transformer knows which joints are "hand" vs "foot".
2. **Spatial Transformer**:
    - Operation: Attention over all 22 joints using a 4-layer Transformer Encoder.
3. **MLP Noise Head**:
    - Architecture: `MLP(128 -> 128 -> 3)`.
    - Output: Predicted velocity vector $v_t$ (or noise $\epsilon$).

---

## 3. Data Layout (HumanML3D 263D)
| Index Range | Feature Type | Description |
| :--- | :--- | :--- |
| 0:4 | Root Global | Rotational velocity, Linear velocity (X, Z) |
| 4:67 | RIC Positions | Relative positions of 21 joints to root |
| 67:193 | RIC Quat/Rot | Rotation matrices/6D/Quaternion for joints |
| 193:259 | Local Vel | Local joint velocities |
| 259:263 | Foot Contact | Binary contact flags for 4 foot points |