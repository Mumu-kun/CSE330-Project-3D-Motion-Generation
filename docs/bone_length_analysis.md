# Bone Length Analysis: T2M_RAW_OFFSETS vs Sample Data

## Executive Summary

The analysis reveals a critical insight: **`T2M_RAW_OFFSETS` are UNIT VECTORS representing bone directions, NOT actual bone lengths.** The actual skeleton has bone lengths that are approximately 10-43% of the unit vectors defined in `T2M_RAW_OFFSETS`.

## Key Findings

### 1. T2M_RAW_OFFSETS are Unit Vectors

The [`T2M_RAW_OFFSETS`](../src/utils/motion_utils.py:39) tensor defines bone **directions** in T-pose, not lengths:

| Direction Vector | Meaning | Example Joints |
|-----------------|---------|----------------|
| `[1, 0, 0]` | +X (right) | R_Hip, R_Shoulder |
| `[-1, 0, 0]` | -X (left) | L_Hip, L_Shoulder |
| `[0, 1, 0]` | +Y (up) | Spine, Neck |
| `[0, -1, 0]` | -Y (down) | Arms, Legs |
| `[0, 0, 1]` | +Z (forward) | Feet, Head |

All non-root vectors have length = 1.0 (unit vectors).

### 2. Actual Skeleton Bone Lengths

From ground truth joint positions (`000070_joint.npy`), the actual bone lengths are:

| Joint | T2M Length | Actual Length | Scale Factor |
|-------|-----------|---------------|--------------|
| R_Hip | 1.0000 | 0.1031 | 0.1031 |
| L_Hip | 1.0000 | 0.1099 | 0.1099 |
| Spine1 | 1.0000 | 0.1316 | 0.1316 |
| R_UpLeg | 1.0000 | 0.3936 | 0.3936 |
| L_UpLeg | 1.0000 | 0.3902 | 0.3902 |
| Spine2 | 1.0000 | 0.1432 | 0.1432 |
| R_LoLeg | 1.0000 | 0.4324 | 0.4324 |
| L_LoLeg | 1.0000 | 0.4256 | 0.4256 |
| Spine3 | 1.0000 | 0.0574 | 0.0574 |
| R_Foot | 1.0000 | 0.1434 | 0.1434 |
| L_Foot | 1.0000 | 0.1494 | 0.1494 |
| Neck | 1.0000 | 0.2194 | 0.2194 |
| R_Shoulder | 1.0000 | 0.1375 | 0.1375 |
| L_Shoulder | 1.0000 | 0.1434 | 0.1434 |
| Head | 1.0000 | 0.1030 | 0.1030 |
| R_UpArm | 1.0000 | 0.1316 | 0.1316 |
| L_UpArm | 1.0000 | 0.1230 | 0.1230 |
| R_LoArm | 1.0000 | 0.2568 | 0.2568 |
| L_LoArm | 1.0000 | 0.2631 | 0.2631 |
| R_Hand | 1.0000 | 0.2660 | 0.2660 |
| L_Hand | 1.0000 | 0.2699 | 0.2699 |

**Key observation**: Ground truth bone lengths have **zero variance** across frames (GT_Std = 0.0000), confirming the skeleton has fixed proportions.

### 3. Reconstructed Bone Lengths Have High Variance

When converting normalized features back to positions using [`features_to_positions()`](../src/utils/motion_utils.py:404), the reconstructed bone lengths show:

- **Mean relative difference**: -41.04% from T2M unit lengths
- **High variance across frames**: std = 0.24 on average
- **All joints have high variance** (std > 0.01)

This indicates the reconstruction process does NOT preserve the constant bone length property of the skeleton.

## Why This Matters

### For Forward Kinematics (`_forward_kinematics`)

The FK function uses `T2M_RAW_OFFSETS` as bone vectors:

```python
positions[:, child_idx] = (
    torch.bmm(matR, offset_vec).squeeze(-1) + positions[:, parent_idx]
)
```

Since `offset_vec` comes from `T2M_RAW_OFFSETS` (unit vectors), the FK produces positions with **unit bone lengths**, not the actual skeleton proportions.

### For Inverse Kinematics (`_compute_ik`)

The IK function computes rotations by comparing:
- `u = offsets[:, child_idx]` - T-pose bone direction (unit vector)
- `v = positions[:, child_idx] - positions[:, parent_idx]` - actual bone direction

The rotation is computed to align `u` to `v`, but this only works correctly if `u` and `v` have the **same length**. When they don't, the IK produces rotations that, when fed back through FK, produce different bone lengths.

## Implications

1. **The current implementation uses unit vectors for FK**, which doesn't match the actual skeleton proportions
2. **IK→FK round-trip doesn't preserve bone lengths** because the offsets don't match actual bone lengths
3. **The RIC (Root-Invariant Coordinates) representation** stores actual joint positions, which naturally have correct bone lengths
4. **The `features_to_positions()` function** uses direct RIC transform, not FK, so it preserves bone lengths from the RIC features

## Recommendations

1. **For FK-based reconstruction**: Scale `T2M_RAW_OFFSETS` by actual bone lengths before use
2. **For current pipeline**: The direct RIC transform in `features_to_positions()` is correct and should be preferred
3. **For IK/FK consistency**: Either:
   - Use actual bone lengths in `T2M_RAW_OFFSETS`, or
   - Normalize input positions to unit bone lengths before IK

## Computed Actual Bone Offsets

If needed, here are the actual bone offsets scaled from T2M_RAW_OFFSETS:

```python
ACTUAL_OFFSETS = torch.tensor([
    [0.000000, 0.000000, 0.000000],  # Root
    [0.103074, 0.000000, 0.000000],  # R_Hip
    [-0.109883, 0.000000, 0.000000],  # L_Hip
    [0.000000, 0.131568, 0.000000],  # Spine1
    [0.000000, -0.393623, 0.000000],  # R_UpLeg
    [0.000000, -0.390188, 0.000000],  # L_UpLeg
    [0.000000, 0.143190, 0.000000],  # Spine2
    [0.000000, -0.432433, 0.000000],  # R_LoLeg
    [0.000000, -0.425643, 0.000000],  # L_LoLeg
    [0.000000, 0.057365, 0.000000],  # Spine3
    [0.000000, 0.000000, 0.143382],  # R_Foot
    [0.000000, 0.000000, 0.149419],  # L_Foot
    [0.000000, 0.219360, 0.000000],  # Neck
    [0.137487, 0.000000, 0.000000],  # R_Shoulder
    [-0.143383, 0.000000, 0.000000],  # L_Shoulder
    [0.000000, 0.000000, 0.103039],  # Head
    [0.000000, -0.131614, 0.000000],  # R_UpArm
    [0.000000, -0.122984, 0.000000],  # L_UpArm
    [0.000000, -0.256840, 0.000000],  # R_LoArm
    [0.000000, -0.263092, 0.000000],  # L_LoArm
    [0.000000, -0.266012, 0.000000],  # R_Hand
    [0.000000, -0.269876, 0.000000],  # L_Hand
], dtype=torch.float32)
```

## Analysis Script

See [`tests/analyze_bone_lengths.py`](../tests/analyze_bone_lengths.py) for the complete analysis code.
