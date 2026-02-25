# Plan: Recompute All Rotations in `flow_output_to_271d`

## Summary

Modify [`flow_output_to_271d`](src/utils/motion_utils.py:773) to compute rotations for **all 22 joints** using inverse kinematics, instead of only updating the root rotation and cloning the rest from the previous frame.

## Current Implementation Problem

The current code at lines 828-831:

```python
# 4. Rotations (update root only)
prev_rot_6d = prev_frame[:, 69:201].reshape(B, 22, 6)
rotations_6d = prev_rot_6d.clone()
rotations_6d[:, 0] = root_rot_6d
```

This approach:
- Only updates the root rotation (joint 0)
- Copies all other joint rotations from the previous frame
- **Causes rotation drift** since joint rotations are not recomputed from actual positions

## Proposed Solution

After computing `new_positions` (line 858), use the existing [`_compute_ik`](src/utils/motion_utils.py:121) function to compute quaternions for all joints, then convert to 6D rotations.

## Implementation Steps

### Step 1: Extract skeleton configuration

The config is already retrieved at line 799, but we need additional fields:

```python
config = get_dataset_config(dataset_type)
fid_r = config["fid_r"]
fid_l = config["fid_l"]
# ADD: Extract skeleton data for IK
raw_offsets = config["raw_offsets"]  # torch.Tensor (22, 3)
kinematic_chain = config["kinematic_chain"]
face_joint_indx = config["face_joint_indx"]
```

### Step 2: Prepare raw_offsets tensor

The `raw_offsets` from config is a torch tensor but needs to be on the correct device and dtype:

```python
raw_offsets = raw_offsets.to(device=device, dtype=dtype)
```

**Note:** `T2M_RAW_OFFSETS` is already a torch tensor (defined at line 39), so no numpy conversion needed.

### Step 3: Compute quaternions using IK

After `new_positions` is computed (line 858), add:

```python
# Compute quaternions for all joints via IK
quaternions = _compute_ik(
    new_positions, raw_offsets, kinematic_chain, face_joint_indx
)  # returns (B, 22, 4)
```

### Step 4: Convert quaternions to 6D rotations

```python
rotations_6d = quaternion_to_cont6d(quaternions)  # (B, 22, 6)
```

### Step 5: Remove old rotation handling

**DELETE lines 828-831:**
```python
# 4. Rotations (update root only)
prev_rot_6d = prev_frame[:, 69:201].reshape(B, 22, 6)
rotations_6d = prev_rot_6d.clone()
rotations_6d[:, 0] = root_rot_6d
```

### Step 6: Verify imports

The required functions are already imported at the top of the file:
- [`_compute_ik`](src/utils/motion_utils.py:121) - defined in the same file
- [`quaternion_to_cont6d`](src/utils/quaternion.py:97) - imported from `utils.quaternion`
- [`get_dataset_config`](src/utils/motion_utils.py:94) - defined in the same file

## Code Changes Summary

| Location | Action |
|----------|--------|
| Lines 799-801 | Add extraction of `raw_offsets`, `kinematic_chain`, `face_joint_indx` |
| After line 801 | Add `raw_offsets = raw_offsets.to(device=device, dtype=dtype)` |
| Lines 828-831 | **DELETE** old rotation handling |
| After line 858 | Add IK computation and 6D conversion |

## Final Code Structure

```python
def flow_output_to_271d(...):
    # ... existing setup ...
    
    # -------------------------------------------------
    # Dataset config (for foot indices AND skeleton)
    # -------------------------------------------------
    config = get_dataset_config(dataset_type)
    fid_r = config["fid_r"]
    fid_l = config["fid_l"]
    raw_offsets = config["raw_offsets"].to(device=device, dtype=dtype)
    kinematic_chain = config["kinematic_chain"]
    face_joint_indx = config["face_joint_indx"]
    
    # ... existing code for root position, RIC, etc ...
    
    # -------------------------------------------------
    # 6. Reconstruct new positions
    # -------------------------------------------------
    # ... existing position reconstruction ...
    new_positions = torch.cat(
        [new_root_pos.unsqueeze(1), global_joints], dim=1
    )  # (B,22,3)
    
    # -------------------------------------------------
    # 7. Compute rotations via IK
    # -------------------------------------------------
    quaternions = _compute_ik(
        new_positions, raw_offsets, kinematic_chain, face_joint_indx
    )  # (B, 22, 4)
    rotations_6d = quaternion_to_cont6d(quaternions)  # (B, 22, 6)
    
    # -------------------------------------------------
    # 8. Local velocities (root-local)
    # -------------------------------------------------
    # ... existing velocity computation ...
    
    # ... rest of function unchanged ...
```

## Verification

After implementation:

1. **Unit test**: Create a test that verifies rotation consistency
   - Generate positions from known features
   - Recompute rotations via IK
   - Verify rotations match expected values

2. **Round-trip test**: Verify `features_to_positions` round-trip
   - Generate motion sequence
   - Extract features via `flow_output_to_271d`
   - Reconstruct positions via `features_to_positions`
   - Compare with original positions

## Benefits

1. **Consistency**: Rotations are derived from actual joint positions
2. **No drift**: Each frame's rotations are computed independently
3. **Correctness**: The 271D features will properly represent the generated motion
4. **Round-trip**: Features will correctly round-trip through `features_to_positions`
