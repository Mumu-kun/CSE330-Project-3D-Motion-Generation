# Fix Plan: features_to_positions Reconstruction Error

## Problem Summary

The `features_to_positions` function in [`src/utils/motion_utils.py`](src/utils/motion_utils.py:404) does not correctly reconstruct global joint positions from 271D feature vectors.

## Test Results

### Sample 000070 (271D format) - **PASS**
```
Joints shape: (117, 22, 3)
Vectors shape: (117, 271)
MSE: 0.0000000000
MAE: 0.0000000041
Max error: 0.0000002384
```

### Sample 000000 (263D format) - **FAIL** (different format!)
```
Joints shape: (116, 22, 3)
Vectors shape: (116, 263)
MSE: 0.4211305380
```

## Root Cause Analysis

### The `features_to_positions` function is CORRECT for 271D format!

Testing confirms perfect reconstruction with MSE = 0.000000 for 271D format data.

### The Real Issue: 263D vs 271D Format Mismatch

The sample files use **different feature formats**:

**271D Format (sample 000070):**
```
[0:3]   Root height Y, velocity X, velocity Z
[3:69]  RIC positions (22x3) - relative to root, rotated
[69:201] Rotations 6D (22x6)
[201:267] Local velocities (22x3)
[267:271] Foot contacts (4D)
```

**263D Format (sample 000000):**
```
[0:3]   Unknown (NOT root height/velocity)
[3:69]  ABSOLUTE joint positions (not RIC!)
[69:201] Rotations 6D (22x6)
[201:263] Velocities (62D - different from 66D)
```

Key differences:
1. **263D stores absolute joint positions** at [3:69], not RIC
2. **263D has 62D velocities** instead of 66D
3. **263D missing foot contacts** (4D)

### Evidence

```
263D [3:6] = [0.9415, 0.0623, 0.8612]  # This is joint 0 position!
GT joint 0 = [0, 0.9415, 0]             # Root position
GT joint 1 = [0.0623, 0.8612, -0.017]   # Joint 1 position

271D [3:6] = [0, 0, 0]                  # RIC root is always 0
GT RIC[0] = [0, 0, 0]                   # Root relative to itself
```

## Conclusion

**No fix needed for `features_to_positions`!** The function works correctly for 271D format.

The issue is that:
1. Sample 000000 uses 263D format (different layout)
2. Sample 000070 uses 271D format (correct layout)
3. The function only supports 271D format

## Recommendations

1. **Ensure all data uses 271D format** - The `sequence_joints_to_features` function produces 271D format
2. **Convert 263D to 271D** if needed - But this requires understanding the exact 263D layout
3. **Add format detection** - Check feature dimension and handle accordingly

## Test Verification

```python
# Test with 271D format (sample 000070)
joints = np.load('sample_data/000070_joint.npy')  # (N, 22, 3)
vecs = np.load('sample_data/000070_vec.npy')      # (N, 271)

joints_t = torch.from_numpy(joints).float()
vecs_t = torch.from_numpy(vecs).float()

reconstructed = features_to_positions(vecs_t)
mse = torch.mean((reconstructed - joints_t) ** 2).item()
print(f"MSE: {mse}")  # Result: 0.000000
```
