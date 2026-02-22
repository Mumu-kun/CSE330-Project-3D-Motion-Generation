# Component-wise Error Analysis Enhancement Plan

## Objective

Modify `tests/test_flow_predictor_comparison.py` to provide detailed component-wise error analysis for:
1. **Root features (9D)**: height(1) + velocity(2) + rotation_6d(6)
2. **RIC positions (63D)**: 21 joints × 3D - local pose correctness
3. **Local velocities (66D)**: 22 joints × 3D - motion dynamics

## Current State

The file already has:
- `extract_root_features_from_72d()` - extracts root height, velocity, rotation
- `extract_ric_from_72d()` - extracts RIC positions
- `extract_local_velocities_from_271d()` - extracts local velocities
- `test_ric_and_velocity_analysis()` - basic component analysis

## Required Changes

### 1. Add New Test: `test_component_errors_detailed()`

Create a comprehensive test that:

```python
def test_component_errors_detailed():
    """
    Detailed component-wise error analysis.
    
    Tests three scenarios:
    1. Single frame prediction with GT history
    2. Teacher forcing rollout
    3. Autoregressive rollout
    
    For each, reports:
    - Root height MSE
    - Root velocity MSE (X, Z)
    - Root rotation 6D MSE
    - RIC positions MSE (per-joint and overall)
    - Local velocities MSE (per-joint and overall)
    """
```

### 2. Add Comparison Table Function

```python
def print_component_comparison_table(tf_errors, ar_errors):
    """
    Print side-by-side comparison of TF vs AR errors.
    
    Format:
    Frame | Root Height | Root Vel | Root Rot | RIC | Local Vel
          | TF    | AR   | TF  | AR | TF  | AR | TF  | AR | TF  | AR
    """
```

### 3. Enhance Existing `test_ric_and_velocity_analysis()`

Add:
- Per-joint RIC error breakdown
- Per-joint velocity error breakdown
- Clearer output formatting

## Feature Format Reference

### 72D Flow Output
| Index | Component | Dimension |
|-------|-----------|-----------|
| 0 | Root height Y | 1D |
| 1-2 | Root velocity X, Z | 2D |
| 3-8 | Root rotation 6D | 6D |
| 9-71 | Joint RIC positions | 63D (21×3) |

### 271D Features
| Index | Component | Dimension |
|-------|-----------|-----------|
| 0 | Root height Y | 1D |
| 1-2 | Root velocity X, Z | 2D |
| 3-68 | RIC positions (22 joints) | 66D (22×3) |
| 69-200 | 6D rotations (22 joints) | 132D (22×6) |
| 201-266 | Local velocities (22 joints) | 66D (22×3) |
| 267-270 | Foot contacts | 4D |

## Test Scenarios

### Scenario 1: Single Frame with GT History
- Use 1, 3, 5, 10, 20 seed frames
- Predict 1 frame ahead
- Report component errors

### Scenario 2: Teacher Forcing vs Autoregressive
- Use 5 seed frames
- Predict 10 frames
- Compare TF vs AR for each component

### Scenario 3: Long Horizon Rollout
- Use 5, 10 seed frames
- Predict 10, 20, 50 frames
- Track component error growth

## Output Format

```
================================================================
Component-wise Error Analysis
================================================================

1. Single Frame Prediction (GT History):
   Seeds | Root H | Root V | Root R | RIC    | Local V
   ------|--------|--------|--------|--------|--------
       1 | 0.0012 | 0.0234 | 0.0089 | 0.0032 | 0.0456
       5 | 0.0010 | 0.0198 | 0.0078 | 0.0023 | 0.0421

2. Teacher Forcing vs Autoregressive (5 seeds, 10 frames):
   Frame | Root Height    | Root Velocity   | RIC Positions   | Local Velocities
         | TF    | AR     | TF    | AR      | TF    | AR      | TF    | AR
   ------|-------|--------|-------|---------|-------|---------|-------|--------
       1 | 0.001 | 0.001  | 0.020 | 0.021   | 0.002 | 0.002   | 0.040 | 0.042
       2 | 0.001 | 0.015  | 0.019 | 0.145   | 0.002 | 0.089   | 0.038 | 0.234

3. Summary Statistics:
   Component        | TF Mean | AR Mean | AR/TF Ratio
   -----------------|---------|---------|------------
   Root Height      | 0.001   | 0.008   | 8.0x
   Root Velocity    | 0.020   | 0.234   | 11.7x
   Root Rotation    | 0.008   | 0.089   | 11.1x
   RIC Positions    | 0.002   | 0.156   | 78.0x
   Local Velocities | 0.039   | 0.345   | 8.8x
```

## Implementation Steps

1. Add helper function `compute_all_component_errors()` that computes all component errors at once
2. Add `test_component_errors_detailed()` with three scenarios
3. Update main block to run the new test
4. Keep existing tests for backward compatibility

## Files to Modify

- `tests/test_flow_predictor_comparison.py` - Add new test function and helper
