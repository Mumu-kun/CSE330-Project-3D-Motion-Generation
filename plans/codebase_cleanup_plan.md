# Codebase Cleanup Plan - Remove Unused Code

## Overview

This plan details unused imports, functions, classes, and redundant code identified in the `src/` directory. The cleanup will streamline the codebase without affecting functionality.

---

## Completed Changes

### 1. src/config.py - Removed Unused Imports ✅
Removed `from typing import Optional, List, Tuple` - these were never used.

### 2. src/models.py - Removed Unused Imports ✅
- Removed `Callable, Any` from typing imports
- Removed `sequence_joints_to_features, get_dataset_config` from motion_utils imports

### 3. src/utils/train_utils.py - Removed Unused Import ✅
Removed `List` from typing imports.

### 4. src/utils/utils.py - Deleted ✅
Deleted the entire file - it was a re-export module that was never imported anywhere.

### 5. src/utils/quaternion.py - Removed Unused Functions ✅
Removed 18 unused functions, keeping only:
- `qinv` - used in motion_utils.py
- `qmul` - used in motion_utils.py
- `qrot` - used in motion_utils.py and pose_validation.py
- `quaternion_to_matrix` - used internally
- `quaternion_to_cont6d` - used in motion_utils.py
- `cont6d_to_matrix` - used in motion_utils.py
- `matrix_to_quaternion` - used internally
- `cont6d_to_quaternion` - used in motion_utils.py

### 6. src/utils/dataset.py - Removed Duplicate Import ✅
Removed duplicate `from typing import List, Dict, Any` statement.

### 7. src/utils/visualization.py - Removed Unused Parameter ✅
Removed `ground_truth` parameter from `visualize_motion()` function.

### 8. src/utils/pose_validation.py - Kept Import ⚠️
`T2M_RAW_OFFSETS` was initially identified as unused but is actually used in `_compute_expected_bone_lengths()` function. Kept the import.

---

## Summary of Changes

| File | Change Type | Items Affected | Status |
|------|-------------|----------------|--------|
| `config.py` | Remove import | `Optional, List, Tuple` from typing | ✅ |
| `models.py` | Remove imports | `Callable, Any` from typing; `sequence_joints_to_features, get_dataset_config` from motion_utils | ✅ |
| `train_utils.py` | Remove import | `List` from typing | ✅ |
| `utils.py` | Delete file | Entire file (unused re-export module) | ✅ |
| `quaternion.py` | Remove functions | 18 unused functions | ✅ |
| `dataset.py` | Remove duplicate | Duplicate `from typing import` | ✅ |
| `visualization.py` | Remove parameter | `ground_truth` parameter (unused) | ✅ |
| `pose_validation.py` | Keep import | `T2M_RAW_OFFSETS` (actually used) | ⚠️ Kept |

---

## Verification

All imports verified working:
- `config.Config`
- `models.MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator`
- `utils.motion_utils.features_to_positions, flow_output_to_271d, RootPositionTracker`
- `utils.quaternion.qinv, qmul, qrot, quaternion_to_cont6d, cont6d_to_matrix, cont6d_to_quaternion`
- `utils.train_utils.extract_prev_frame_features, extract_clean_target, EMAModel, train, validate, generate_free_running`

---

## Lines of Code Reduced

- `config.py`: -1 line
- `models.py`: -4 lines
- `train_utils.py`: -1 line
- `utils.py`: -49 lines (entire file deleted)
- `quaternion.py`: ~-350 lines (18 functions removed)
- `dataset.py`: -2 lines
- `visualization.py`: -2 lines

**Total: ~-409 lines of code removed**
