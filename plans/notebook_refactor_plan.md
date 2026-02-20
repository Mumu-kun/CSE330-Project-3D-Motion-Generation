# Notebook Refactoring Plan: Use src/utils Imports

## Objective
Refactor `misc/humanml3d-subset-generator.ipynb` to use imports from `src/utils/` instead of inline implementations, ensuring feature extraction parity with the training pipeline.

## Problem Summary

The notebook currently has **inline implementations** that differ from `src/utils/motion_utils.py`:

| Component | Notebook | motion_utils.py | MoMask |
|-----------|----------|-----------------|--------|
| RIC rotation | `qrot(qinv(root_quat), ric)` | `qrot(root_quat, ric)` | `qrot(r_rot, ric)` |
| Local velocity rotation | `qrot(qinv(root_quat), vel)` | `qrot(root_quat, vel)` | `qrot(r_rot, vel)` |
| `quaternion_to_cont6d` | Custom (potential bug) | Standard (columns 0,1) | Standard |

**Result**: The notebook produces **different feature values** than motion_utils.py, causing model incompatibility.

---

## Solution: Embedded Files Pattern (Like build/notebook.ipynb)

Following the pattern in `build/notebook.ipynb`, we will create a setup cell that embeds the necessary source files inline and writes them to disk. This makes the notebook self-contained for Kaggle/Colab environments.

### Required Files to Embed

Based on the imports needed for feature extraction:

1. **`utils/quaternion.py`** - Core quaternion operations
2. **`utils/motion_utils.py`** - Feature extraction functions (uses pure PyTorch IK, no Skeleton dependency)

### Setup Cell Structure

```python
# =========================================================
# CLOUD ENVIRONMENT SETUP (AUTO-GENERATED)
# =========================================================
import os
import sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IN_COLAB or IN_KAGGLE:
    print("Running in Cloud Environment")
    
    # Write supporting files
    FILES = {
        'utils/__init__.py': '# Utils module\n',
        'utils/quaternion.py': '''... [content from src/utils/quaternion.py] ...''',
        'utils/motion_utils.py': '''... [content from src/utils/motion_utils.py, with Skeleton import removed] ...''',
    }
    
    for filepath, content in FILES.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Created {filepath}')
    
    print("Setup Complete!")
else:
    print("Running locally. No setup needed.")
```

---

## Implementation Details

### Files to Embed (from src/utils/)

The following files need to be embedded in the notebook's setup cell:

1. **`utils/quaternion.py`** (17089 chars) - Core quaternion operations:
   - `qinv`, `qinv_np` - Quaternion inverse
   - `qmul`, `qmul_np` - Quaternion multiplication
   - `qrot`, `qrot_np` - Rotate vector by quaternion
   - `qbetween`, `qbetween_np` - Quaternion between two vectors
   - `quaternion_to_cont6d`, `quaternion_to_cont6d_np` - Convert to 6D representation
   - `cont6d_to_matrix`, `cont6d_to_matrix_np` - Convert 6D to rotation matrix

2. **`utils/motion_utils.py`** (21655 chars) - Feature extraction:
   - `T2M_RAW_OFFSETS`, `T2M_KINEMATIC_CHAIN` - Skeleton definitions
   - `DATASET_CONFIGS` - Dataset configuration
   - `preprocess_sequence()` - PyTorch-based 271D feature extraction (uses `_compute_ik` - pure PyTorch, no Skeleton class needed)
   - `features_to_positions()` - Reconstruction from features
   - `IncrementalFeatureExtractor` - Frame-by-frame extraction

**Note**: `motion_utils.py` imports `Skeleton` but doesn't actually use it - the `_compute_ik` function is a pure PyTorch implementation. We can remove the Skeleton import when embedding.

### Dependency Requirements

The embedded files require:
```
torch
numpy
```

**Note**: No `scipy` dependency needed since we're not using `skeleton.py`.

---

## Changes Required in Notebook

### 1. Add Setup Cell (After Configuration Cell)

Insert a new cell after line 26 that embeds the utility files:

```python
# =========================================================
# CLOUD ENVIRONMENT SETUP - EMBEDDED UTILITIES
# =========================================================
import os
import sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

# Always create utils files (for both cloud and local)
print("Setting up utility files...")

FILES = {
    'utils/__init__.py': '# Utils module\n',
    'utils/quaternion.py': '''...[content from src/utils/quaternion.py]...''',
    'utils/motion_utils.py': '''...[content from src/utils/motion_utils.py, with Skeleton import removed]...''',
}

for filepath, content in FILES.items():
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Created {filepath}')

# Add to Python path
sys.path.insert(0, str(Path.cwd()))
print("Setup Complete!")
```

### 2. Replace Inline Definitions (lines 138-588)

**DELETE** the following inline implementations:
- Lines 138-184: `T2M_RAW_OFFSETS` definition
- Lines 188-200: `T2M_KINEMATIC_CHAIN` definition
- Lines 204-208: `FACE_JOINT_INDX`, `FID_R`, `FID_L` definitions
- Lines 214-265: `qnormalize`, `qmul`, `qinv`, `qrot`, `_qbetween` functions
- Lines 268-332: `quaternion_to_cont6d` function
- Lines 336-434: `_compute_ik_np` function
- Lines 438-584: `extract_271d_features` function

**REPLACE** with imports:

```python
# --- FEATURE EXTRACTION (Using src/utils imports) ---
import torch
from utils.motion_utils import (
    T2M_RAW_OFFSETS,
    T2M_KINEMATIC_CHAIN,
    DATASET_CONFIGS,
    get_dataset_config,
    preprocess_sequence,  # 271D format - matches notebook dimension
)
from utils.quaternion import (
    qrot,
    qinv,
    qmul,
    quaternion_to_cont6d,
    quaternion_to_cont6d_np,
)

# Wrapper for 271D feature extraction (matching training pipeline)
def extract_271d_features(positions: np.ndarray, feet_thre: float = 0.002) -> np.ndarray:
    """
    Extract 271D features from joint positions.
    Uses the canonical implementation from motion_utils.py.
    
    Feature Layout:
        [0:3]   Root height Y, velocity X, velocity Z
        [3:69]  22 RIC positions (22 * 3)
        [69:201] 22 6D rotations (22 * 6)
        [201:267] 22 local velocities (22 * 3)
        [267:271] Foot contacts (4D)
    """
    positions_torch = torch.from_numpy(positions).float()
    features_torch = preprocess_sequence(positions_torch, feet_thre=feet_thre)
    return features_torch.numpy()

print("Feature extraction functions loaded from src/utils/")
```

### 3. Feature Extraction Call (line ~619)

No changes needed - the wrapper function maintains the same name `extract_271d_features()`.

---

## Summary of Changes

| Location | Change |
|----------|--------|
| After line 26 | Add setup cell with embedded utility files |
| Lines 138-588 | Delete inline implementations, replace with imports |
| Line ~619 | No change (wrapper function has same name) |
| Throughout | Feature layout remains 271D (same as before) |

**Key Fix**: The quaternion rotation direction is corrected by using `preprocess_sequence()` from motion_utils.py, which uses `qrot(root_quat, ric)` instead of `qrot(qinv(root_quat), ric)`.

### 4. Update Feature Layout Documentation

Both the notebook and `motion_utils.py` use **271D features** with the same layout:

**271D Feature Layout** (from motion_utils.py FEATURE_SLICES):
```
[0:3]   Root features: height Y, velocity X, velocity Z (3D)
[3:69]  22 RIC positions (66D = 22 * 3)
[69:201] 22 6D rotations (132D = 22 * 6)
[201:267] 22 local velocities (66D = 22 * 3)
[267:271] Foot contacts (4D)
```

**The only difference is the quaternion rotation direction**:
- Notebook: `qrot(qinv(root_quat), ric)` - **INCORRECT**
- motion_utils.py: `qrot(root_quat, ric)` - **CORRECT** (matches MoMask)

---

## Verification Steps

After refactoring:

1. **Run the notebook** on a small subset
2. **Compare features** with features generated by `motion_utils.preprocess_sequence()`
3. **Verify round-trip reconstruction** using `features_to_positions()`
4. **Check model compatibility** - features should work with models trained on motion_utils-processed data

---

## Impact

- **Training data**: Features will match the training pipeline
- **Model compatibility**: Models trained on motion_utils features will work correctly
- **Code maintenance**: Single source of truth for feature extraction logic
