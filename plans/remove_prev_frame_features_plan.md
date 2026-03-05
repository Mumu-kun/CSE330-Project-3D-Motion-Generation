# Plan: Remove prev_frame_features from FlowMatchingPredictor

## Overview
This document outlines the changes needed to remove the `prev_frame_features` input from the `FlowMatchingPredictor` model design, including updates to all usage sites across the codebase.

---

## Current State Analysis

### FlowMatchingPredictor Current Design
The `FlowMatchingPredictor` currently accepts:
- `history_features`: (B, 22, per_joint_dim) - Context from MotionHistoryEncoder
- `noisy_target`: (B, 72) - Current noisy state x_t
- `prev_frame_features`: (B, 261) - Previous frame features (being removed)
- `noise_level`: (B,) - Flow time t

### Files Affected
1. `src/models.py` - FlowMatchingPredictor class and HumanMotionGenerator
2. `src/utils/train_utils.py` - Training and validation loops
3. `src/utils/motion_utils.py` - Helper functions for normalization
4. `tests/test_training_loop.py` - Unit tests

---

## Detailed Implementation Steps

### Step 1: Update FlowMatchingPredictor.__init__ (src/models.py)

**Remove these components:**
```python
# Lines 623-627 - Prev frame projections
self.input_proj_prev_root = nn.Linear(9, model_dim)  # root: height + vel + rot
self.input_proj_prev_joint = nn.Linear(12, model_dim)  # joints: RIC + rot + velocity

# Lines 661-663 - Null tokens for prev frame
self.null_prev_root = nn.Parameter(torch.zeros(1, 9))
self.null_prev_joint = nn.Parameter(torch.zeros(1, joint_count - 1, 12))
```

**Update docstring** to remove `prev_frame_features` from Input section.

---

### Step 2: Update FlowMatchingPredictor.forward (src/models.py)

**Current signature (lines 679-687):**
```python
def forward(
    self,
    history_features: torch.Tensor,  # (B, 22, per_joint_dim)
    noise_level: torch.Tensor,  # (B,)
    noisy_target: Optional[torch.Tensor] = None,  # (B, 72)
    prev_frame_features: Optional[torch.Tensor] = None,  # (B, 261) - REMOVE
    temporal_progress: Optional[torch.Tensor] = None,
    normalize: bool = True,  # REMOVE (only used for prev_frame_features)
):
```

**New signature:**
```python
def forward(
    self,
    history_features: torch.Tensor,  # (B, 22, per_joint_dim)
    noise_level: torch.Tensor,  # (B,)
    noisy_target: Optional[torch.Tensor] = None,  # (B, 72)
    temporal_progress: Optional[torch.Tensor] = None,
):
```

**Remove processing logic (lines 706-736):**
```python
# REMOVE: Normalization block (lines 706-712)
if self.normalizer is not None and normalize:
    if prev_frame_features is not None:
        prev_frame_features = self.normalizer.normalize_prev_frame_features(...)

# REMOVE: Null token handling (lines 714-722)
if prev_frame_features is None:
    prev_root = self.null_prev_root.expand(B, 9)
    prev_joints = self.null_prev_joint.expand(B, J - 1, 12)
else:
    prev_root = prev_frame_features[:, :9]
    prev_joints = prev_frame_features[:, 9:].reshape(B, J - 1, 12)

# REMOVE: Projection of prev frame (lines 727-733)
prev_root_proj = self.input_proj_prev_root(prev_root).unsqueeze(1)
prev_joint_proj = self.input_proj_prev_joint(prev_joints)
prev_proj = torch.cat([prev_root_proj, prev_joint_proj], dim=1)

# UPDATE: Change line 736 from:
cond_proj = history_proj + prev_proj
# To:
cond_proj = history_proj
```

---

### Step 3: Update HumanMotionGenerator.generate_sequence (src/models.py)

**Remove import (line 824):**
```python
from utils.train_utils import extract_prev_frame_features  # REMOVE
```

**Remove extraction (lines 897-900):**
```python
# REMOVE: Extract prev_frame_features
prev_frame_features = extract_prev_frame_features(last_frame)
```

**Update predictor calls (lines 915-926):**
Remove `prev_frame_features=prev_frame_features` and `normalize=True` from both CFG calls:
```python
# Conditional prediction
pred_cond = self.predictor(
    history_features=context_cond,
    noise_level=t,
    noisy_target=x_t,
    # REMOVE: prev_frame_features=prev_frame_features,
    # REMOVE: normalize=True,
)

# Unconditional prediction  
pred_uncond = self.predictor(
    history_features=context_uncond,
    noise_level=t,
    noisy_target=x_t,
    # REMOVE: prev_frame_features=prev_frame_features,
    # REMOVE: normalize=True,
)
```

---

### Step 4: Update train_utils.py

#### 4a. Option A: Remove extract_prev_frame_features entirely
If this function is ONLY used for prev_frame_features:

**Remove function (lines 40-79):**
```python
def extract_prev_frame_features(frame: torch.Tensor) -> torch.Tensor:
    """Extract 261D prev_frame_features from 271D frame."""
    # ... entire function
```

#### 4b. Update training loop (around lines 390-420)
**Remove:**
```python
prev_features = extract_prev_frame_features(prev_flat)  # (B*N, 261)
```

**Update predictor call (lines 412-413):**
```python
pred = predictor(
    history_features=contexts_flat,
    noise_level=t,
    noisy_target=x_t,
    # REMOVE: prev_frame_features=prev_features,
    # REMOVE: normalize=False,
)
```

#### 4c. Update validation loop (around lines 700-730)
**Remove:**
```python
prev_features = extract_prev_frame_features(hist[:, -1])
```

**Update predictor call (lines 722-724):**
```python
pred = predictor(
    history_features=context,
    noise_level=t,
    noisy_target=x_t,
    # REMOVE: prev_frame_features=prev_features,
)
```

---

### Step 5: Update motion_utils.py (Optional)

If `normalize_prev_frame_features` and `denormalize_prev_frame_features` are ONLY used for prev_frame_features:

**Remove methods from FeatureNormalizer class (lines 446-495):**
```python
def normalize_prev_frame_features(self, features: torch.Tensor) -> torch.Tensor:
    """Normalize 261D prev_frame_features..."""
    # ... method body

def denormalize_prev_frame_features(self, features: torch.Tensor) -> torch.Tensor:
    """Denormalize 261D prev_frame_features..."""
    # ... method body
```

---

### Step 6: Update tests/test_training_loop.py

#### 6a. Remove test for extract_prev_frame_features
**Remove function `test_extract_prev_frame_features` (lines 75-129)**

**Remove test call (lines 542-546):**
```python
try:
    results["extract_prev_frame_features"] = test_extract_prev_frame_features()
except Exception as e:
    results["extract_prev_frame_features"] = False
    print(f"[FAIL] extract_prev_frame_features: {e}\n")
```

#### 6b. Update test_predictor_forward (lines 280-318)
**Remove:**
```python
prev_frame_features = torch.randn(B, 261)  # 261D prev frame
```

**Update predictor call (lines 296-303):**
```python
output = predictor(
    history_features=history_features,
    noise_level=noise_level,
    noisy_target=noisy_target,
    # REMOVE: prev_frame_features=prev_frame_features,
    # REMOVE: normalize=True,
)
```

**Remove print statement:**
```python
print(f"Prev frame features shape: {prev_frame_features.shape}")
```

#### 6c. Update test_training_iteration (lines 326-419)
**Remove:**
```python
prev_features = extract_prev_frame_features(prev_flat)
```

**Update predictor call (lines 395-401):**
```python
pred = predictor(
    history_features=contexts_flat,
    noise_level=t,
    noisy_target=x_t,
    # REMOVE: prev_frame_features=prev_features,
    # REMOVE: normalize=False,
)
```

---

## Verification Checklist

After implementation, verify:
- [ ] `FlowMatchingPredictor` initializes without prev frame projection layers
- [ ] `FlowMatchingPredictor.forward()` accepts only required parameters
- [ ] Training loop runs without errors
- [ ] Validation loop runs without errors
- [ ] `HumanMotionGenerator.generate_sequence()` produces valid output
- [ ] All tests pass (or are appropriately updated/removed)

---

## Notes

1. The model architecture will be simplified to rely solely on `history_features` from the MotionHistoryEncoder for context
2. The noisy target is still projected and combined with history features
3. Time embedding and kinematic encoding remain unchanged
4. Output heads (root_head and joint_head) remain unchanged
