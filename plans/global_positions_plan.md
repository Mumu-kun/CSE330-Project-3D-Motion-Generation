# Plan: Change FlowMatchingPredictor to Generate Global Positions

## Summary

Change the FlowMatchingPredictor from predicting **displacements** (velocities) to predicting **global joint positions** directly.

## Current Implementation

### 1. `build_prev_and_clean_diffs` ([`train_utils.py:12-61`](src/utils/train_utils.py:12))

Currently extracts:
- `prev_pos`: (B, 22, 3) - RIC positions from last history frame
- `prev_rot6d`: (B, 22, 6) - RIC rotations from last history frame
- `prev_v`: (B, 22, 3) - velocities from last history frame
- `clean_v`: (B, 22, 3) - **velocities from target frame** (displacement target)

The target `clean_v` uses velocity features at indices `[193:259]`.

### 2. Training Loop ([`train_utils.py:64-367`](src/utils/train_utils.py:64))

```python
# Current flow matching with displacements
eps = torch.randn_like(clean_diffs)  # clean_diffs = velocities
t = torch.rand(B, device=device)
t_b = t.view(B, 1, 1)

# Interpolate: x_t = t * clean + (1-t) * noise
noisy_target_diffs = t_b * clean_diffs + (1.0 - t_b) * eps

# Predict velocity field
pred_eps = flow_predictor(...)
flow_target = clean_diffs - eps
loss = F.mse_loss(pred_eps, flow_target)
```

### 3. `generate_sequence` ([`models.py:460-610`](src/models.py:460))

Currently:
1. Generates displacement `x_t` via flow matching
2. Updates positions: `new_joints_global = current_joints_global + x_t`
3. Uses `IncrementalFeatureExtractor` to convert back to 263D features

---

## Proposed Changes

### 1. Rename and Modify `build_prev_and_clean_diffs` → `build_prev_and_clean_positions`

**File:** [`src/utils/train_utils.py`](src/utils/train_utils.py)

**Changes:**
- Rename function to `build_prev_and_clean_positions`
- Instead of extracting velocities as target, compute global joint positions
- Use [`feature_to_joints()`](src/utils/motion_utils.py:354) or [`recover_from_ric()`](src/utils/motion_utils.py:283) to convert target frame to global positions

**New signature:**
```python
def build_prev_and_clean_positions(
    hist: torch.Tensor, 
    future: torch.Tensor, 
    joint_count: int = 22,
    dataset_type: str = "t2m"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts spatial features for the previous frame and the target global positions.
    
    Returns:
        prev_pos:   (B, 22, 3) - RIC positions from last history frame
        prev_rot6d: (B, 22, 6) - RIC rotations from last history frame  
        prev_v:     (B, 22, 3) - velocities from last history frame
        clean_pos:  (B, 22, 3) - Global joint positions of target frame
    """
```

**Implementation:**
```python
from utils.motion_utils import feature_to_joints

def build_prev_and_clean_positions(
    hist: torch.Tensor, 
    future: torch.Tensor, 
    joint_count: int = 22,
    dataset_type: str = "t2m"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_hist, _ = hist.shape
    last_frame = hist[:, -1]
    target_frame = future[:, 0]
    
    # 1. RIC Position extraction (unchanged)
    def get_pos(frame):
        ric_21 = frame[:, 4:67].reshape(B, 21, 3)
        root_pos = torch.zeros((B, 1, 3), device=frame.device, dtype=frame.dtype)
        return torch.cat([root_pos, ric_21], dim=1)
    
    prev_pos = get_pos(last_frame)
    
    # 2. RIC Rotation extraction (unchanged)
    def get_rot(frame):
        rot_21 = frame[:, 67:193].reshape(B, 21, 6)
        root_rot = torch.zeros((B, 1, 6), device=frame.device, dtype=frame.dtype)
        root_rot[:, 0, 0] = 1.0
        root_rot[:, 0, 4] = 1.0
        return torch.cat([root_rot, rot_21], dim=1)
    
    prev_rot6d = get_rot(last_frame)
    
    # 3. Velocity extraction (unchanged)
    prev_v = last_frame[:, 193:259].contiguous().view(B, 22, 3)
    
    # 4. NEW: Global positions as target (instead of velocities)
    # Convert target frame features to global joint positions
    clean_pos = feature_to_joints(target_frame, dataset_type=dataset_type)  # (B, 22, 3)
    
    return prev_pos, prev_rot6d, prev_v, clean_pos
```

### 2. Update Training Loop

**File:** [`src/utils/train_utils.py`](src/utils/train_utils.py)

**Changes:**
- Replace `build_prev_and_clean_diffs` with `build_prev_and_clean_positions`
- Rename `clean_diffs` to `clean_pos` (or `clean_positions`)
- Update zero-shot branch to use global positions

**Key changes:**
```python
# Standard window sampling branch
prev_pos, prev_rot6d, prev_v, clean_pos = build_prev_and_clean_positions(
    hist_actual_slice, future, dataset_type="t2m"
)
prev_features = torch.cat([prev_pos, prev_rot6d, prev_v], dim=-1)

# Zero-shot branch - need to convert first frame to global positions
if is_zero_shot:
    # ...
    clean_pos = feature_to_joints(future[:, 0], dataset_type="t2m")  # (B, 22, 3)

# Flow matching with global positions
eps = torch.randn_like(clean_pos)  # Now positions, not velocities
t = torch.rand(B, device=device)
t_b = t.view(B, 1, 1)

# Interpolate between noise and clean positions
noisy_target = t_b * clean_pos + (1.0 - t_b) * eps

# Predict velocity field (direction from noise to clean)
pred_v = flow_predictor(
    history_features=history_context,
    noise_level=t,
    noisy_target=noisy_target,  # Rename parameter
    prev_frame_features=prev_features,
)

flow_target = clean_pos - eps
loss = F.mse_loss(pred_v, flow_target)
```

### 3. Update `generate_sequence`

**File:** [`src/models.py`](src/models.py)

**Changes:**
- The output `x_t` from flow matching is now **global positions directly**
- Remove the addition: `new_joints_global = current_joints_global + x_t`
- Use predicted positions directly: `new_joints_global = x_t`

**Key changes:**
```python
# Flow matching loop - output is now positions, not displacement
for step in range(num_steps):
    t = torch.full((B,), step * dt, device=device)
    
    v_cond = self.predictor(
        history_features=context_cond,
        noise_level=t,
        noisy_target=x_t,  # Current noisy position estimate
        prev_frame_features=prev_frame_features,
        temporal_progress=t_prog,
    )
    
    v_uncond = self.predictor(
        history_features=context_uncond,
        noise_level=t,
        noisy_target=x_t,
        prev_frame_features=prev_frame_features,
        temporal_progress=t_prog,
    )
    
    v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
    x_t = x_t + v_t * dt  # x_t converges to clean positions

# NEW: x_t is now the predicted global positions directly
new_joints_global = x_t  # No longer: current_joints_global + x_t
current_joints_global = new_joints_global.clone()
```

### 4. Update FlowMatchingPredictor Parameter Name (Optional)

**File:** [`src/models.py`](src/models.py)

Consider renaming `noisy_target_diffs` to `noisy_target` since it's no longer specifically displacements:

```python
def forward(
    self,
    history_features: torch.Tensor,
    noise_level: torch.Tensor,
    noisy_target: Optional[torch.Tensor] = None,  # Renamed from noisy_target_diffs
    prev_frame_features: Optional[torch.Tensor] = None,
    temporal_progress: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```

---

## Implications

### Advantages
1. **Direct position prediction**: Model predicts absolute positions, which may be easier to learn
2. **Simpler inference**: No need to track and accumulate displacements
3. **Better global consistency**: Positions are predicted in global coordinate frame

### Considerations
1. **Larger output range**: Global positions have larger magnitude than displacements
   - May need to adjust learning rate or normalization
2. **Root motion**: Global positions include root translation, which was previously handled separately
3. **Scale**: Position values may be larger than velocity values - consider normalization

---

## Files to Modify

| File | Changes |
|------|---------|
| [`src/utils/train_utils.py`](src/utils/train_utils.py) | Rename function, update training loop |
| [`src/models.py`](src/models.py) | Update `generate_sequence`, optionally rename parameter |
| [`docs/reference.md`](docs/reference.md) | Update algorithm description |

---

## Testing

After implementation:
1. Run [`tests/test_training_utils.py`](tests/test_training_utils.py) to verify `build_prev_and_clean_positions`
2. Run [`tests/test_flow_predictor.py`](tests/test_flow_predictor.py) to verify model still works
3. Test generation with `generate_sequence` and verify output shapes
