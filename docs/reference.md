# Project Reference: Text-to-Motion Generation

## Overview

| Property         | Value                                                            |
| :--------------- | :--------------------------------------------------------------- |
| **Architecture** | MotionHistoryEncoder (GRU) + FlowMatchingPredictor (Transformer) |
| **Dataset**      | HumanML3D 271D features, 22 joints, 20fps                        |

## Constants

```
motion_dim: 271
num_joints: 22
joint_dim: 3
fps: 20
max_frames: 200
feature_slices: 0:3(global_root) 3:69(RIC pos 22x3) 69:201(RIC rot 22x6) 201:267(vel 22x3) 267:271(foot)
```

## 271D Feature Layout

*Source: [`src/utils/motion_utils.py`](src/utils/motion_utils.py)*

| Index Range | Feature Type     | Dimension | Description                                           |
| :---------- | :--------------- | :-------- | :---------------------------------------------------- |
| `[0:3]`     | Root Global      | 3D        | Height Y, Velocity X, Velocity Z                      |
| `[3:69]`    | RIC Positions    | 66D       | 22 joints × 3D local positions relative to root       |
| `[69:201]`  | 6D Rotations     | 132D      | 22 joints × 6D rotation (auxiliary from IK)           |
| `[201:267]` | Local Velocities | 66D       | 22 joints × 3D causal velocities (current - previous) |
| `[267:271]` | Foot Contacts    | 4D        | Binary contact flags                                  |

## Joint Structure

*Source: [`src/utils/motion_utils.py:t2m_kinematic_chain`](src/utils/motion_utils.py)*

### Kinematic Chains

| Chain ID | Name      | Joints                 |
| :------- | :-------- | :--------------------- |
| 0        | Left Leg  | `[0, 2, 5, 8, 11]`     |
| 1        | Right Leg | `[0, 1, 4, 7, 10]`     |
| 2        | Spine     | `[0, 3, 6, 9, 12, 15]` |
| 3        | Right Arm | `[9, 14, 17, 19, 21]`  |
| 4        | Left Arm  | `[9, 13, 16, 18, 20]`  |

### Special Joint Groups

- **Face joints**: `[2, 1, 17, 16]`
- **Right foot**: `[8, 11]`

## Troubleshooting

### Issue: Cold Start Generation Fails

**Error:**
```
cannot access local variable 'feature_history' where it is not associated with a value
```

**Root Cause:**
In [`HumanMotionGenerator.generate_sequence()`](src/models.py:533), when `input_features` is `None` (cold start), the code creates a `RootPositionTracker` but doesn't initialize `feature_history`. The loop then fails when trying to access `feature_history[:, -1]`.

**Fix:**
Initialize `feature_history` with a zero frame when `input_features` is `None`:

```python
if input_features is None:
    root_pos = torch.zeros((B, 3), device=device)
    root_tracker = RootPositionTracker(root_pos)
    # Initialize with a zero frame for cold start
    feature_history = torch.zeros((B, 1, 271), device=device)
else:
    # ... existing logic
```

**Verification:**
```bash
python tests/test_human_motion_generator.py
```

All 8 tests should pass including cold start and checkpoint loading.

---

## Normalization

*Source: [`src/utils/motion_utils.py:FeatureNormalizer`](src/utils/motion_utils.py)*

### FeatureNormalizer Methods

| Method                                 | Input         | Output              | Description                 |
| :------------------------------------- | :------------ | :------------------ | :-------------------------- |
| `normalize(features_271d)`             | `(B, T, 271)` | `(B, T, 271)`       | Normalize raw features      |
| `denormalize(features_271d)`           | `(B, T, 271)` | `(B, T, 271)`       | Denormalize to raw features |
| `normalize_flow_output(flow_72d)`      | `(B, 72)`     | `(B, 72)`           | Normalize flow output       |
| `denormalize_flow_output(flow_72d)`    | `(B, 72)`     | `(B, 72)`           | Denormalize flow output     |
| `load_from_files(mean_path, std_path)` | paths         | `FeatureNormalizer` | Load from Mean.npy, Std.npy |

### Data Flow

```
Dataset → RAW features
    │
    ▼
Training: normalize in train_utils.py → normalized features → models(normalize=False)
    │
    ▼
Loss computed in normalized space
    │
    ▼
Inference: models(normalize=True) → normalize internally → ODE in normalized space
    │
    ▼
Denormalize output for reconstruction (via flow_output_to_271d)
```

---

## Training

*Source: [`src/utils/train_utils.py`](src/utils/train_utils.py)*

### Validation During Training

The training loop supports periodic validation to monitor generalization and detect overfitting.

**Configuration** (in [`Config`](src/config.py)):
| Setting         | Type | Default | Description                                            |
| --------------- | ---- | ------- | ------------------------------------------------------ |
| `val_interval`  | int  | 5       | Run validation every N epochs                          |
| `val_batches`   | int  | 20      | Number of validation batches per run (-1 for full set) |
| `val_use_ema`   | bool | True    | Use EMA models for validation (more stable)            |
| `save_best_val` | bool | True    | Save separate checkpoint for best validation loss      |

**Usage:**
```python
from utils.train_utils import train
from config import Config

config = Config(val_interval=5, val_batches=20, val_use_ema=True)

# Create train and validation dataloaders
train_loader = DataLoader(train_dataset, ...)
val_loader = DataLoader(val_dataset, ...)

train(
    encoder=encoder,
    predictor=predictor,
    dataloader=train_loader,
    save_dir="./checkpoints",
    config=config,
    val_dataloader=val_loader,  # Enable validation
)
```

**Output:**
- `val/loss` logged to W&B
- `epoch/val_loss` logged to W&B
- Checkpoints: `best.pt` (best training loss), `best_val.pt` (best validation loss)
- Checkpoints include validation state: `best_val_loss`, `best_val_epoch`

### Train Function

| Parameter        | Type                | Description                    |
| ---------------- | ------------------- | ------------------------------ |
| `encoder`        | `nn.Module`         | MotionHistoryEncoder model     |
| `predictor`      | `nn.Module`         | FlowMatchingPredictor model    |
| `dataloader`     | `DataLoader`        | Training data (RAW features)   |
| `save_dir`       | `str`               | Checkpoint directory           |
| `config`         | `Config`            | Training configuration         |
| `val_dataloader` | `DataLoader`        | Optional validation data       |
| `clip_encoder`   | `nn.Module`         | Optional CLIP encoder for text |
| `normalizer`     | `FeatureNormalizer` | Optional feature normalizer    |

### Validation Function

| Parameter      | Type                | Default | Description                     |
| -------------- | ------------------- | ------- | ------------------------------- |
| `encoder`      | `nn.Module`         | -       | Encoder model (should be EMA)   |
| `predictor`    | `nn.Module`         | -       | Predictor model (should be EMA) |
| `dataloader`   | `DataLoader`        | -       | Validation data                 |
| `horizon`      | `int`               | -       | Current horizon for validation  |
| `device`       | `str`               | "cuda"  | Device to validate on           |
| `num_batches`  | `int`               | 10      | Number of batches to validate   |
| `clip_encoder` | `nn.Module`         | None    | Optional CLIP encoder           |
| `normalizer`   | `FeatureNormalizer` | None    | Optional normalizer             |

**Returns:** `{"val_loss": float}`

---

## Models

*Source: [`src/models.py`](src/models.py)*

### MotionHistoryEncoder (GRU-based Context Encoder)

#### Input/Output

| Tensor       | Shape                    | Description                         |
| :----------- | :----------------------- | :---------------------------------- |
| `motion_seq` | `(B, T, 271)`            | Motion features (raw or normalized) |
| `text_emb`   | `(B, 512)`               | CLIP text embedding (pooled)        |
| `output`     | `(B, 22, per_joint_dim)` | Per-joint context vectors           |

#### `__init__` Parameters

| Parameter            | Type                          | Description                              |
| :------------------- | :---------------------------- | :--------------------------------------- |
| `frame_feature_dim`  | `int`                         | Input motion feature dimension (271)     |
| `text_embedding_dim` | `int`                         | Text embedding dimension (512 from CLIP) |
| `text_proj_dim`      | `int`                         | Text projection dimension                |
| `model_dim`          | `int`                         | GRU hidden dimension                     |
| `per_joint_out_dim`  | `int`                         | Output per-joint dimension               |
| `num_layers`         | `int`                         | Number of GRU layers (default: 2)        |
| `joint_count`        | `int`                         | Number of joints (default: 22)           |
| `text_scale`         | `float`                       | Text conditioning scale (default: 1.0)   |
| `dropout`            | `float`                       | Dropout between GRU layers               |
| `normalizer`         | `Optional[FeatureNormalizer]` | Feature normalizer for preprocessing     |

#### `forward()` Signature

```python
def forward(self, motion_seq: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """
    motion_seq: (B, T, motion_dim)  full history window
    text_emb:   (B, text_dim)       global text embedding
    Returns:
      history_features: (B, 22, per_joint_dim)
    """
```

#### `step()` Signature (for AR inference)

```python
def step(
    self,
    x_t: torch.Tensor,        # (B, motion_dim) single frame
    text_emb: torch.Tensor,    # (B, text_dim)
    h: Optional[torch.Tensor], # (L, B, H) or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-step update for AR inference.
    Returns:
      history_features: (B, 22, per_joint_dim)
      h_next: (L, B, H) next hidden state
    """
```

#### Architecture

1. **Text Projection**: `Linear(text_embedding_dim, text_proj_dim)` → project CLIP embedding
2. **Text to Hidden**: `Linear(text_embedding_dim, model_dim)` → initialize GRU hidden state
3. **GRU**: `input_size=motion_dim + text_proj_dim`, `hidden_size=model_dim`
4. **MLP**: `Linear(model_dim, model_dim) → ReLU → Linear(model_dim, joint_count * per_joint_dim)`
5. **Output**: Reshape to `(B, joint_count, per_joint_dim)`

---

### CLIPEncoder

*Source: [`src/utils/text_encoder.py`](src/utils/text_encoder.py)*

| Property | Value                                                                              |
| :------- | :--------------------------------------------------------------------------------- |
| Input    | `text: str` or `List[str]`                                                         |
| Output   | `(B, 1, 512)` pooled embeddings                                                    |
| Note     | Uses `pooler_output`, not full sequence. `max_text_seq_len` in config is 1, not 77 |

---

### FlowMatchingPredictor

#### Signature

```
(context: Bx22x64, t: B, x_t: Bx72, progress) → v: Bx72
```

#### Inputs

| Tensor              | Shape                    | Description                                |
| :------------------ | :----------------------- | :----------------------------------------- |
| `history_features`  | `(B, 22, per_joint_dim)` | Context from MotionHistoryEncoder          |
| `noise_level`       | `(B,)`                   | Flow time t in [0,1]                       |
| `noisy_target`      | `(B, 72)`                | Current noisy state x_t (normalized space) |
| `temporal_progress` | `(B,)`                   | Optional normalized frame progress         |

#### noisy_target Layout (72D)

| Index    | Content                                        | Dimension |
| :------- | :--------------------------------------------- | :-------- |
| `[0:9]`  | Root: height(1) + velocity(2) + rotation_6d(6) | 9D        |
| `[9:72]` | Joint RIC positions: 21 joints × 3D            | 63D       |

#### Output (72D)

| Index    | Content                                        | Dimension |
| :------- | :--------------------------------------------- | :-------- |
| `[0:9]`  | Root: height(1) + velocity(2) + rotation_6d(6) | 9D        |
| `[9:72]` | Joint RIC: 21 × 3D                             | 63D       |

#### Architecture

1. History projection: `(B, 22, per_joint_dim)` → `(B, 22, model_dim)`
2. Noisy target projection: `root(9D) + joints(63D)` → `(B, 22, model_dim)`
3. Time embedding: sinusoidal → MLP → `(B, model_dim)`
4. Kinematic bias: `KinematicChainEncoder` → `(22, model_dim)`
5. Combine: `history + noisy + time + kinematic` → `(B, 22, model_dim)`
6. Spatial Transformer: 2-4 layers of transformer encoder
7. Output heads: `root_head → (B, 9)`, `joint_head → (B, 21, 3)`
8. Concatenate: `(B, 72)`

---

### Reconstruction Functions

*Source: [`src/utils/motion_utils.py`](src/utils/motion_utils.py)*

#### `features_to_positions(features: Nx271) → positions: Nx22x3`

- Reconstruct global joint positions from 271D features
- Uses cumulative sum along time dimension (dim=0) for root X,Z from velocities
- Verified: MSE=0.0, perfect reconstruction

#### `flow_output_to_positions(flow_output: Bx72, prev_root_pos: Bx3, prev_root_rot_6d: Bx6) → positions: Bx22x3`

- Reconstruct global joint positions from FlowMatchingPredictor output
- Uses predicted root velocity to update position, rotation for coordinate transform
- Verified: MSE=0.0, perfect reconstruction

#### `flow_output_to_displacements(flow_output: Bx72) → displacements: Bx22x3`

- Extract joint displacements from FlowMatchingPredictor output
- Simpler interpretation: output as per-joint deltas to add to current positions
- Verified: Root velocity extraction matches 271D features

#### `flow_output_to_271d(flow_output: Bx72, prev_frame: Bx271, prev_root_pos: Bx3) → (new_frame: Bx271, new_root_pos: Bx3)`

- Incrementally compute next 271D feature frame from flow output
- Computes ALL 22 joint rotations via IK from new_positions (not just root)
- Uses `_compute_ik()` to derive quaternions, then `quaternion_to_cont6d()` for 6D rotations
- Fully Markov and AR-safe; no cumulative sum or full sequence reconstruction

---

### HumanMotionGenerator

#### Signature

```
(text, num_frames, num_steps, guidance_scale, input_features) → joints: (B, T, 22, 3)
```

#### Features

- Integrates MotionHistoryEncoder and FlowMatchingPredictor
- No longer uses `prev_frame_features` (removed from FlowMatchingPredictor)
- Classifier-free guidance: `v = v_uncond + scale * (v_cond - v_uncond)`
- Input text: `str`, `List[str]`, or pre-encoded tensor `(B, 1, 512)` from CLIPEncoder
- `input_features`: Optional, shape `(B, N, 271)` or `(B, 271)` — initial motion history
- `load_from_checkpoint`: Uses `config.max_text_seq_len` (not hardcoded 77)

#### Class Methods

| Method                      | Description                                      |
| :-------------------------- | :----------------------------------------------- |
| `eval()`                    | Set encoder and predictor to eval mode           |
| `train(mode=True)`          | Set encoder and predictor to train mode          |
| `parameters()`              | Yield parameters from both encoder and predictor |
| `to(device)`                | Move models to specified device                  |
| `generate_sequence(...)`    | Generate motion sequence autoregressively        |
| `load_from_checkpoint(...)` | Load from checkpoint (class method)              |

Note: This class does not inherit from `nn.Module` but delegates to its encoder and predictor.

#### History Tracking

| Tensor             | Shape           | Description                           |
| :----------------- | :-------------- | :------------------------------------ |
| `position_history` | `(B, N, 22, 3)` | Global joint positions for all frames |
| `feature_history`  | `(B, N, 271)`   | Feature vectors for encoder context   |

Both grow with each generated frame using `flow_output_to_271d` and `RootPositionTracker`.

#### Autoregressive Loop (per frame)

1. Extract `last_frame` from `feature_history (B, 271)`
2. Get `prev_root_pos` from `RootPositionTracker.get()`
3. Encode context from FULL `feature_history` with CFG (cond and uncond)
4. Flow matching ODE loop → `flow_output (B, 72)`
5. `flow_output_to_271d(flow_output, prev_frame, prev_root_pos)` → `(new_frame, new_root_pos)`
6. `RootPositionTracker.update(new_frame)` → update root position
7. Append `new_frame` to `feature_history`
8. Final: Convert `feature_history` to positions via `features_to_positions`

---

## Configuration

*Source: [`src/config.py`](src/config.py)*

### Model Parameters

| Parameter               | Value | Description                    |
| :---------------------- | :---- | :----------------------------- |
| `encoder_motion_dim`    | 271   | Input motion feature dimension |
| `encoder_text_dim`      | 512   | CLIP embedding dimension       |
| `encoder_text_proj_dim` | 128   | Text projection dimension      |
| `encoder_hidden_dim`    | 256   | GRU hidden dimension           |
| `encoder_per_joint_dim` | 64    | Per-joint output dimension     |
| `encoder_num_layers`    | 2     | Number of GRU layers           |
| `encoder_num_joints`    | 22    | Number of joints               |
| `encoder_text_scale`    | 1.0   | Text conditioning scale        |
| `encoder_dropout`       | 0.1   | GRU dropout rate               |

### FlowMatchingPredictor Parameters

| Parameter                  | Value | Description                      |
| :------------------------- | :---- | :------------------------------- |
| `predictor_per_joint_dim`  | 64    | Must match encoder_per_joint_dim |
| `predictor_model_dim`      | 128   | Spatial transformer hidden dim   |
| `predictor_num_layers`     | 2     | Spatial transformer layers       |
| `predictor_num_heads`      | 4     | Attention heads                  |
| `predictor_time_embed_dim` | 64    | Sinusoidal time embedding        |
| `predictor_dropout`        | 0.1   | Dropout rate                     |

### Training Parameters

| Parameter       | Value | Description              |
| :-------------- | :---- | :----------------------- |
| `batch_size`    | 192   | Training batch size      |
| `learning_rate` | 1e-4  | Adam learning rate       |
| `num_epochs`    | 1000  | Total training epochs    |
| `ema_decay`     | 0.999 | EMA decay for validation |
| `cfg_dropout`   | 0.1   | CFG dropout probability  |

---

## Algorithms

### Flow Matching

```
x_t = t * clean + (1-t) * noise
predict v = clean - noise
loss = MSE(v_pred, v_target)
```

### Classifier-Free Guidance (CFG)

```
v = v_uncond + scale * (v_cond - v_uncond)
x_t += v * dt
```

### Inference (Canonical Round-trip)

```
Initialize: history = input_features[:,-1:] or null_history

Per frame:
  1. last_frame → features_to_positions → prev_positions
  2. CFG loop N steps → flow_output (72D)
  3. flow_output_to_positions → new_positions
  4. stack(prev_positions, new_positions) → sequence_joints_to_features → extract frame 1 → new_frame_271d
  5. history = new_frame_271d.unsqueeze(1)
```

---

## Training

*Source: [`src/utils/train_utils.py`](src/utils/train_utils.py)*

### Progressive Horizon Curriculum

| Stage | Frames         |
| :---- | :------------- |
| 1     | 16             |
| 2     | 32             |
| 3     | 64             |
| 4     | 128 (optional) |

### Training Loop

#### Data Preparation

1. Unpack batch: `motion_raw(B, T, 271)` — RAW features from dataset
2. Normalize: `motion = normalizer.normalize(motion_raw)` if normalizer provided
3. Sample window based on horizon (uses `batch["lengths"]` for padding awareness)
4. Extract history: `hist = motion[:, start_idx:end_idx]` (normalized)
5. Extract targets: `target_frames = motion[:, start_idx+1:end_idx+1]` (normalized)

#### Teacher Forcing Training (Primary)

1. CFG dropout: `text_input = text if rand() > cfg_dropout else None`
2. Encode context: `contexts = encoder(text=text_input, input_features=hist, normalize=False)`
3. Extract prediction contexts: `pred_contexts = contexts[:, -num_pred_frames:]`
4. Flatten for predictor: `contexts_flat(B*N, 22, D)`, `prev_flat(B*N, 271)`, `targets_flat(B*N, 271)`
5. Extract features: `prev_features = extract_prev_frame_features(prev_flat)` → `(B*N, 261)`
6. Extract clean targets: `clean_targets = extract_clean_target(targets_flat)` → `(B*N, 72)`
7. Flow matching: sample `t ~ U(0,1)`, create `x_t = t*clean + (1-t)*noise`
8. Predict: `pred = predictor(contexts_flat, t, x_t, prev_features, normalize=False)`
9. Loss: `loss_tf = MSE(pred, clean_targets - noise)`

#### Rollout Training (Commented Out)

- Code exists but disabled (`loss_ar = 0`, `lambda_ar = 0.5`)
- Uses mini ODE sampling (10 steps) for autoregressive training
- Updates history with predicted frames via `flow_output_to_271d`

#### Optimization

- Single AdamW optimizer for both encoder and predictor
- Mixed precision training (CPU-safe): `torch.amp.autocast`
- Gradient clipping: `clip_grad_norm_(params, max_grad_norm)`
- EMA update every step: `encoder_ema.update(encoder)`, `predictor_ema.update(predictor)`

### Feature Extraction Helpers

#### `extract_prev_frame_features(frame: Bx271) → (B, 261)`

| Index     | Content                              | Dimension |
| :-------- | :----------------------------------- | :-------- |
| `[0:9]`   | Root: height(1) + vel(2) + rot_6d(6) | 9D        |
| `[9:261]` | Joints: 21 × 12D (RIC + rot + vel)   | 252D      |

#### `extract_clean_target(frame: Bx271) → (B, 72)`

| Index    | Content                              | Dimension |
| :------- | :----------------------------------- | :-------- |
| `[0:9]`  | Root: height(1) + vel(2) + rot_6d(6) | 9D        |
| `[9:72]` | Joint RIC: 21 × 3D                   | 63D       |

### EMA Model Management

- `EMAModel` class wraps model with `decay=0.999`
- Updated every training step
- Used for validation and checkpointing
- Checkpoints save both regular and EMA weights

### Checkpointing

| File        | Trigger                           |
| :---------- | :-------------------------------- |
| `latest.pt` | Every epoch                       |
| `best.pt`   | When `avg_epoch_loss < best_loss` |

**Checkpoint Contents**: encoder, predictor, encoder_ema, predictor_ema, optimizer, scaler, epoch, global_step

### Key Design Decisions

- Full teacher forcing (no scheduled sampling)
- Fixed learning rate (no scheduling)
- Simple MSE loss on velocity field prediction
- CFG dropout for conditional generation capability
- Manual stage advancement (no automatic progression)
- External normalization (models receive pre-normalized features with `normalize=False`)

---

## Dependencies

```
config → (none)
models ← config, motion_utils, text_encoder
dataset ← config, motion_utils
motion_utils ← quaternion
train_utils ← models, dataset
visualization ← motion_utils
```

---

## Files

| File                                                     | Purpose                               |
| :------------------------------------------------------- | :------------------------------------ |
| [`src/config.py`](src/config.py)                         | Hyperparameters                       |
| [`src/models.py`](src/models.py)                         | MHE, FMP, Generator                   |
| [`src/utils/dataset.py`](src/utils/dataset.py)           | Text2MotionDataset                    |
| [`src/utils/motion_utils.py`](src/utils/motion_utils.py) | Features, IncrementalFeatureExtractor |
| [`src/utils/train_utils.py`](src/utils/train_utils.py)   | Training loop, EMA                    |
| [`src/utils/quaternion.py`](src/utils/quaternion.py)     | qrot, qmul, qinv                      |
| [`src/utils/text_encoder.py`](src/utils/text_encoder.py) | CLIP encoding                         |

---

## Tests

*Update test files on code interface change; remove previous redundant tests if new test is written*

| Test File                                                              | Purpose                                                                         |
| :--------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| [`tests/test_training_loop.py`](tests/test_training_loop.py)           | Training loop and HumanMotionGenerator verification (8 tests)                   |
| [`tests/test_rotation_roundtrip.py`](tests/test_rotation_roundtrip.py) | Rotation roundtrip verification for flow_output_to_271d                         |
| [`tests/test_flow_predictor.py`](tests/test_flow_predictor.py)         | FlowMatchingPredictor unit tests (I/O shapes: 72D noisy, 261D prev, 72D output) |
| [`tests/test_motion_encoder.py`](tests/test_motion_encoder.py)         | MotionHistoryEncoder unit tests                                                 |
| [`tests/test_nan_fix.py`](tests/test_nan_fix.py)                       | HumanMotionGenerator.generate_sequence() NaN fix verification                   |
| [`tests/test_nan_debug.py`](tests/test_nan_debug.py)                   | Debug test for tracing NaN propagation                                          |
| [`tests/test_pose_validation.py`](tests/test_pose_validation.py)       | Pose validation unit tests                                                      |

---

## Pose Validation

*Source: [`src/utils/pose_validation.py`](src/utils/pose_validation.py)*

### `validate_pose(positions, check_bone_ratios, check_ric_bounds, check_kinematic_chain, report_scaling) → PoseValidationResult`

- Comprehensive pose validation for generated motion
- Checks bone length ratios within kinematic chains (handles scaling)
- Validates RIC positions against dataset-derived bounds
- Detects joint dislocations and chain discontinuities
- Reports scaling deviation metrics

### `validate_pose_sequence(joints, frame_threshold) → (overall_result, per_frame_results)`

- Frame-by-frame validation of motion sequences
- Returns aggregated and per-frame results

### PoseValidationResult

| Property               | Type                   |
| :--------------------- | :--------------------- |
| `is_valid`             | `bool`                 |
| `bone_ratio_issues`    | `List[BoneRatioIssue]` |
| `ric_bound_violations` | `Dict[str, List[int]]` |
| `kinematic_issues`     | `List[KinematicIssue]` |
| `scaling_metrics`      | `ScalingMetrics`       |

**Methods**:
- `summary() → str`: Human-readable summary
- `to_dict() → Dict`: For logging/serialization

### Usage

```python
result = validate_pose(joints)  # (B, 22, 3) or (N, 22, 3)
print(result.summary())
if not result.is_valid:
    print(f"Found {len(result.kinematic_issues)} kinematic issues")
```
