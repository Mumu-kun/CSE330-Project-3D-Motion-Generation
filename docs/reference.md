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

### Issue: Cold Start Generation Failed (Historical)

**Error:**
```
cannot access local variable 'feature_history' where it is not associated with a value
```

**Root Cause:**
In older `generate_sequence()` versions, the cold-start branch did not initialize history buffers before the autoregressive loop.

**Fix:**
Current implementation initializes both position and feature history when `input_positions` is `None`:

```python
if input_positions is None:
    position_history = torch.zeros((B, 1, 22, 3), device=device)
    feature_history = sequence_joints_to_features(position_history)
else:
    # ... existing logic
```

**Verification:**
```bash
pytest tests/test_human_motion_generator.py
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

### Incremental Conversion Note

- `generated_positions_to_271d(..., normalizer=None)` returns raw 271D features by default.
- If `normalizer` is provided, only `new_frame` is normalized; `new_root_pos` remains absolute global `(B, 3)`.
- Trainer rollout and `HumanMotionGenerator.generate_sequence()` pass their configured normalizer so AR updates stay in the same feature space as encoder inputs.

---

## Training

*Source: [`src/utils/train_utils.py`](src/utils/train_utils.py)*

### Validation During Training

The training loop supports periodic validation to monitor generalization and detect overfitting.

**Configuration** (in [`Config`](src/config.py)):
| Setting                     | Type  | Default | Description                                                   |
| --------------------------- | ----- | ------- | ------------------------------------------------------------- |
| `val_interval`              | int   | 5       | Run validation every N epochs                                 |
| `val_batches`               | int   | 20      | Number of validation batches per run (-1 for full set)        |
| `val_use_ema`               | bool  | True    | Use EMA models for validation (more stable)                   |
| `save_best_val`             | bool  | True    | Save separate checkpoint for best validation loss             |
| `enable_profiling`          | bool  | False   | Enable timing instrumentation for train loop and forward pass |
| `timing_log_interval`       | int   | 100     | Log averaged timing metrics every N training steps            |
| `rollout_prob_start`        | float | 0.0     | Rollout probability at epoch 0 for AR routing                 |
| `rollout_prob_end`          | float | 0.5     | Rollout probability at final epoch for AR routing             |
| `rollout_integration_steps` | int   | 10      | Number of ODE integration steps in rollout branch             |
| `use_consistency_loss`      | bool  | False   | Enables endpoint consistency recompute after no-grad rollout  |

Rollout probability is linearly interpolated from `rollout_prob_start` to `rollout_prob_end` and is applied in both training and validation loss unroll.
When `use_consistency_loss` is enabled, rollout integration stays in no-grad and consistency gradients come from a single endpoint recompute on rolled samples.

**Usage:**
```python
from utils.train_utils import Trainer
from config import Config

config = Config(val_interval=5, val_batches=20, val_use_ema=True)

# Create train and validation dataloaders
train_loader = DataLoader(train_dataset, ...)
val_loader = DataLoader(val_dataset, ...)

# Canonical API
encoder_ema, predictor_ema = Trainer.train(
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)
```

**Output:**
- `val/loss` logged to W&B
- `epoch/val_loss` logged to W&B
- Checkpoints: `best.pt` (best training loss), `best_val.pt` (best validation loss)
- Checkpoints include validation state: `best_val_loss`, `best_val_epoch`

When profiling is enabled, `Trainer` also records and logs averaged timing metrics:
- Per-step interval logs: `time/{op}_ms` (e.g., `time/forward_ms`, `time/forward/predictor_flow_ms`, `time/backward_ms`)
- End-of-epoch logs: `epoch_time/{op}_ms`
- Forward breakdown keys: `forward/gru_init`, `forward/predictor_flow`, `forward/rollout_ode`, `forward/rollout_ode_step`, `forward/consistency_pred`, `forward/pos_transform`, `forward/gru_step`

### Train Classmethod

| Parameter            | Type                | Description                                   |
| -------------------- | ------------------- | --------------------------------------------- |
| `config`             | `Config`            | Training configuration                        |
| `train_dataloader`   | `DataLoader`        | Training data (RAW features)                  |
| `val_dataloader`     | `DataLoader`        | Optional validation data                      |
| `clip_encoder`       | `nn.Module`         | Optional CLIP encoder for text                |
| `normalizer`         | `FeatureNormalizer` | Optional feature normalizer                   |
| `encoder_override`   | `nn.Module`         | Optional explicit encoder (tests/custom only) |
| `predictor_override` | `nn.Module`         | Optional explicit predictor (tests/custom)    |

### Trainer Class

`Trainer` is available in `src/utils/train_utils.py` and stores global training
dependencies/config once in `__init__`, with canonical entrypoint
`Trainer.train(config, train_dataloader, val_dataloader=...)`.
Most training utilities are exposed as `Trainer` methods (e.g.,
`setup_training_environment`, `setup_curriculum_state`, `unpack_batch`,
`incremental_flow_loss`, `validate`, `handle_checkpointing`) so shared config
and runtime state can be consumed directly from the class instance.

### Validation Function

| Parameter            | Type                | Default | Description                        |
| -------------------- | ------------------- | ------- | ---------------------------------- |
| `encoder`            | `nn.Module`         | -       | Encoder model (should be EMA)      |
| `predictor`          | `nn.Module`         | -       | Predictor model (should be EMA)    |
| `dataloader`         | `DataLoader`        | -       | Validation data                    |
| `horizon`            | `int`               | -       | Current horizon for validation     |
| `rollout_prob_start` | `float`             | -       | Rollout probability at first epoch |
| `rollout_prob_end`   | `float`             | -       | Rollout probability at final epoch |
| `device`             | `str`               | "cuda"  | Device to validate on              |
| `num_batches`        | `int`               | 10      | Number of batches to validate      |
| `clip_encoder`       | `nn.Module`         | None    | Optional CLIP encoder              |
| `normalizer`         | `FeatureNormalizer` | None    | Optional normalizer                |

**Returns:** `{"val_loss": float}`

### E2E Smoke Test

*Source: [`tests/test_pipeline_e2e.py`](tests/test_pipeline_e2e.py)*

- Uses a tiny fixture dataset at `tests/dataset/humanml3d-subset-mini` (derived from `sample_data/humanml3d-subset`) for fast CPU runs.
- Validates current `MotionHistoryEncoder` and `FlowMatchingPredictor` tensor contracts (`(B,T,271)` input motion, `(B,22,3)` predictor output).
- Validates training utilities with W&B disabled via `setup_training_environment(..., wandb_project=None)` and checkpoint save helper.
- Produces checkpoint artifact at `tests/checkpoints/test_e2e/latest.pt`.
- Run with: `python tests/test_pipeline_e2e.py`

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

| Property | Value                                                   |
| :------- | :------------------------------------------------------ |
| Input    | `text: str` or `List[str]`                              |
| Output   | `(B, 1, 512)` pooled embeddings                         |
| Note     | Uses `pooler_output` with explicit singleton token axis |

**Strict text-shape policy**
- Dataset/cache sample embedding: `(1, 512)`
- Dataloader `text_clip`: `(B, 1, 512)`
- Encoder/predictor conditioning tensor: `(B, 512)` after explicit squeeze at the model boundary
- Legacy sequence embeddings `(77, 512)` are rejected and require cache regeneration

---

### FlowMatchingPredictor

#### Signature

```
(noised_tracks: Bx22xD, timesteps: B, text_embedding: BxF,
 track_features: Bx22xF, relative_shifts: Bx22xD|None)
    -> (flow_prediction: Bx22xD, hidden_states|None, attentions|None)
```

#### Inputs

| Tensor            | Shape                  | Description                                              |
| :---------------- | :--------------------- | :------------------------------------------------------- |
| `noised_tracks`   | `(B, 22, D)`           | Current noisy track-space state in ODE integration       |
| `timesteps`       | `(B,)` or `(B,1)`      | Flow time $t \in [0,1]$                                  |
| `text_embedding`  | `(B, F)`               | Text embedding projected to global conditioning          |
| `track_features`  | `(B, 22, F)`           | Per-track context features from `MotionHistoryEncoder`   |
| `relative_shifts` | `(B, 22, D)` or `None` | Optional precomputed relative shifts (used when enabled) |

#### Output

| Tensor            | Shape                    | Description                                                      |
| :---------------- | :----------------------- | :--------------------------------------------------------------- |
| `flow_prediction` | `(B, 22, D)`             | Predicted flow field of clean next relative shift in track space |
| `hidden_states`   | `Optional[List[Tensor]]` | Per-layer hidden states (when requested)                         |
| `attentions`      | `Optional[List[Tensor]]` | Per-layer attention weights (when requested)                     |

#### Prediction Goal

The predictor is trained and used to model the flow field that transports noisy track-space states toward the clean next relative shift. During inference, ODE integration yields a clean next relative shift, which is then applied as displacement to current 22-track positions.

#### Architecture

1. Concatenate `noised_tracks`, `track_features`, and optionally `relative_shifts`.
2. Project concatenated per-track features to predictor hidden size.
3. Embed `timesteps` with `SinusoidalEmbedder` and combine with projected `text_embedding` for AdaLN conditioning.
4. Process with stacked `SpatialTrackLayer` blocks (MHA + gated MLP, AdaLN-modulated, no attention masking).
5. Apply output LayerNorm + output AdaLN modulation.
6. Project to `out_channels` to produce `(B, 22, D)` flow prediction.

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
(text, num_frames, num_steps, horizon, input_positions, total_duration, guidance_scale, dataset_type)
    -> (position_history, feature_history, prev_relative_shifts)
```

#### Features

- Integrates MotionHistoryEncoder and FlowMatchingPredictor
- Uses track-space ODE state `x_t` with shape `(B, 22, D)`
- Uses relative-shift conditioning (`relative_shifts = x_t - x_t[:, :1, :]`) when enabled
- Input text: `str`, `List[str]`, or pre-encoded tensor `(B, 1, 512)`
- `input_positions`: Optional, shape `(B, 22, 3)` or `(B, N, 22, 3)` — initial global positions
- `load_from_checkpoint`: Rebuilds models from `Config` + checkpoint weights (prefers EMA)

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

| Tensor                 | Shape           | Description                           |
| :--------------------- | :-------------- | :------------------------------------ |
| `position_history`     | `(B, N, 22, 3)` | Global joint positions for all frames |
| `feature_history`      | `(B, N, 271)`   | Feature vectors for encoder context   |
| `prev_relative_shifts` | `(B, N, 22, 3)` | Per-frame relative shifts             |

All three histories grow each generated frame.

#### Autoregressive Loop (per frame)

1. Extract `current_positions` from `position_history[:, -1]`
2. Encode context from recent `feature_history` frames to `context_cond (B, 22, per_joint_dim)`
3. Initialize `x_t ~ N(0, I)` in track space `(B, 22, D)`
4. ODE loop: predict `pred = predictor(...)[0]`, update `x_t = x_t + pred * dt`
5. Interpret integrated `x_t` as relative shift and compute `new_positions = current_positions + x_t`
6. Derive `new_frame` via `generated_positions_to_271d(...)`
7. Append `new_positions`, `new_frame`, and relative shift histories

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
| `encoder_num_layers`    | 4     | Number of GRU layers           |
| `encoder_num_joints`    | 22    | Number of joints               |
| `encoder_text_scale`    | 1.0   | Text conditioning scale        |
| `encoder_dropout`       | 0.1   | GRU dropout rate               |

### FlowMatchingPredictor Parameters

| Parameter                  | Value | Description                      |
| :------------------------- | :---- | :------------------------------- |
| `predictor_per_joint_dim`  | 64    | Must match encoder_per_joint_dim |
| `predictor_model_dim`      | 256   | Spatial transformer hidden dim   |
| `predictor_num_layers`     | 4     | Spatial transformer layers       |
| `predictor_num_heads`      | 8     | Attention heads                  |
| `predictor_time_embed_dim` | 64    | Sinusoidal time embedding        |
| `predictor_dropout`        | 0.1   | Dropout rate                     |

### Training Parameters

| Parameter       | Value | Description              |
| :-------------- | :---- | :----------------------- |
| `batch_size`    | 192   | Training batch size      |
| `learning_rate` | 1e-4  | Adam learning rate       |
| `num_epochs`    | 2000  | Total training epochs    |
| `ema_decay`     | 0.999 | EMA decay for validation |
| `cfg_dropout`   | 0.1   | CFG dropout probability  |

### Curriculum Defaults

| Stage | Horizon | Epoch Target |
| :---- | :------ | :----------- |
| 1     | 1       | 500          |
| 2     | 2       | 900          |
| 3     | 5       | 1200         |
| 4     | 10      | 1400         |
| 5     | 20      | 1600         |
| 6     | 40      | 2000         |

---

## Algorithms

### Flow Matching

```
x_t in R^(B x 22 x D)
predict flow f_theta(x_t, t, cond) for clean next relative shift
x_t <- x_t + f_theta(x_t, t, cond) * dt
after integration: Delta_rel_next_clean = x_t
next_tracks = current_tracks + Delta_rel_next_clean
```

### Inference (Canonical Round-trip)

```
Initialize: position_history = input_positions or zero cold-start frame
            feature_history = sequence_joints_to_features(position_history)

Per frame:
    1. recent feature history + text -> context_cond (B,22,F)
    2. x_t = randn(B,22,D)
    3. repeat N ODE steps:
             relative_shifts = x_t - x_t[:, :1, :]
             pred = predictor(noised_tracks=x_t, timesteps=t,
                                                text_embedding=text,
                                                track_features=context_cond,
                                                relative_shifts=relative_shifts)[0]
             x_t = x_t + pred * dt
    4. interpret x_t as clean next relative shift; displace current tracks
    5. convert to 271D via generated_positions_to_271d and append to history
```

---

## Training

*Source: [`src/utils/train_utils.py`](src/utils/train_utils.py)*

### Progressive Horizon Curriculum

| Stage | Frames |
| :---- | :----- |
| 1     | 1      |
| 2     | 2      |
| 3     | 5      |
| 4     | 10     |
| 5     | 20     |
| 6     | 40     |

### Training Loop

#### Data Preparation

1. Unpack batch: `motion_raw(B, T, 271)` — RAW features from dataset
2. Normalize: `motion = normalizer.normalize(motion_raw)` if normalizer provided
3. Sample window based on horizon (uses `batch["lengths"]` for padding awareness)
4. Extract history: `hist = motion[:, start_idx:end_idx]` (normalized)
5. Extract targets: `target_frames = motion[:, start_idx+1:end_idx+1]` (normalized)

#### Teacher Forcing Training (Primary)

1. CFG dropout: `text_input = text if rand() > cfg_dropout else None`
2. Encode per-track context from history using `MotionHistoryEncoder`
3. Build clean next relative-shift target `Delta_rel_next_clean (B*N, 22, D)`
4. Sample flow time `t ~ U(0,1)` and noise `eps ~ N(0,I)` in track space
5. Construct noisy state: `x_t = t * Delta_rel_next_clean + (1 - t) * eps`
6. Compute `relative_shifts = x_t - x_t[:, :1, :]` (if enabled)
7. Predict flow: `pred = predictor(noised_tracks=x_t, timesteps=t, text_embedding=text_for_encoder, track_features=contexts_flat, relative_shifts=relative_shifts)[0]`
8. Target flow: `v_target = Delta_rel_next_clean - eps`
9. Loss: `loss_tf = MSE(pred, v_target)`

#### Rollout Training (Commented Out)

- Code exists but disabled (`loss_ar = 0`, `lambda_ar = 0.5`)
- Uses mini ODE sampling (10 steps) for autoregressive training
- Updates history with predicted frames via `flow_output_to_271d`

#### Optimization

- Single AdamW optimizer for both encoder and predictor
- Mixed precision training (CPU-safe): `torch.amp.autocast`
- Gradient clipping: `clip_grad_norm_(params, max_grad_norm)`
- EMA update every step: `encoder_ema.update(encoder)`, `predictor_ema.update(predictor)`

### Track-Space Target Construction

| Tensor                 | Shape          | Description                                               |
| :--------------------- | :------------- | :-------------------------------------------------------- |
| `Delta_rel_next_clean` | `(B*N, 22, D)` | Clean next relative shift target for flow matching        |
| `eps`                  | `(B*N, 22, D)` | Gaussian noise used for interpolation target construction |
| `x_t`                  | `(B*N, 22, D)` | Noisy interpolated state at flow time `t`                 |
| `v_target`             | `(B*N, 22, D)` | Supervision target `Delta_rel_next_clean - eps`           |

Note: Legacy 72D/261D extraction paths may remain in compatibility code during migration, but the predictor objective is now defined in track space.

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

| Test File                                                                                        | Purpose                                                                                                                                             |
| :----------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`tests/test_training_loop.py`](tests/test_training_loop.py)                                     | Training loop and HumanMotionGenerator verification (7 tests)                                                                                       |
| [`tests/test_rotation_roundtrip.py`](tests/test_rotation_roundtrip.py)                           | Rotation roundtrip verification for flow_output_to_271d                                                                                             |
| [`tests/test_flow_matching_predictor_new_api.py`](tests/test_flow_matching_predictor_new_api.py) | FlowMatchingPredictor new API tests (`text_embedding`, optional `relative_shifts`)                                                                  |
| [`tests/test_human_motion_generator.py`](tests/test_human_motion_generator.py)                   | HumanMotionGenerator API/cold-start/checkpoint coverage (8 tests)                                                                                   |
| [`tests/test_pipeline_e2e.py`](tests/test_pipeline_e2e.py)                                       | End-to-end smoke tests for data/model/training utility integration (5 tests)                                                                        |
| [`tests/test_checkpoint_save_and_load.py`](tests/test_checkpoint_save_and_load.py)               | Pytest coverage for checkpoint save latency/overhead and HumanMotionGenerator load behavior (EMA priority/fallback/config/missing-key/full-payload) |
| [`tests/test_predictor_minimal.py`](tests/test_predictor_minimal.py)                             | Script-style minimal predictor API sanity check (run with `python`, not pytest)                                                                     |

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
