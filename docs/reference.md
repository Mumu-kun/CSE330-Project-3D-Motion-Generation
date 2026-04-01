# Project Reference: Text-to-Motion Generation

This document describes the current implementation in `src/`. It intentionally prefers the code that is running today over older design notes or legacy naming.

## Overview

| Property | Current implementation |
| :-- | :-- |
| Task | Text-conditioned autoregressive 3D human motion generation |
| Dataset format | HumanML3D-style 22-joint sequences with 271D per-frame features |
| Context encoder | `MotionHistoryEncoder` (`GRU` over motion history + pooled CLIP text) |
| Next-frame predictor | `FlowMatchingPredictor` (22-token spatial transformer with AdaLN conditioning) |
| Predictor target space | 68D reduced state, not the older 72D/track-shift description |
| Inference state | Absolute joint positions plus derived 271D feature history |
| Training entrypoint | `Trainer.train(...)` in `src/utils/train_utils.py` |

## Core Files

| File | Purpose |
| :-- | :-- |
| `src/config.py` | Project configuration dataclasses and default hyperparameters |
| `src/models.py` | Encoder, predictor, and `HumanMotionGenerator` |
| `src/utils/motion_utils.py` | 271D feature conversions, reduced-state packing, FK/IK, normalization |
| `src/utils/train_utils.py` | Trainer, EMA, rollout training logic, validation, checkpointing |
| `src/utils/dataset.py` | HumanML3D dataset loader, text embedding cache, dataloader factory |
| `src/utils/text_encoder.py` | CLIP text encoder wrapper returning pooled `(B, 1, 512)` embeddings |

## Data Representations

### 271D frame format

Source: `src/utils/motion_utils.py`

| Slice | Size | Meaning |
| :-- | :-- | :-- |
| `[0:3]` | 3 | Root features: absolute height `y`, root velocity `x`, root velocity `z` |
| `[3:69]` | 66 | Root-invariant coordinates (RIC) for all 22 joints |
| `[69:201]` | 132 | 22 joint rotations in 6D representation |
| `[201:267]` | 66 | Root-local causal joint velocities |
| `[267:271]` | 4 | Foot-contact flags |

Notes:
- Root `x` and `z` are stored as velocities for autoregressive stability.
- The representation is used for dataset loading, GRU history encoding, and frame-by-frame reconstruction.

### Reduced predictor state: 68D

Source: `subset_271d_to_72d(...)` in `src/utils/motion_utils.py`

Despite the legacy function name, the current predictor operates on **68D**:

| Slice | Size | Meaning |
| :-- | :-- | :-- |
| `[0:5]` | 5 | Root state: height, velocity `x/z`, `sin(dyaw)`, `cos(dyaw)` |
| `[5:68]` | 63 | 21 non-root joint RIC coordinates |

This is the target `x1` used by flow matching during training, and the state integrated at inference time.

### Previous-frame conditioning: 257D

Source: `extract_prev_frame_features(...)` in `src/utils/motion_utils.py`

| Slice | Size | Meaning |
| :-- | :-- | :-- |
| `[0:5]` | 5 | Root state: height, velocity `x/z`, `sin(yaw)`, `cos(yaw)` |
| `[5:257]` | 252 | 21 non-root joints, each with `RIC(3) + rot6d(6) + vel(3)` |

This 257D vector conditions the predictor on the current frame while the GRU supplies longer-range motion history.

### Skeleton metadata

Source: `T2M_KINEMATIC_CHAIN` and dataset config in `src/utils/motion_utils.py`

- Joint count: `22`
- FPS: `20`
- Kinematic chains:
  - Left leg: `[0, 2, 5, 8, 11]`
  - Right leg: `[0, 1, 4, 7, 10]`
  - Spine: `[0, 3, 6, 9, 12, 15]`
  - Right arm: `[9, 14, 17, 19, 21]`
  - Left arm: `[9, 13, 16, 18, 20]`
- Facing-direction joints: `[2, 1, 17, 16]`
- Left foot joints: `[7, 10]`
- Right foot joints: `[8, 11]`

## Models

### `MotionHistoryEncoder`

Source: `src/models.py`

Purpose:
- Encodes a variable-length window of 271D frames.
- Conditions every timestep on pooled CLIP text.
- Produces one per-joint context token for the predictor.

Current interface:

```python
MotionHistoryEncoder(
    frame_feature_dim=271,
    text_embedding_dim=512,
    text_proj_dim=128,
    model_dim=512,
    per_joint_out_dim=64,
    num_layers=3,
    joint_count=22,
    text_scale=1.0,
    dropout=0.1,
    normalizer=None,
)
```

Important methods:

```python
forward(motion_seq: Tensor[B, T, 271], text_emb: Tensor[B, 512]) -> Tensor[B, 22, 64]
gru_step(x_t: Tensor[B, 271], text_emb: Tensor[B, 512], h=None, use_normalization=False)
output_dim -> int
```

Implementation notes:
- Text is projected twice:
  - once to initialize GRU hidden state;
  - once to create a per-timestep conditioning vector concatenated with motion features.
- The final GRU hidden state is mapped to `22 * per_joint_out_dim` and reshaped to per-joint tokens.
- `gru_step(...)` is the autoregressive one-frame update used during training rollouts.

### `FlowMatchingPredictor`

Source: `src/models.py`

Purpose:
- Predicts flow in the reduced 68D next-frame state space.
- Uses one token for the root and 21 tokens for non-root joints.
- Conditions on text, time, motion-history tokens, current-frame features, and kinematic-chain embeddings.

Current interface:

```python
FlowMatchingPredictor(
    feature_size=64,
    config=FlowMatchingPredictorConfig(...),
)
```

Current tensor contract:

```python
forward(
    noisy_features: Tensor[B, 68],
    timesteps: Tensor[B] | Tensor[B, 1],
    text_embedding: Tensor[B, 512],
    track_features: Tensor[B, 22, 64],
    current_frame_features: Tensor[B, 257],
) -> tuple[Tensor[B, 68], hidden_states | None, attentions | None]
```

Implementation details:
- Root token input = noisy root state `5D` + root history token + root current-frame features `5D`
- Joint token input = noisy joint state `3D` + joint history token + joint current-frame features `12D`
- Time conditioning uses `SinusoidalEmbedder`
- Text conditioning is projected and added to the time embedding
- Each transformer block applies AdaLN modulation
- Each layer also receives a gated kinematic-chain embedding prior
- Output head predicts `5D` for root and `3D` for each of the 21 non-root joints

### `HumanMotionGenerator`

Source: `src/models.py`

Purpose:
- Wraps the encoder and predictor for autoregressive sampling.
- Uses **absolute positions** as the primary rolling state.
- Re-derives 271D features after every generated frame.

Current interface:

```python
generate_sequence(
    text,
    num_frames=200,
    num_steps=10,
    horizon=None,
    input_positions=None,
    total_duration=None,
    guidance_scale=1.0,
    dataset_type="t2m",
    use_fk=True,
) -> tuple[position_history, feature_history, relative_shift_history]
```

Behavior notes:
- `text` may be a string, list of strings, or pre-encoded `(B, 1, 512)` tensor.
- `input_positions` may be `(B, 22, 3)` or `(B, T, 22, 3)`.
- Cold start initializes a single zero-pose frame and derives its 271D features.
- `guidance_scale` and `total_duration` are present in the signature but are not currently used inside `generate_sequence(...)`.
- `relative_shift_history` is returned for inspection, but the true recurrent state is `position_history` + `feature_history`.

Checkpoint loading:

```python
HumanMotionGenerator.load_from_checkpoint(
    checkpoint_path,
    config,
    device="cpu",
    normalizer=None,
)
```

When available, EMA weights are preferred for generation.

## Feature / Motion Utilities

Source: `src/utils/motion_utils.py`

### Canonical conversions

```python
sequence_joints_to_features(positions) -> features_271
features_to_positions(features_271) -> positions
flow_output_to_positions(flow_output_68, prev_root_pos, prev_root_rot_6d) -> positions
generated_positions_to_271d(new_positions, prev_positions=None, fk_offsets=None, normalizer=None)
```

What each one does:
- `sequence_joints_to_features(...)`: converts ground-truth joint sequences to the 271D format.
- `features_to_positions(...)`: reconstructs global positions from 271D features.
- `flow_output_to_positions(...)`: converts a predicted 68D reduced state into absolute global joint positions.
- `generated_positions_to_271d(...)`: converts one newly generated frame of positions back into a 271D frame, optionally normalizing it and optionally replacing it with FK-consistent positions.

### Normalization

Source: `FeatureNormalizer` in `src/utils/motion_utils.py`

Supported operations:

```python
normalize(features_271)
denormalize(features_271)
normalize_flow_output(flow_output_68)
denormalize_flow_output(flow_output_68)
load_from_files(mean_path, std_path)
```

Important detail:
- 271D normalization uses dataset `Mean.npy` and `Std.npy`.
- 68D normalization reuses only the corresponding height/velocity/RIC statistics.
- The yaw `sin/cos` channels are left in natural scale.

## Dataset Pipeline

Source: `src/utils/dataset.py`

### `Text2MotionDataset`

Current behavior:
- Reads split files from `config.dataset_path`.
- Loads motion from `new_joint_vecs/*.npy`.
- Loads joints from `new_joints/*.npy`.
- Loads text annotations from `texts/*.txt`.
- Filters clips to `40 <= length < 200`.
- Supports timestamped text segments by creating derived sub-clips.
- Returns **raw** motion features; normalization is applied later by the trainer or generator.

Current sample format:

```python
caption, motion, joints, valid_length, text_embedding
```

Shapes:
- `motion`: `(T, 271)`
- `joints`: `(T, 22, 3)`
- `text_embedding`: `(1, 512)`

### Text embeddings

Source: `src/utils/text_encoder.py`

Current implementation:
- Uses Hugging Face `CLIPTokenizer` and `CLIPTextModel`
- Default model: `openai/clip-vit-base-patch32`
- Returns pooled text embeddings with shape `(B, 1, 512)`
- Uses tokenizer truncation; the current code does not implement chunk-and-average long-text handling despite older comments suggesting that behavior

### Dataloader factory

```python
create_dataloader(config, split="train", shuffle=True) -> (dataloader, normalizer)
```

This helper:
- loads `Mean.npy` and `Std.npy`;
- builds `Text2MotionDataset`;
- returns a `FeatureNormalizer` alongside the `DataLoader`.

## Training

Source: `src/utils/train_utils.py`

### Canonical entrypoint

```python
encoder_ema, predictor_ema = Trainer.train(
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    normalizer=normalizer,
    clip_encoder=None,
    wandb_project=None,
    wandb_run_name=None,
    resume_from=None,
    encoder_override=None,
    predictor_override=None,
)
```

### Current training flow

1. Load raw `(B, T, 271)` motion and normalize it if a `FeatureNormalizer` is provided.
2. Drop the first frame so each step has a previous frame and a target next frame.
3. Run `encoder.gru_step(...)` over the sequence one frame at a time.
4. Build:
   - current-frame conditioning with `extract_prev_frame_features(...)`
   - target reduced state with `subset_271d_to_72d(...)` which currently returns `68D`
5. Sample noise `x0`, noise level `t`, and interpolated state `xt = t*x1 + (1-t)*x0`.
6. Predict flow and optimize MSE against `x1 - x0`.
7. Optionally replace some teacher-forced next inputs with model rollouts based on `rollout_prob`.
8. Update EMA models every step and checkpoint periodically.

### Validation

Validation reuses `incremental_flow_loss(...)` with rollout enabled but without CFG dropout.

Reported values:
- `val_loss`
- `val_flow_loss`
- `val_consistency_loss`

At the moment, `consistency_loss` is returned but remains zero in the active training code path.

### EMA, diagnostics, and checkpoints

Implemented features:
- `EMAModel` wrappers for encoder and predictor
- `loss-vs-t` CSV and PNG diagnostics under `output/diagnostics/loss_vs_t`
- checkpoint fields for:
  - live model weights
  - EMA weights
  - optimizer and scaler state
  - epoch and global step
  - best train / validation losses
  - curriculum horizon state
  - serialized `Config`

Saved checkpoints may include:
- `latest.pt`
- `best.pt`
- `best_val.pt`
- interval checkpoints based on `checkpoint_interval`

## Configuration

Source: `src/config.py`

### Active default model settings

```python
motion_dim = 271
num_joints = 22
joint_dim = 3
max_motion_length = 200
fps = 20

encoder_hidden_dim = 512
encoder_text_proj_dim = 128
encoder_per_joint_dim = 64
encoder_num_layers = 3

predictor.hidden_size = 128
predictor.intermediate_size = 384
predictor.num_hidden_layers = 3
predictor.num_attention_heads = 8
```

### Active default training settings

```python
batch_size = 200
learning_rate = 1e-4
num_epochs = 400
weight_decay = 1e-5
gradient_clip = 1.0
ema_decay = 0.999
horizon = 40
cfg_dropout = 0.0
rollout_prob_start = 0.0
rollout_prob_end = 0.0
rollout_integration_steps = 5
val_interval = 5
val_batches = 20
checkpoint_interval = 50
```

### Default curriculum

```python
[
    {"horizon": 5, "epochs": 50},
    {"horizon": 10, "epochs": 100},
    {"horizon": 20, "epochs": 150},
    {"horizon": 40, "epochs": 400},
]
```

### Config fields worth treating cautiously

Some config fields exist for planned or experimental paths and are not central to the current training loop, including:
- `use_consistency_loss`
- `consistency_loss_weight`
- `use_fk` in training config
- degenerate-pose guard thresholds
- `guidance_scale` in config versus generator-time sampling

Keep the source code as the authority before relying on those flags in new work.

## Tests and Fixtures

Current notable tests in `tests/`:
- `test_pipeline_e2e.py`: end-to-end smoke test with a tiny HumanML3D fixture dataset
- `test_human_motion_generator.py`: generator shape / cold-start / NaN checks
- `test_checkpoint_save_and_load.py`: checkpoint serialization and restore
- `test_loss_vs_t_diagnostics.py`: artifact generation for flow-loss diagnostics
- `test_root_motion_conversions.py`: root velocity / position conversion helpers
- `test_fk_consistent_rollout.py`: FK-consistent frame conversion checks

Fixture dataset:
- `tests/dataset/humanml3d-subset-mini`

## Known Naming Mismatches

These are legacy names still present in code or tests:
- `subset_271d_to_72d(...)` currently returns **68D**
- `tests/test_flow_matching_predictor_72d.py` targets the current reduced-state predictor even though the file name still says `72d`

They are naming leftovers, not indicators of the current runtime tensor shapes.
