# Project Reference: Text-to-Motion Generation

This document describes the implementation that currently lives in `src/`. When comments, older notebooks, or archived notes disagree, prefer the source code.

## Overview

| Property | Current implementation |
| :-- | :-- |
| Task | Text-conditioned autoregressive 3D human motion generation |
| Dataset format | HumanML3D-style 22-joint sequences with 271D per-frame features |
| Context encoder | `MotionHistoryEncoder` using causal temporal self-attention with RoPE |
| Next-frame predictor | `FlowMatchingPredictor` using a 22-token spatial transformer with AdaLN conditioning |
| Predictor target space | 68D reduced motion state |
| Inference state | Absolute positions plus derived 271D feature history |
| Training entrypoint | `Trainer.train(...)` in `src/utils/train_utils.py` |

## Core Files

| File | Purpose |
| :-- | :-- |
| `src/config.py` | Project configuration dataclasses and active defaults |
| `src/models.py` | Temporal encoder, predictor, ODE integrator, and `HumanMotionGenerator` |
| `src/utils/motion_utils.py` | 271D feature conversions, reduced-state packing, IK/FK helpers, normalization |
| `src/utils/train_utils.py` | Trainer, EMA, diagnostics, validation, checkpointing |
| `src/utils/dataset.py` | HumanML3D dataset loader, text embedding cache, dataloader factory |
| `src/utils/text_encoder.py` | CLIP wrapper returning pooled embeddings of shape `(B, 1, 512)` |

## Data Representations

### 271D frame format

Source: `src/utils/motion_utils.py`

| Slice | Size | Meaning |
| :-- | :-- | :-- |
| `[0:3]` | 3 | Root features: absolute height `y`, root velocity `x`, root velocity `z` |
| `[3:69]` | 66 | Root-invariant coordinates (RIC) for all 22 joints |
| `[69:201]` | 132 | 22 joint rotations in 6D form |
| `[201:267]` | 66 | Root-local causal joint velocities |
| `[267:271]` | 4 | Foot-contact flags |

Notes:
- Root `x` and `z` are stored as velocities for autoregressive stability.
- This representation is used for dataset loading, temporal encoding, normalization, and frame-by-frame reconstruction.

### Reduced predictor state: 68D

Source: `subset_271d_to_68d(...)` in `src/utils/motion_utils.py`

| Slice | Size | Meaning |
| :-- | :-- | :-- |
| `[0:5]` | 5 | Root state: height, velocity `x/z`, `sin(dyaw)`, `cos(dyaw)` |
| `[5:68]` | 63 | 21 non-root joint RIC coordinates |

This is the state used for the flow-matching target `x1` during training and the state integrated by the inference-time ODE solver.

### Previous-frame conditioning: 257D

Source: `extract_prev_frame_features(...)` in `src/utils/motion_utils.py`

| Slice | Size | Meaning |
| :-- | :-- | :-- |
| `[0:5]` | 5 | Root state: height, velocity `x/z`, `sin(yaw)`, `cos(yaw)` |
| `[5:257]` | 252 | 21 non-root joints, each with `RIC(3) + rot6d(6) + vel(3)` |

This 257D vector conditions the predictor on the current frame while the temporal encoder supplies longer-range history.

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
- Applies causal temporal attention with rotary position encoding.
- Uses text-conditioned shift/scale modulation on every timestep.
- Produces one per-joint context token for the predictor.

Current config dataclass:

```python
MotionHistoryEncoderConfig(
    frame_feature_dim=271,
    text_embedding_dim=512,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=3,
    num_attention_heads=8,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    attention_bias=True,
    attention_dropout=0.1,
    mlp_bias=True,
    dropout=0.1,
    per_joint_output_dim=64,
    joint_count=22,
    text_scale=1.0,
)
```

Active defaults inside `Config()`:

```python
hidden_size=256
intermediate_size=512
num_hidden_layers=4
num_attention_heads=8
per_joint_output_dim=64
```

Important methods:

```python
forward(
    motion_seq: Tensor[B, T, 271],
    text_emb: Tensor[B, 512],
    return_all: bool = False,
) -> Tensor[B, 22, 64] | Tensor[B, T, 22, 64]

step(
    x_t: Tensor[B, 271],
    text_emb: Tensor[B, 512],
    frame_buffer: Tensor[B, T, 271] | None,
    cache_state: TemporalCacheState | None = None,
) -> tuple[Tensor[B, 22, 64], Tensor[B, T+1, 271], TemporalCacheState]

output_dim -> int
```

Implementation notes:
- `frame_projection` maps each 271D frame into model space.
- `text_projection` outputs `2 * hidden_size`, which is split into per-sample shift and scale terms.
- Each temporal layer is `LayerNorm -> causal RoPE attention -> LayerNorm -> gated MLP`.
- `global_to_joints` maps each timestep to `22 * per_joint_output_dim`, then reshapes to per-joint tokens.
- `return_all=True` returns contexts for every timestep and is the path used by training.
- `step(...)` is used by autoregressive generation. It concatenates the new frame onto `frame_buffer` and runs a full forward pass over that buffer.
- `TemporalCacheState` exists, but there is no real key/value caching yet. The cache object is mainly a placeholder interface right now.

### `FlowMatchingPredictor`

Source: `src/models.py`

Purpose:
- Predicts flow in the reduced 68D next-frame state space.
- Reasons jointly over root and non-root joints using attention across 22 tokens.

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
- A `KinematicChainEncoder` provides a fixed per-joint structural prior
- Each layer adds the structural prior through a learned bounded gate
- Output head predicts `5D` for root and `3D` for each of the 21 non-root joints

### `HumanMotionGenerator`

Source: `src/models.py`

Purpose:
- Wraps the encoder and predictor for autoregressive sampling.
- Uses absolute positions as the primary rolling state.
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
- Context trimming is controlled by the `horizon` argument, falling back to `config.horizon` when omitted.
- The encoder no longer owns a separate context-length limit.
- `guidance_scale` and `total_duration` are present in the signature but are not currently used inside `generate_sequence(...)`.
- `relative_shift_history` is returned for inspection; the true recurrent state is `position_history` plus `feature_history`.

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
- `generated_positions_to_271d(...)`: converts one newly generated frame of positions back into a 271D frame, optionally normalizing it and optionally producing FK-consistent positions.

### Normalization

Source: `FeatureNormalizer` in `src/utils/motion_utils.py`

Supported operations:

```python
normalize(features_271)
denormalize(features_271)
normalize_flow_output(flow_output_68)
denormalize_flow_output(flow_output_68)
normalize_current_frame_features(features_257)
load_from_files(mean_path, std_path)
```

Important detail:
- 271D normalization uses dataset `Mean.npy` and `Std.npy`.
- 68D normalization reuses only the corresponding height, velocity, and RIC statistics.
- Yaw `sin/cos` channels remain in natural scale.

## Dataset Pipeline

Source: `src/utils/dataset.py`

### `Text2MotionDataset`

Current behavior:
- Reads split files from `config.dataset_path`.
- Loads motion from `new_joint_vecs/*.npy`.
- Loads joints from `new_joints/*.npy`.
- Loads text annotations from `texts/*.txt`.
- Filters clips to `40 <= length < 200`.
- Supports timestamped text segments by materializing sub-clips.
- Returns raw motion features; normalization is applied later by the trainer or generator.
- Uses `set_horizon(...)` to pad or crop returned samples to the requested sequence length.

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
- Returns pooled embeddings with shape `(B, 1, 512)`
- Uses tokenizer truncation; the current code does not implement long-text chunk averaging

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

`Trainer.incremental_flow_loss(...)` is the core loss builder. The current implementation is parallel over all available history prefixes rather than using an online recurrent step API.

1. Load raw `(B, T, 271)` motion and normalize it if a `FeatureNormalizer` is available.
2. Optionally zero text conditioning for CFG-style dropout.
3. Drop the first frame so each prediction has a previous frame.
4. Build:
   - `hist = motion[:, :-1]`
   - `target_motion = motion[:, 1:]`
5. Run `enc(hist, text_for_encoder, return_all=True)` to obtain per-timestep context tokens.
6. Flatten batch and time dimensions.
7. Build:
   - current-frame conditioning with `extract_prev_frame_features(...)`
   - target reduced state with `subset_271d_to_68d(...)`
8. Sample Gaussian noise `x0`, noise level `t`, and interpolated state `xt = t*x1 + (1-t)*x0`.
9. Predict flow and optimize MSE against `x1 - x0`.
10. Optionally add consistency loss for samples with `t > consistency_loss_t_threshold`.

### Rollout status

There are rollout-schedule helpers and config fields, but the current `incremental_flow_loss(...)` path does not actually perform stochastic rollout replacement. The relevant code is commented out and the method currently returns `rollout_prob = 0.0`.

Treat these fields as preparatory or dormant until rollout logic is re-enabled:
- `rollout_prob_start`
- `rollout_prob_end`
- `rollout_warmup_fraction`
- `rollout_block_len_start`
- `rollout_block_len_end`
- `rollout_integration_steps`

### Validation

Validation reuses `incremental_flow_loss(...)` without CFG dropout.

Reported values:
- `val_loss`
- `val_flow_loss`
- `val_consistency_loss`

Unlike older notes, consistency loss is not just a placeholder. It is active by default in `Config()` and contributes to `total_loss` when enabled.

### EMA, diagnostics, and checkpoints

Implemented features:
- `EMAModel` wrappers for encoder and predictor
- loss-vs-t CSV and PNG diagnostics under `output/diagnostics/loss_vs_t`
- checkpoint payloads containing:
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

### Active default model settings in `Config()`

```python
motion_dim = 271
num_joints = 22
joint_dim = 3
max_motion_length = 200
fps = 20

encoder.hidden_size = 256
encoder.intermediate_size = 512
encoder.num_hidden_layers = 4
encoder.num_attention_heads = 8
encoder.per_joint_output_dim = 64

predictor.hidden_size = 96
predictor.intermediate_size = 384
predictor.num_hidden_layers = 3
predictor.num_attention_heads = 4
```

### Active default training settings

```python
batch_size = 200
learning_rate = 1e-4
weight_decay = 1e-5
gradient_clip = 1.0
ema_decay = 0.999
horizon = 40
cfg_dropout = 0.0
t_sampling_mode = "power"
t_sampling_power = 2.0
use_consistency_loss = True
consistency_loss_weight = 0.2
num_inference_steps = 20
val_interval = 5
val_batches = 20
checkpoint_interval = 50
```

### Default curriculum

```python
[
    {"horizon": 10, "epochs": 100},
    {"horizon": 20, "epochs": 200},
    {"horizon": 40, "epochs": 400},
]
```

### Config fields worth treating cautiously

Some config fields exist for planned, partially implemented, or inference-only paths:
- rollout schedule fields are present but not active in the current training loss path
- `guidance_scale` exists in config and generator signatures but is not used in sampling yet
- degenerate-pose guard fields are present but not central to the current default path

Check the source before depending on these fields for new work.

## Tests and Fixtures

Current notable tests in `tests/`:
- `test_pipeline_e2e.py`: end-to-end smoke test with a tiny HumanML3D fixture dataset
- `test_human_motion_generator.py`: generator shape, cold-start, and NaN checks
- `test_checkpoint_save_and_load.py`: checkpoint serialization and restore
- `test_loss_vs_t_diagnostics.py`: artifact generation for flow-loss diagnostics
- `test_root_motion_conversions.py`: root velocity / position conversion helpers
- `test_fk_consistent_rollout.py`: FK-consistent frame conversion checks

Fixture dataset:
- `tests/dataset/humanml3d-subset-mini`

## Known Stale Names and Comments

A few names and comments still reflect older iterations of the codebase:
- the top-level docstring in `src/models.py` still mentions older names such as `AutoregressiveContextEncoder`
- `TemporalCacheState` suggests incremental caching, but the current encoder still recomputes over the full frame buffer
- some older docs and notebooks still talk about a GRU-based encoder or older reduced-state sizes

Those are historical leftovers, not accurate descriptions of the active runtime path.
