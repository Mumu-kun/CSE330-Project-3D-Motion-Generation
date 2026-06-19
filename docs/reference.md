# Project Reference: Text-to-Motion Generation

This document describes the implementation that currently lives in `src/`. When docs disagree, prefer the source code.

## Overview

| Property | Current implementation |
| :--- | :--- |
| Task | Text-conditioned autoregressive 3D human motion generation |
| Dataset format | HumanML3D-style 22-joint sequences with 271D per-frame features |
| Context encoder | `MotionHistoryEncoder` using causal temporal self-attention with RoPE |
| Pretraining predictor | `JepaPredictor` (predicts future encoder representations) |
| Finetuning predictor | `LatentDecoder` (reconstructs 68D latent targets) |
| Predictor target space | 68D reduced motion state |
| Inference state | Absolute positions plus derived 271D feature history |
| Pretraining entrypoint | `train_pretrain()` in `src/utils/models/pretrain_trainer.py` |
| Finetuning entrypoint | `train_finetune()` in `src/utils/models/finetune_trainer.py` |

## Core Files

| File | Purpose |
| :--- | :--- |
| `src/config.py` | Project configuration dataclasses and active defaults |
| `src/utils/models/motion_history_encoder.py` | Temporal encoder with `JepaPredictor`, `LinearProbe`, `LatentDecoder` |
| `src/utils/models/flow_matching_predictor.py` | Spatial transformer with `FlowMatchingPredictor`, `LatentDecoder` |
| `src/utils/models/pretrain_trainer.py` | JEPA pretraining trainer |
| `src/utils/models/finetune_trainer.py` | Decoder-only finetuning trainer |
| `src/utils/models/human_motion_generator.py` | Inference wrapper with `integrate_flow_ode()` |
| `src/utils/motion_utils.py` | 271D feature conversions, IK/FK helpers |
| `src/utils/dataset.py` | HumanML3D dataset loader |

## Data Representations

### 271D frame format

| Slice | Size | Description |
| :--- | :--- | :--- |
| [0:3] | 3 | Root height Y, Root velocity X, Root velocity Z |
| [3:69] | 66 | 22 joints × 3D RIC positions |
| [69:201] | 132 | 22 joints × 6D rotations |
| [201:267] | 66 | 22 joints × 3D local velocities |
| [267:271] | 4 | Foot contact flags |

### 68D reduced predictor state

| Slice | Size | Description |
| :--- | :--- | :--- |
| [0:1] | 1 | Root height Y |
| [1:2] | 1 | Root velocity X |
| [2:3] | 1 | Root velocity Z |
| [3:5] | 2 | sin(dyaw), cos(dyaw) |
| [5:68] | 63 | 21 non-root joints × 3D RIC positions |

### Skeleton metadata

| Property | Value |
| :--- | :--- |
| Joint count | 22 |
| FPS | 20 |
| Kinematic chains | Left leg: [0,2,5,8,11], Right leg: [0,1,4,7,10], Spine: [0,3,6,9,12,15], Right arm: [9,14,17,19,21], Left arm: [9,13,16,18,20] |
| Facing-direction joints | [2, 1, 17, 16] |
| Left foot joints | [7, 10] |
| Right foot joints | [8, 11] |

## Models

### `MotionHistoryEncoder`

Source: `src/utils/models/motion_history_encoder.py`

Purpose:
- Encodes variable-length 271D motion sequences
- Causal temporal attention with rotary position encoding
- Returns per-timestep hidden states or all-layer outputs

Config:

```python
MotionHistoryEncoderConfig(
    frame_feature_dim=271,
    text_embedding_dim=512,
    hidden_size=512,               # Default
    intermediate_size=4 * 512,
    num_hidden_layers=4,             # Default
    num_attention_heads=16,
    hidden_act="silu",
    layer_norm_eps=1e-5,
    attention_bias=True,
    attention_dropout=0.1,
    mlp_bias=True,
    dropout=0.1,
    joint_count=22,
    num_registers=2,
)
```

Interface:

```python
forward(
    motion_seq: Tensor[B, T, 271],
    text_emb: Tensor[B, 512],
    mask: Optional[Tensor[B, T]] = None,
    return_layer_outputs: bool = False,
    is_causal: bool = False,
) -> Tensor[B, T, hidden_size] | Tensor[B, T, L, hidden_size]

step(
    x_t: Tensor[B, 271],
    text_emb: Tensor[B, 512],
    frame_buffer: Optional[Tensor[B, T, 271]],
    cache_state: Optional[TemporalCacheState] = None,
) -> Tuple[Tensor[B, hidden_size], Tensor[B, T+1, 271], TemporalCacheState]
```

### `FlowMatchingPredictor`

Source: `src/utils/models/flow_matching_predictor.py`

Purpose:
- Predicts 68D flow for denoising diffusion
- Cross-attention to encoder context features
- AdaLN conditioning on text + time

Config:

```python
FlowMatchingPredictorConfig(
    hidden_size=256,
    intermediate_size=768,
    num_hidden_layers=4,
    num_attention_heads=8,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    attention_bias=True,
    attention_dropout=0.1,
    mlp_bias=True,
)
```

Interface:

```python
FlowMatchingPredictor(
    config: Config,
    **kwargs,
)

forward(
    noisy_states: Tensor[B, 68],
    timesteps: Tensor[B] | Tensor[B, 1],
    track_features: Tensor[B, 22, H_enc],
    text_embedding: Tensor[B, 512],
    current_frame_features: Optional[Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[Tensor[B, 68], Optional[List[Tensor]]]
```

### `JepaPredictor` (pretraining)

Source: `src/utils/models/motion_history_encoder.py`

Purpose:
- Predicts future encoder representations during pretraining
- Takes masked context and predicts target context

Interface:

```python
forward(motion_history_emb: Tensor[B, T, L, H]) -> Tensor[B, T, L, H]
```

### `LinearProbe`

Source: `src/utils/models/motion_history_encoder.py`

Purpose:
- Contrastive probe of encoder representations
- Maps hidden states to text embedding space

### `LatentDecoder`

Source: `src/utils/models/flow_matching_predictor.py`

Purpose:
- Decodes encoder latents to 68D predictions
- Used for auxiliary reconstruction in pretraining

## Pretraining Architecture

### `PretrainTrainer`

Entrypoint: `train_pretrain(config, wandb_project=None, max_epochs=None)`

Returns: `(ema_encoder, None, checkpoint_path)`

Loss components during pretraining:

| Loss | Weight | Description |
| :--- | :--- | :--- |
| Mask loss | 1.0 | L1 on masked frame positions |
| Context loss | `jepa_ctx_weight` (0.2) | Weighted L1 on unmasked positions |
| Probe loss | 1.0 | Cosine contrast with text embedding |
| Decoder loss | 1.0 | Latent → positions → 271D reconstruction |

### Training step

1. Load motion `(B, T, 271)`, joints `(B, T, 22, 3)`, text `(B, 1, 512)`
2. Normalize motion features
3. Build random span mask (~25% of frames)
4. Forward `encoder` (masked) and `ema_encoder` (target, no grad) with `return_layer_outputs=True`
5. `jepa_predictor(masked_context) → predicted_context`
6. Compute JEPA loss with distance-weighted context terms
7. Compute probe loss on target context
8. Compute decoder loss via `LatentDecoder`
9. Accumulate gradients and update
10. Update EMA and schedulers

### Curriculum

```python
curriculum = [
    {"horizon": 5,  "epochs": 100},
    {"horizon": 10, "epochs": 200},
    {"horizon": 20, "epochs": 300},
    {"horizon": 40, "epochs": 1000},
]
```

## Finetuning Architecture

### `FinetuneTrainer`

Entrypoint: `train_finetune(config, pretrained_checkpoint_path, train_loader=None, val_loader=None, normalizer=None, wandb_project=None, max_epochs=None)`

Returns: `(ema_encoder, ema_decoder, checkpoint_path)`

Loading rules:

1. Prefer `encoder_ema` from the checkpoint for the frozen encoder copy.
2. Fall back to `encoder` if `encoder_ema` is missing.
3. Freeze both encoder copies (the frozen encoder and its EMA copy).
4. Train only the new `LatentDecoder`.

Loss components:

| Loss | Weight | Description |
| :--- | :--- | :--- |
| 68D reconstruction | 1.0 | MSE between decoded and target 68D latent state |
| Velocity loss | 0.5 | Smooth L1 on reconstructed 271D local velocity slice |
| Foot contact loss | 0.5 | Auxiliary penalty on contact-miss foot velocity |

### Training step

1. Load and normalize motion and joints.
2. Encode motion with the frozen pretrained encoder.
3. Decode `context[:, 1:, -1, :]` with a fresh `LatentDecoder`.
4. Reconstruct the target 68D state and 271D features from the decoded latents.
5. Backprop only through the decoder and update its EMA copy.

## Inference Architecture

### `HumanMotionGenerator`

Source: `src/utils/models/human_motion_generator.py`

Key methods:

```python
generate_sequence(
    text: str | List[str] | Tensor[B, 1, 512],
    num_frames: int = 200,
    num_steps: int = 10,
    horizon: Optional[int] = None,
    input_positions: Optional[Tensor] = None,
    guidance_scale: float = 1.0,
    guidance_drop_text: bool = True,
    guidance_drop_context: bool = False,
    dataset_type: str = "t2m",
    use_fk: bool = True,
) -> Tuple[
    Tensor[B, T, 22, 3],   # position_history
    Tensor[B, T, 271],      # feature_history
    Tensor[B, T, 22, 3],    # relative_shift_history
]
```

Generation loop:

1. Initialize position_history from `input_positions` or zero pose
2. Derive feature_history via `sequence_joints_to_features()`
3. For each frame:
   - Encode context via `encoder.step()`
   - Integrate ODE via `integrate_flow_ode()` with Heun solver
   - Convert output via `x68_to_positions()`
   - Re-derive 271D features via `generated_positions_to_x271()`
   - Append to histories

### ODE Integration

Uses end-biased schedule `t = 1 - (1-s)^p` where `p = inference_t_schedule_power`.

Heun (2nd-order) solver:
```
k1 = predict_velocity(x_t, t_start)
x_euler = x_t + dt * k1
k2 = predict_velocity(x_euler, t_end)
x_t = x_t + 0.5 * dt * (k1 + k2)
```

CFG (Classifier-Free Guidance) when `guidance_scale != 1.0`:
```
velocity = uncond_velocity + guidance_scale * (cond_velocity - uncond_velocity)
```

## Configuration

Active defaults from `Config()`:

```python
# Model defaults
encoder.hidden_size = 512
encoder.intermediate_size = 2048
encoder.num_hidden_layers = 4
encoder.num_attention_heads = 16
encoder.num_registers = 2

predictor.hidden_size = 256
predictor.intermediate_size = 768
predictor.num_hidden_layers = 4
predictor.num_attention_heads = 8

# Training
batch_size = 200
effective_batch_size = 400
learning_rate = 0.5e-4
weight_decay = 1e-5
gradient_clip = 30.0
ema_decay = 0.999
lr_warmup_epochs = 5

# Curriculum
curriculum = [
    {"horizon": 5, "epochs": 100},
    {"horizon": 10, "epochs": 200},
    {"horizon": 20, "epochs": 300},
    {"horizon": 40, "epochs": 1000},
]

# JEPA
jepa_ctx_weight = 0.2

# CFG
cfg_dropout = 0.1

# Timestep sampling
t_sampling_mode = "power"
t_sampling_power = 3.0

# Inference
num_inference_steps = 20
inference_t_schedule_power = 3.0
guidance_scale = 1.0

# Validation
val_interval = 5
val_batches = 20
val_use_ema = True
save_best_val = True

# Checkpointing
checkpoint_interval = 50
```

## Checkpoint Format

| Key | Content |
| :--- | :--- |
| `encoder` | Live encoder weights (frozen in finetuning) |
| `encoder_ema` | EMA encoder weights (preferred for loading in finetuning) |
| `predictor` | Live predictor weights (finetuning) |
| `predictor_ema` | EMA predictor weights (finetuning) |
| `jepa_predictor` | Live JEPA predictor weights (pretraining) |
| `decoder` | LatentDecoder weights |
| `decoder_ema` | EMA LatentDecoder weights |
| `linear_probe` | LinearProbe weights |
| `optimizer` | AdamW optimizer state |
| `aux_optimizer` | Auxiliary optimizer (pretraining) |
| `scaler` | GradScaler state |
| `lr_scheduler` | OneCycleLR state |
| `metadata` | Session info (session_id, resume_id) |
| `config` | Serialized Config (if saved) |

## Source Files

| File | Models |
| :--- | :--- |
| `src/utils/models/__init__.py` | `AdaLN`, `GatedMLP`, `TemporalRoPEAttention`, `TemporalCacheState`, `TemporalLayerCache`, `init_weights` |
| `src/utils/models/motion_history_encoder.py` | `MotionHistoryEncoder`, `JepaPredictor`, `LinearProbe`, `JepaMLPBlock` |
| `src/utils/models/flow_matching_predictor.py` | `FlowMatchingPredictor`, `SinusoidalEmbedder`, `LatentDecoder` |
| `src/utils/models/pretrain_trainer.py` | `PretrainTrainer`, `PretrainEngine`, `PretrainState`, `EMAModel`, `ProgressScheduler` |
| `src/utils/models/finetune_trainer.py` | `FinetuneTrainer`, `train_finetune`, `EMAModel` |
| `src/utils/models/human_motion_generator.py` | `HumanMotionGenerator`, `integrate_flow_ode` |