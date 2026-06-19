# High-Level Architecture Design: Human Motion Generation

This document summarizes the architecture that is actually implemented in `src/utils/models/`. The project is a text-conditioned autoregressive motion generator built around a causal temporal history encoder, a flow-matching spatial transformer, and a frame-by-frame conversion loop between absolute positions and compact motion features.

## 1. System Summary

At a high level, each generated frame is produced in four stages:

1. Encode recent 271D motion history with text conditioning.
2. Assemble predictor inputs: track features from encoder, noisy reduced-state, timesteps, text embedding.
3. Predict a denoised next-frame state in a reduced 68D representation.
4. Convert that reduced state into absolute global joint positions.
5. Convert the new positions back into a 271D frame and append them to history.

The recurrent state is therefore not a latent-only state. It is a pair of concrete, interpretable histories:
- absolute joint positions
- derived 271D feature frames

## 2. Main Components

### A. Motion History Encoder

Implemented in `src/utils/models/motion_history_encoder.py` as `MotionHistoryEncoder`.

Role:
- reads a variable-length motion window `(B, T, 271)`
- conditions each timestep on CLIP text `(B, 512)` via AdaLN modulation
- emits per-timestep features `(B, T, hidden_size)` or all-layer outputs when `return_all=True`

Data path:

```text
motion history (B, T, 271)
text embedding (B, 512)
    -> frame_projection(motion) -> (B, T, H)
    -> [register tokens prepended]
    -> AdaLN modulation per layer (text -> shift/scale)
    -> causal temporal self-attention with RoPE (non-registers)
    -> gated MLP blocks
    -> final norm
    -> (B, T, H) features
```

Tensor flow details:
- `frame_projection` maps 271D features to hidden dimension independently per timestep.
- `step(...)` appends the newest 271D frame to the explicit `frame_buffer` and recomputes the encoder over the buffered sequence.
- `register_tokens` (default: 2) are prepended to the sequence; they do not receive RoPE.
- `mask_token` replaces masked positions during pretraining.

Why it exists:
- It gives the predictor a temporally informed summary without forcing the predictor itself to model full spatiotemporal attention.
- The encoder is causal, so each context only depends on current and past frames.

Important implementation note:
- `step()` is used for autoregressive generation but recomputes over the full `frame_buffer`.
- `TemporalCacheState` exists as an interface, but KV-cache acceleration is not yet implemented.

### B. Flow Matching Predictor

Implemented in `src/utils/models/flow_matching_predictor.py` as `FlowMatchingPredictor`.

Role:
- predicts flow in a reduced next-frame state space (68D)
- reasons jointly over joints using cross-attention to encoder context

Forward path:

```text
noisy_states (B, 68)
track_features (B, 22, H_enc)  <- From MotionHistoryEncoder
timesteps (B,)
text_embedding (B, 512)
    -> text_proj(text) + time_embedder(t) -> adaln_cond (B, 2*H_pred)
    -> latent_in_proj(track_features) -> hidden (B, 68, H_pred)
    -> [Transformer layers with AdaLN]
    -> latent_out_proj -> (B, 68, H_enc)
    -> output_adaln(adaln_cond) -> shift, scale
    -> output_norm(shift + scale * hidden) -> flow (B, 68)
```

Conditioning path:
- Time t → `SinusoidalEmbedder` → MLP → hidden_size vector
- Text → `Linear(512 → hidden_size)` → hidden_size vector
- Concatenated: `[text_cond, time_cond]` → used for AdaLN modulation

### C. Pretraining Trainer

Implemented in `src/utils/models/pretrain_trainer.py` as `PretrainTrainer`.

The pretraining uses a JEPA (Joint-Embedding Predictive Architecture) approach with multiple loss components:

1. **JEPA Loss**: Masked autoencoder where `JepaPredictor` predicts target encoder outputs
2. **Linear Probe Loss**: Contrastive loss aligning encoder output with text embedding
3. **Decoder Loss**: Reconstruction loss for 68D → positions → 271D

During pretraining, no `FlowMatchingPredictor` is used. Instead:
- `JepaPredictor` learns to predict future encoder representations from past context
- `LatentDecoder` provides auxiliary reconstruction signal
- `LinearProbe` provides text-alignment signal

### D. Finetuning Trainer

Implemented in `src/utils/models/finetune_trainer.py` as `FinetuneTrainer`.

During finetuning:
- Loads pretrained `MotionHistoryEncoder` and its EMA copy from checkpoint
- Freezes the encoder pair and trains only a fresh `LatentDecoder`
- Uses the pretraining decoder reconstruction loss on the 68D latent target

### E. Human Motion Generator

Implemented in `src/utils/models/human_motion_generator.py` as `HumanMotionGenerator`.

Role:
- orchestrates encoder, predictor, and conversion utilities for autoregressive generation

Important implementation choices:
- keeps absolute positions as the primary autoregressive state
- 271D features are re-derived every step from positions (not persistent latent)
- history length controlled by `horizon` argument or `config.horizon`
- uses `integrate_flow_ode()` with Heun solver for inference

## 3. Current Data Representations

### 271D frame representation

| Slice | Size | Description |
|-------|------|-------------|
| [0:3] | 3 | Root height Y, Root velocity X, Root velocity Z |
| [3:69] | 66 | 22 joints × 3D RIC positions |
| [69:201] | 132 | 22 joints × 6D rotations |
| [201:267] | 66 | 22 joints × 3D local velocities |
| [267:271] | 4 | Foot contact flags |

### 68D reduced predictor state

| Slice | Size | Description |
|-------|------|-------------|
| [0:1] | 1 | Root height Y |
| [1:2] | 1 | Root velocity X |
| [2:3] | 1 | Root velocity Z |
| [3:5] | 2 | sin(dyaw), cos(dyaw) |
| [5:68] | 63 | 21 non-root joints × 3D RIC positions |

## 4. Pretraining Architecture

### Training step

For each batch:

1. Load raw motion `(B, T, 271)` and joints `(B, T, 22, 3)`
2. Normalize 271D motion via `FeatureNormalizer`
3. Build random span mask (~25% of frames)
4. Forward pass:
   - `encoder(motion, zeros_like(text), mask=mask, return_all=True)` → masked_context
   - `ema_encoder(motion, zeros_like(text), mask=None, return_all=True)` → target_context (no grad)
5. `jepa_predictor(masked_context)` → predicted_context
6. Compute combined loss:
   - Mask loss on positions corresponding to mask
   - Context loss on unmasked positions with distance weighting
   - Probe loss on target context vs text embedding
   - Decoder loss via `LatentDecoder(target_context[:, 1:])`
7. Update with gradient accumulation

### Curriculum Learning

Horizon increases during training:

```python
curriculum = [
    {"horizon": 5,  "epochs": 100},
    {"horizon": 10, "epochs": 200},
    {"horizon": 20, "epochs": 300},
    {"horizon": 40, "epochs": 1000},
]
```

### Checkpoint Artifacts

| File | Content |
|------|---------|
| `pretrain_latest.pt` | Latest model state (every 50 epochs) |
| `pretrain_best_val.pt` | Best validation loss |
| `pretrain_best_eval.pt` | Best decoder loss |

## 5. Inference Architecture

### Generation loop (`generate_sequence`)

For each output frame:

1. Extract current frame from position/feature history
2. Encode context: `encoder.step(current_frame, text_emb, frame_buffer)` → track_features
3. Initialize noise: `x_t ~ N(0, I)` in 68D
4. Integrate ODE via `integrate_flow_ode()`:
   - Uses end-biased schedule: `t = 1 - (1-s)^p`
   - Heun solver (2nd order)
   - Optional CFG via `unconditional_text_embedding` / `unconditional_track_features`
5. Convert: `x68_to_positions(x_t, prev_root_pos, prev_root_rot_6d)` → new_positions
6. Re-derive features: positions → 271D via `generated_positions_to_x271()`
7. Append to history buffers

### Masked sequence generation (`generate_sequence_masked`)

Alternative path that:
1. Appends mask tokens for all future frames at once
2. Single encoder forward pass with full context
3. Autoregressive prediction over masked positions

## 6. Why This Architecture Makes Sense

The design separates concerns cleanly:

- **Temporal encoder**: Handles causal accumulation over motion history
- **Flow predictor**: Solves conditioned next-frame denoising with strong context
- **Conversion utilities**: Keep model grounded in explicit geometry

This decoupling is efficient because the predictor doesn't need to model spatiotemporal attention—it only denoises within the context already shaped by the encoder.

## 7. Source of Truth

When docs and comments disagree, prefer these files:

- `src/utils/models/motion_history_encoder.py`
- `src/utils/models/flow_matching_predictor.py`
- `src/utils/models/pretrain_trainer.py`
- `src/utils/motion_utils.py`
- `src/config.py`