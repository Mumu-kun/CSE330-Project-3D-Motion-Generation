# High-Level Design: Human Motion Generation

**Last Updated**: 2026-06-25

---

## System Overview

Text-conditioned autoregressive 3D human motion generator. Given a text description (e.g., "a person walks forward and then turns left"), the system produces a sequence of 3D joint positions over time.

The system operates in three stages:

1. **Pretraining** — Learn general motion representations via masked prediction with a **non-causal** encoder
2. **Finetuning** — Train decoder on reconstruction with the encoder operating in **causal** mode (encoder frozen)
3. **Inference** — Generate new motion autoregressively from text input

---

## Generation Loop

Each output frame is produced in five steps:

1. **Encode** — Encode recent 271D motion history with CLIP text conditioning (causal attention)
2. **Assemble** — Combine encoder track features, noisy 68D state, timestep, and text embedding
3. **Denoise** — Predict the next-frame 68D state via ODE integration (flow matching)
4. **Convert** — Map 68D reduced state → 22 joint global positions (3D)
5. **Re-derive** — Convert positions back to 271D features, append to history

The autoregressive state is a pair of histories:
- Absolute joint positions `(T, 22, 3)`
- Derived 271D feature frames `(T, 271)`

---

## Main Components

### Motion History Encoder (MHE)

- Reads a variable-length motion window `(B, T, 271)`
- Conditions each timestep on CLIP text `(B, 512)` via AdaLN modulation
- Emits per-timestep features `(B, T, hidden_size)` or all-layer outputs `(B, T, L, hidden_size)`
- Temporal self-attention with RoPE
- 2 register tokens prepended to the sequence (excluded from RoPE)
- `mask_token` replaces masked positions during pretraining

**Causal mode (configurable):**
- **Non-causal** (pretraining): Bidirectional attention — sees past and future context
- **Causal** (finetuning + inference): Autoregressive attention — only sees current and past frames

The same encoder supports both modes via an `is_causal` flag. Non-causal pretraining produces stronger representations by leveraging full-sequence context. Causal mode ensures the autoregressive constraint during generation.

### Flow Matching Predictor

- Predicts denoising flow in the encoder's hidden space `(B, N, H_enc)`
- Cross-attention over encoder track features
- Conditioned on text (linear projection) and timestep (sinusoidal embedding) via AdaLN
- **Inputs**: noisy states `[B, N, H_enc]`, timesteps `[B]`, track features `[B, N, H_enc]`, text `[B, D]`
- **Outputs**: flow prediction `[B, N, H_enc]`
- Used during inference (not yet trained during finetuning)

### JEPA Predictor (Pretraining Only)

- Predicts EMA encoder representations from masked encoder inputs
- V-JEPA 2 style: injects random noise to model multiple plausible futures
- Multi-layer prediction: all encoder layer outputs `(B, T, L, H)` are predicted simultaneously
- Operates on non-causal encoder outputs (full sequence context available)

### Latent Decoder

- Maps encoder hidden states to 68D reduced motion representation
- Separate prediction heads for root height, root velocity, delta yaw, and joint velocities
- Yaw output is L2-normalized to unit circle
- Used in pretraining (auxiliary reconstruction) and finetuning (primary trainable)

### Linear Probe

- Contrastive probe mapping encoder representations to CLIP text embedding space
- Temporal pooling → LayerNorm → Linear → L2-normalize
- Encourages text-alignment of encoder outputs during pretraining

---

## Data Representations

### 271D Frame Format

| Slice | Size | Description |
|-------|------|-------------|
| [0:3] | 3 | Root height Y, Root velocity X, Root velocity Z |
| [3:69] | 66 | 22 joints × 3D RIC positions |
| [69:201] | 132 | 22 joints × 6D rotations |
| [201:267] | 66 | 22 joints × 3D local velocities |
| [267:271] | 4 | Foot contact flags |

### 68D Reduced Predictor State

| Slice | Size | Description |
|-------|------|-------------|
| [0:1] | 1 | Root height Y |
| [1:2] | 1 | Root velocity X |
| [2:3] | 1 | Root velocity Z |
| [3:5] | 2 | sin(delta_yaw), cos(delta_yaw) |
| [5:68] | 63 | 21 non-root joints × 3D RIC velocities |

### Skeleton

- 22 joints, 20 FPS
- Kinematic chains: legs, spine, arms
- Foot contact joints used for training supervision

---

## Training Pipeline

### Stage 1: Masked JEPA Pretraining

**Objective**: Learn temporal representations by predicting masked encoder outputs.

**Encoder mode: Non-causal.** Bidirectional attention lets the encoder build rich representations using full-sequence context.

```
Encoder (masked input, non-causal)    → masked_context
EMA Encoder (full input, non-causal)  → target_context (no gradient)
JepaPredictor(masked_context) → predicted_context
Loss: L1(predicted, target) on masked + distance-weighted unmasked positions
```

Additional losses:
- **Linear probe**: Contrastive alignment with CLIP text embeddings
- **Decoder**: 68D → positions → 271D reconstruction

Curriculum: sequence length increases 5 → 10 → 20 → 40 over training.

### Stage 2: Decoder Finetuning

**Objective**: Train `LatentDecoder` on reconstruction. Encoder is frozen and operates in causal mode.

**Encoder mode: Causal.** Matches inference conditions. Both encoder copies (live + EMA) are frozen.

```
EMA Encoder (frozen, causal)  → target_context[:, 1:, -1, :] → latent
LatentDecoder (trainable)     → 68D → positions → 271D
Loss: MSE + SmoothL1 on motion components
```

Only the decoder and its EMA copy are updated. The `FlowMatchingPredictor` is **not** trained at this stage — it will be added in a future phase.

### Inference (Future)

**Objective**: Generate motion from text using the flow matching predictor.

```
For each frame:
1. encoder.step(current_frame, text_emb, history) → track_features  [causal]
2. Initialize x_t ~ N(0, I) in H_enc
3. FlowMatchingPredictor(x_t, t, track_features, text) → flow  [B, N, H_enc]
4. Integrate ODE (Heun solver, 20 steps, end-biased schedule)
5. Convert output → 68D → positions (22, 3) → 271D → append to history
```

Optional classifier-free guidance: `v = uncond + scale × (cond - uncond)`.

---

## Architecture Rationale

**Why non-causal pretraining then causal finetuning?**
During pretraining, bidirectional attention lets the encoder build rich representations using full-sequence context — similar to how BERT benefits from bidirectional context. During finetuning and inference, the same encoder operates causally to respect the autoregressive constraint. The frozen encoder's representations are already strong; the decoder simply learns to reconstruct from them.

**Why 68D intermediate representation?**
The 271D format contains redundant information (rotations, velocities, contacts). The 68D reduced state captures the essential degrees of freedom for next-frame prediction, making the denoising task easier.

**Why JEPA pretraining?**
Masked prediction in representation space (rather than raw input space) forces the model to learn abstract temporal features that transfer well to downstream generation. The V-JEPA 2 noise injection handles the inherent stochasticity of motion.

---

## Configuration

### Model

| Component | Parameter | Default |
|-----------|-----------|---------|
| Encoder | hidden_size | 512 |
| Encoder | intermediate_size | 1024 |
| Encoder | num_hidden_layers | 4 |
| Encoder | num_attention_heads | 16 |
| Encoder | num_registers | 2 |
| Predictor | hidden_size | 256 |
| Predictor | intermediate_size | 768 |
| Predictor | num_hidden_layers | 4 |
| Predictor | num_attention_heads | 8 |
| JEPA | hidden_size | 192 |
| JEPA | num_hidden_layers | 2 |
| Decoder | hidden_size | 512 |
| Decoder | num_layers | 4 |

### Training

| Parameter | Value |
|-----------|-------|
| learning_rate | 5e-4 |
| batch_size | 200 |
| effective_batch_size | 400 |
| weight_decay | 1e-5 |
| gradient_clip | 30.0 |
| ema_decay | 0.999 |
| lr_warmup_epochs | 5 |
| lr_schedule | OneCycleLR (cosine) |

### Inference

| Parameter | Value |
|-----------|-------|
| num_inference_steps | 20 |
| inference_t_schedule_power | 3.0 |
| guidance_scale | 1.0 |

---

## Checkpoints

### Pretraining

| File | Content |
|------|---------|
| `pretrain_latest_{step}.pt` | Latest model state |
| `pretrain_best_val.pt` | Best validation loss |
| `pretrain_best_eval.pt` | Best decoder loss |

### Finetuning

| File | Content |
|------|---------|
| `finetune_latest_{step}.pt` | Latest model state |
| `finetune_best_val.pt` | Best validation loss |

### Checkpoint Keys

`encoder`, `encoder_ema`, `jepa_predictor`, `decoder`, `decoder_ema`, `linear_probe`, `optimizer`, `scaler`, `lr_scheduler`, `metadata`, `config`
