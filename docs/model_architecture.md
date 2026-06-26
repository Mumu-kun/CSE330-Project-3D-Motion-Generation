# Model Architecture

**Last Updated**: 2026-06-25

---

## Overview

Three core components:

1. **Motion History Encoder (MHE)** — Temporal transformer with configurable causal/non-causal attention
2. **Flow Matching Predictor** — Spatial transformer predicting denoising flow in encoder hidden space
3. **JEPA Predictor** — Masked-latent prediction for pretraining (V-JEPA 2 style)

Supporting modules: `LatentDecoder` (68D prediction head), `LinearProbe` (text alignment).

---

## Motion History Encoder (MHE)

**Class**: `MotionHistoryEncoder`

Encodes variable-length motion windows with CLIP text conditioning via temporal self-attention.

```
motion_seq (B, T, 271) → frame_projection (271→H) → [register tokens prepended]
→ N × EncoderLayer (AdaLN + RoPE self-attn + gated MLP)
→ final_norm → (B, T, H)
```

### Causal Mode (Configurable)

The encoder supports both causal and non-causal attention via an `is_causal` flag:

| Mode | When | Behavior |
|------|------|----------|
| **Non-causal** | Pretraining | Bidirectional attention — sees full sequence (past + future) |
| **Causal** | Finetuning + Inference | Autoregressive attention — only sees current + past frames |

The same weights and architecture are used in both modes.

### EncoderLayer

Each layer applies:
1. AdaLN-modulated self-attention with RoPE (causal or non-causal)
2. AdaLN-modulated gated MLP
3. Gated residual connections after both sub-layers

Text conditioning: CLIP embedding → AdaLN shift/scale for both attention and MLP.

Key properties:
- 2 register tokens prepended (excluded from RoPE)
- `mask_token` replaces masked positions during pretraining
- `step()` method for autoregressive generation (causal, recomputes over full buffer)
- `return_layer_outputs=True` → `(B, T, L, H)` for JEPA predictor

---

## Flow Matching Predictor

**Class**: `FlowMatchingPredictor`

Predicts denoising flow in the encoder's hidden space. Used during inference (not pretraining or current finetuning).

```
noisy_states (B, N, H_enc) → latent_in_proj (H_enc→H_pred)
→ N × PredictorLayer (AdaLN + cross-attn to track_features + gated MLP)
→ latent_out_proj (H_pred→H_enc) → output_adaln(shift, scale) → flow (B, N, H_enc)
```

### Inputs

| Input | Shape | Description |
|-------|-------|-------------|
| noisy_states | `[B, N, H_enc]` | Noisy reduced states in encoder hidden dim |
| timesteps | `[B]` or `[B, 1]` | Denoising timestep |
| track_features | `[B, N, H_enc]` | Encoder context (cross-attention keys/values) |
| text_embedding | `[B, D]` | CLIP text embedding |

### Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| flow_prediction | `[B, N, H_enc]` | Predicted flow in encoder hidden dim |

### Conditioning

| Input | Transformation | Output |
|-------|----------------|--------|
| text_embedding (B, 512) | Linear(512 → H_pred) | (B, H_pred) |
| timesteps (B,) | SinusoidalEmbedder → MLP | (B, H_pred) |
| Both | `cat([text, time])` | (B, 2×H_pred) → AdaLN |

### Cross-Attention

`PredictorRopeCrossAttention`: query = hidden states (with RoPE), key/value = encoder track features. No causal mask — spatial attention over joints.

### PredictorLayer

Each layer:
1. AdaLN → Cross-attention → gated residual
2. AdaLN → Gated MLP → gated residual

---

## JEPA Predictor (Pretraining Only)

**Class**: `JepaPredictor`

Predicts multi-layer encoder representations from masked inputs. Based on V-JEPA 2.

```
masked_context (B, T, L, H) → reshape → input_mlp → concat with V-JEPA noise z
→ MLP blocks → output_mlp → per-layer output_proj → (B, T, L, H)
```

### V-JEPA 2 Noise Injection

Random noise vector `z ~ N(0, I)` is projected and concatenated with the masked context embedding. Different noise samples produce different valid motion continuations.

### Multi-Layer Prediction

Input and output are all encoder layer outputs stacked: `(B, T, num_layers, hidden_dim)`. Separate output projection per layer.

**Operates on non-causal encoder outputs** — full sequence context is available during pretraining.

---

## Latent Decoder

**Class**: `LatentDecoder`

Maps encoder hidden states to 68D reduced motion representation.

```
latent (B, H_enc) → down_proj → DecoderMLP × 4 → head_norm
→ root_head_y (1) + root_head_xz (2) + yaw_head (2, normalized) + ric_head (63)
→ cat → (B, 68)
```

### Output Heads

| Head | Size | Output |
|------|------|--------|
| root_head_y | 1 | Absolute root height |
| root_head_xz | 2 | Root velocity X, Z |
| yaw_head | 2 | (sin(δyaw), cos(δyaw)), L2-normalized |
| ric_head | 63 | 21 non-root joint RIC velocities |

Used in: pretraining (auxiliary reconstruction) and finetuning (primary trainable module).

---

## Linear Probe

**Class**: `LinearProbe`

Contrastive probe mapping encoder representations to CLIP text embedding space.

```
target_context → mean(dim=1) → LayerNorm → Linear(H→512) → L2-normalize
→ contrastive loss vs CLIP text embedding
```

Loss: `1 - cosine_similarity(probe_out, text_emb)`.

---

## Data Representations

### 271D Frame

| Slice | Size | Content |
|-------|------|---------|
| [0:3] | 3 | Root Y, Root vel X, Root vel Z |
| [3:69] | 66 | 22 joints × 3D RIC positions |
| [69:201] | 132 | 22 × 6D rotations |
| [201:267] | 66 | 22 × 3D local velocities |
| [267:271] | 4 | Foot contact flags |

### 68D Reduced State

| Slice | Size | Content |
|-------|------|---------|
| [0:1] | 1 | Root height Y |
| [1:3] | 2 | Root velocity X, Z |
| [3:5] | 2 | sin(δyaw), cos(δyaw) |
| [5:68] | 63 | 21 non-root joints × 3D RIC velocities |

### Conversion Pipeline

```
271D → denormalize → extract yaw delta, joint velocities → normalize → 68D
68D → denormalize → integrate yaw, build root, apply FK → positions (22, 3)
positions (22, 3) → compute IK rotations, RIC, local vel, foot contacts → 271D
```

---

## Configuration Defaults

### Encoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 512 | Hidden dimension |
| intermediate_size | 1024 | MLP intermediate dimension |
| num_hidden_layers | 4 | Transformer layers |
| num_attention_heads | 16 | Attention heads (head_dim=32) |
| num_registers | 2 | Register tokens prepended |

### Predictor

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 256 | Hidden dimension |
| intermediate_size | 768 | MLP intermediate dimension |
| num_hidden_layers | 4 | Transformer layers |
| num_attention_heads | 8 | Attention heads (head_dim=32) |

### JEPA

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 192 | Hidden dimension |
| intermediate_size | 512 | MLP intermediate dimension |
| num_hidden_layers | 2 | MLP blocks |

### Decoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 512 | Hidden dimension |
| num_layers | 4 | Residual MLP blocks |
