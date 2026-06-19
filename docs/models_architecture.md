# Model Architecture: Motion History Encoder and Flow Matching Predictor

This document describes the architecture of the three core model files in `src/utils/models/`:

- `motion_history_encoder.py` - Temporal encoder with causal attention
- `flow_matching_predictor.py` - Spatial transformer for flow prediction
- `pretrain_trainer.py` - JEPA-style pretraining loop

## Motion History Encoder (`motion_history_encoder.py`)

### Overview

The `MotionHistoryEncoder` is a causal temporal transformer that processes sequences of 271D motion features and produces contextual representations for each joint at each timestep.

### Architecture

```
Input: (B, T, 271) motion features + (B, 512) text embedding
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  frame_projection: Linear(271 → hidden_size)         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Register Tokens: (num_registers, hidden_size)       │
│  prepended to sequence [B, R+T, hidden_size]       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Transformer Layers (num_hidden_layers)              │
│  ┌─────────────────────────────────────────────────┐│
│  │ AdaLN + Causal RoPE Self-Attention             ││
│  │   - Query/KEY attend with causal mask            ││
│  │   - RoPE applied to non-register positions     ││
│  └─────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────┐│
│  │ AdaLN + Gated MLP                               ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  final_norm: LayerNorm                              │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output: (B, T, hidden_size) per-timestep features
```

### Key Components

#### `EncoderRoPEAttention`
- Inherits from `TemporalRoPEAttention`
- Projects queries, keys, values from hidden dimension
- Applies rotary position encoding to non-register positions
- Uses causal attention mask
- Register tokens do not receive RoPE (position embedding-free)

#### `EncoderLayer`
Two sub-layers per block:

1. **Self-Attention**: AdaLN-modulated attention over the full sequence
2. **MLP**: AdaLN-modulated gated MLP with configurable hidden act

#### `MotionHistoryEncoder`
- `frame_projection`: Maps 271D features to hidden dimension
- `register_tokens`: Learnable parameters (default: 2) prepended to input
- `mask_token`: Learnable parameter for masked position prediction
- `step()`: Autoregressive method that appends a new frame to frame_buffer and re-computes forward pass

### Interface

```python
def forward(
    motion_seq: Tensor[B, T, 271],
    text_emb: Tensor[B, 512],
    mask: Optional[Tensor[B, T]] = None,      # Replace masked positions with mask_token
    return_layer_outputs: bool = False,       # Return all layer outputs stacked
    is_causal: bool = False,                  # Apply causal mask to attention
) -> Tensor[B, T, hidden_size] | Tensor[B, T, L, hidden_size]

def step(
    x_t: Tensor[B, 271],           # New frame to append
    text_emb: Tensor[B, 512],
    frame_buffer: Optional[Tensor[B, T, 271]],  # History buffer
    cache_state: Optional[TemporalCacheState] = None,
) -> Tuple[
    Tensor[B, hidden_size],        # Context for last timestep
    Tensor[B, T+1, 271],           # Updated frame buffer
    TemporalCacheState,           # For potential KV caching (not yet implemented)
]
```

## Flow Matching Predictor (`flow_matching_predictor.py`)

### Overview

The `FlowMatchingPredictor` predicts motion flow in a reduced 68D state space using cross-attention over encoder context features. It implements a Denoising Diffusion Probabilistic Model (DDPM) / flow matching objective.

### Architecture

```
Input: (B, 68) noisy states + (B, 512) text + (B, 22, H_enc) track features
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Text Projection: Linear(512 → pred_hidden_size)    │
│  Time Embedding: SinusoidalEmbedder(68 → pred_hidden_size)│
│                                                    │
│  adaln_cond = text_cond + time_cond                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  latent_in_proj: Linear(H_enc → pred_hidden_size)     │
│                                                    │
│  ┌─────────────────────────────────────────────────┐│
│  │ Predictor Layer (num_hidden_layers)             ││
│  │ ── AdaLN → Cross-Attention (track_features)    ││
│  │ ── Residual connection with gate                ││
│  │ ── AdaLN → Gated MLP                            ││
│  │ ── Residual connection with gate                ││
│  └─────────────────────────────────────────────────┘│
│                                                    │
│  latent_out_proj: Linear(pred_hidden_size → H_enc)    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  output_adaln: Linear(H_enc*2 → H_enc*2) + SiLU       │
│  output_norm: LayerNorm(H_enc)                       │
│                                                    │
│  output = norm(shift + scale * hidden_states)        │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output: (B, 68) predicted flow
```

### Key Components

#### `SinusoidalEmbedder`
- Embeds scalar timesteps t ∈ [0,1] into hidden dimension
- Uses frequency-based embedding: cos(t) and sin(t) at different frequencies
- MLP projection to hidden size

#### `PredictorRopeCrossAttention`
- Cross-attention from hidden states to encoder track features
- Query has RoPE applied; keys/values come from cached encoder K/V
- No causal mask (spatial attention over joints)

#### `PredictorLayer`
Each layer applies:

1. **AdaLN-modulated cross-attention** to encoder context
2. **AdaLN-modulated gated MLP**

The AdaLN gate controls residual contribution after each operation.

### Interface

```python
def forward(
    noisy_states: Tensor[B, 68],       # Noisy reduced state
    timesteps: Tensor[B] | Tensor[B, 1],  # Denoising timestep
    track_features: Tensor[B, 22, H_enc],  # Encoder context
    text_embedding: Tensor[B, 512],       # Text conditioning
    current_frame_features: Optional[Tensor[B, ...]] = None,
    output_attentions: bool = False,
) -> Tuple[Tensor[B, 68], Optional[List[Tensor[B, 22, H_pred]]]
```

**Note**: The `current_frame_features` parameter is accepted but not used in the current implementation.

### LatentDecoder

A separate module for converting encoder latents to 68D predictions:

```python
def forward(latent: Tensor[B, H_enc]) -> Tensor[B, 68]

def decode(latent, prev_pos, prev_frame, normalizer) -> Tuple[
    Tensor[B, 271],    # Normalized 271D features
    Tensor[B, 22, 3],  # Joint positions
]
```

## Pretraining Trainer (`pretrain_trainer.py`)

### Overview

The `PretrainTrainer` implements JEPA (Joint-Embedding Predictive Architecture) pretraining with three concurrent loss objectives.

### Architecture

```
                     ┌─────────────────┐
                     │  Motion Sequence│
                     │  (B, T, 271)    │
                     └────────┬────────┘
                              │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Masked Input │    │  Target       │    │  Decoder      │
│  (~25% frames)│    │  (full seq)   │    │  Loss         │
└───────┬───────┘    └────────┬──────┘    └───────┬───────┘
        ▼                     ▼                   ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ MotionHistory │    │ EMA Encoder   │    │ Joint Positions │
│ Encoder       │    │ (no grad)     │    │ (ground truth)│
│ (return_all)  │    │ (return_all)  │    └───────┬───────┘
└───────┬───────┘    └────────┬──────┘            ▼
        ▼                     ▼          ┌───────────────┐
┌───────────────┐    ┌───────────────┐    │ LatentDecoder │
│ JepaPredictor │    │ (B, T, L, H)  │    │               │
└───┬───────────┘    └────────┬──────┘    └───────┬───────┘
    ▼                         ▼                   ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Predicted     │    │ Target        │    │ x68 Pred →    │
│ Context       │    │ Context       │    │ Positions →   │
│ (B, T, L, H)  │    │ (B, T, L, H)  │    │ 271D → Loss   │
└───────┬───────┘    └────────┬──────┘    └───────┬───────┘
        │                     │                   │
        └──────────┬──────────┴───────────────────┘
                   ▼
         ┌─────────────────┐
         │  Combined Loss  │
         │  - Mask Loss    │
         │  - Context Loss │
         │  - Probe Loss   │
         │  - Decoder Loss │
         └─────────────────┘
```

### Loss Components

#### JEPA Loss (Primary)

Uses masked autoencoder approach:

1. Generate random span masks covering ~25% of sequence
2. Forward pass with masked positions replaced by `mask_token`
3. Target encoder (EMA) processes full sequence without gradient
4. `JepaPredictor` learns to predict target encoder output for masked positions

**Loss formula**:
```
mask_loss = L1(predicted[mask], target[mask])
pos_weight = 1 / sqrt(min_distance_to_masked + 1)
context_loss = L1(predicted[unmasked] * pos_weight, target[unmasked] * pos_weight)
jepa_loss = mask_loss + jepa_ctx_weight * context_loss
```

#### Linear Probe Loss

Contrastive loss between encoder representation and text embedding:

```python
probe_out = LinearProbe(target_context[:, :, -1, :])  # Last layer output
probe_out = normalize(probe_out)
probe_loss = 1 - mean(cosine_similarity(probe_out, normalize(text_emb)))
```

#### Decoder Loss

Reconstruction loss for joint positions and velocities:

```python
# Decode latent → predict 68D
decoded = LatentDecoder(target_context[:, 1:, -1, :])

# Convert to 271D and compare with ground truth
# Terms:
# - MSE for 68D prediction
# - Smooth L1 for velocities  
# - Penalty for foot contact violations
```

### Training Flow

```
Per training step:
1. Load batch: motion (B, T, 271), joints (B, T, 22, 3), text_clip (B, 1, 512)
2. Normalize motion features
3. Build random span mask (~25% of frames)
4. Forward pass:
   a. encoder(motion, zeros_like(text), mask=mask, return_all=True) → masked_context
   b. ema_encoder(motion, zeros_like(text), mask=None, return_all=True) → target_context
5. Compute JEPA loss on masked_context vs target_context
6. Compute probe loss on target_context
7. Compute decoder loss on LatentDecoder(target_context)
8. Accumulate gradients (gradient accumulation steps)
9. Update parameters and EMA
10. Log metrics via W&B (every 50 steps)
```

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

The horizon controls sequence length. It's updated via `set_horizon()` called at the start of each batch.

### Checkpoint Artifacts

Saved to `checkpoint_dir/`:

| File | Content |
|------|---------|
| `pretrain_latest.pt` | Latest model state |
| `pretrain_best_val.pt` | Best validation loss |
| `pretrain_best_eval.pt` | Best decoder loss |

Each checkpoint contains:
- encoder, jepa_predictor weights
- encoder_ema (preferred for generation)
- optimizer, scaler state
- lr_scheduler, aux_lr_scheduler state
- metadata (session_id, resume_id)

## Data Representations

### 271D Feature Format

| Slice | Size | Description |
|-------|------|-------------|
| [0:3] | 3 | Root height Y, Root velocity X, Root velocity Z |
| [3:69] | 66 | 22 joints × 3D RIC positions |
| [69:201] | 132 | 22 joints × 6D rotations |
| [201:267] | 66 | 22 joints × 3D local velocities |
| [267:271] | 4 | Foot contact flags |

### 68D Reduced State

| Slice | Size | Description |
|-------|------|-------------|
| [0:1] | 1 | Root height Y |
| [1:2] | 1 | Root velocity X |
| [2:3] | 1 | Root velocity Z |
| [3:5] | 2 | sin(dyaw), cos(dyaw) |
| [5:68] | 63 | 21 joints × 3D RIC positions |

## Configuration (`Config`)

```python
MotionHistoryEncoderConfig(
    frame_feature_dim=271,
    text_embedding_dim=512,
    hidden_size=512,                 # Default in Config()
    intermediate_size=4 * 512,
    num_hidden_layers=4,             # Default in Config()
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

FlowMatchingPredictorConfig(
    hidden_size=256,
    intermediate_size=768,           # Default
    num_hidden_layers=4,             # Default
    num_attention_heads=8,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    attention_bias=True,
    attention_dropout=0.1,
)
```

Active defaults from `Config()`:

```python
encoder_config = MotionHistoryEncoderConfig()  # Uses above defaults
predictor_config = FlowMatchingPredictorConfig()  # Uses above defaults
```

## Integration Points

### During Pretraining
```
train_pretrain(config) → PretrainTrainer
  ├── Creates dataloader + normalizer via create_dataloader()
  ├── Instantiates: encoder, jepa_predictor, decoder, linear_probe
  ├── EMA wrapper on encoder only
  ├── OneCycleLR schedulers
  └── Saves checkpoints every checkpoint_interval epochs
```

### During Finetuning (separate file: `finetune_trainer.py`)
```
train_finetune(config, train_loader, val_loader) → FiTrainer
  ├── Uses encoder and FlowMatchingPredictor (not JepaPredictor)
  ├── EMA on both encoder and predictor
  └── Flow matching objective with consistency loss
```

### During Inference
```
HumanMotionGenerator.load_from_checkpoint()
  ├── Loads encoder_ema or encoder weights
  ├── Loads predictor_ema or predictor weights  
  └── Inference via generate_sequence()
```