# Pipeline: Training & Inference

**Last Updated**: 2026-06-25

---

## Stage 1: Masked JEPA Pretraining

**Goal**: Learn general temporal motion representations from unlabeled data.

**Encoder mode: Non-causal.** Bidirectional attention lets the encoder build rich representations using full-sequence context.

### Architecture

```
EMA Encoder (full input, non-causal)  → target_context (B, T, L, H)     [no gradient]
Encoder (masked input, non-causal)    → masked_context (B, T, L, H)
                                      → JepaPredictor → predicted_context
                                      → L1(predicted, target)
```

### Mask Generation

Random contiguous spans (not independent frames). Configurable number of spans and min/max span length. Masked positions replaced by learnable `mask_token`.

### Loss Function

```python
per_frame_loss = L1(predicted, target).mean(dim=(layers, H))   # (B, T)

mask_loss = (per_frame_loss * mask).sum() / mask.sum()
context_loss = (per_frame_loss * ~mask * pos_weight).sum() / ~mask.sum()

jepa_loss = mask_loss + 0.2 * context_loss
```

`pos_weight = 1 / sqrt(distance_to_nearest_mask)` — higher weight near mask boundaries.

### Auxiliary Losses

**Linear Probe** (text alignment):
```python
probe_out = LinearProbe(target_context)  # → (B, 512)
loss = 1 - cosine_similarity(probe_out, text_emb)
```

**Decoder** (geometric grounding):
```python
decoded = LatentDecoder(target_context[:, 1:, -1, :])  # → 68D
# → positions → 271D → compare with ground truth
# Loss: MSE root_y + 0.2 MSE root_xz + 1.0 MSE yaw + 0.5 SmoothL1 vel + 1.0 MSE joints
```

### Optimizers

- **Main**: encoder + jepa_predictor parameters
- **Auxiliary**: linear_probe + decoder parameters

### Curriculum

```python
curriculum = [
    {"horizon": 5,  "epochs": 100},
    {"horizon": 10, "epochs": 200},
    {"horizon": 20, "epochs": 300},
    {"horizon": 40, "epochs": 1000},
]
```

Number of masked spans also increases from 2 to 6.

### Checkpoints

| File | Trigger |
|------|---------|
| `pretrain_latest_{step}.pt` | Every 50 epochs |
| `pretrain_best_val.pt` | Best validation loss |
| `pretrain_best_eval.pt` | Best decoder loss |

---

## Stage 2: Decoder Finetuning

**Goal**: Train `LatentDecoder` on reconstruction. Encoder is frozen and operates in causal mode.

**Encoder mode: Causal.** Matches inference conditions. Both encoder copies (live + EMA) are frozen. Only the decoder and its EMA copy are trained.

### Architecture

```
EMA Encoder (frozen, causal)  → target_context[:, 1:, -1, :] → latent
LatentDecoder (trainable)     → 68D → positions → 271D
```

### Loss

```python
decoded = decoder(latent)
loss = 1.0 × MSE(root_y) + 0.2 × MSE(root_xz) + 1.0 × MSE(yaw)
     + 0.5 × SmoothL1(vel) + 1.0 × MSE(joint_positions)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 5e-4 |
| weight_decay | 1e-5 |
| ema_decay (decoder) | 0.999 |
| batch_size | 200 |
| effective_batch_size | 400 (2× accumulation) |
| gradient_clip | 30.0 |
| lr_schedule | OneCycleLR (cosine), 5-epoch warmup |

### Checkpoints

| File | Trigger |
|------|---------|
| `finetune_latest_{step}.pt` | Every 50 epochs |
| `finetune_best_val.pt` | Best validation loss |

---

## Stage 3: Inference (Future)

**Goal**: Generate motion from text using the flow matching predictor.

### Generation Loop

```
For each frame:
1. encoder.step(current_frame, text_emb, history) → track_features  [causal]
2. Initialize x_t ~ N(0, I) in H_enc
3. FlowMatchingPredictor(x_t, t, track_features, text) → flow  [B, N, H_enc]
4. Integrate ODE (Heun solver, 20 steps, end-biased schedule)
5. Convert flow → 68D → positions (22, 3) → 271D → append to history
```

### ODE Integration

- **Schedule**: End-biased `t = 1 - (1-s)^p` where `p = 3.0`
- **Solver**: Heun (2nd order)
  ```
  k1 = predict_velocity(x_t, t_start)
  x_euler = x_t + dt × k1
  k2 = predict_velocity(x_euler, t_end)
  x_{t+1} = x_t + 0.5 × dt × (k1 + k2)
  ```

### Classifier-Free Guidance

When `guidance_scale != 1.0`:
```
velocity = uncond_velocity + guidance_scale × (cond_velocity - uncond_velocity)
```

### Input/Output

- **Input**: Text string, list of strings, or pre-computed CLIP embedding `(B, 1, 512)`
- **Output**: Position history `(B, T, 22, 3)`, feature history `(B, T, 271)`, relative shifts `(B, T, 22, 3)`

---

## Training Loop (Stages 1 & 2)

```
Per step:
1. Load batch: motion (B, T, 271), joints (B, T, 22, 3), CLIP (B, 1, 512)
2. Normalize motion features
3. Generate mask (stage 1 only)
4. Forward through encoder/training module (non-causal for stage 1, causal for stage 2)
5. Compute loss
6. Gradient accumulation (2 steps)
7. Optimizer step → EMA update → scheduler step
8. Log to W&B
9. Validate every 5 epochs
```

---

## Configuration

| Parameter | Default | Stage |
|-----------|---------|-------|
| learning_rate | 5e-4 | Both |
| batch_size | 200 | Both |
| effective_batch_size | 400 | Both |
| weight_decay | 1e-5 | Both |
| gradient_clip | 30.0 | Both |
| ema_decay | 0.999 | Both |
| lr_warmup_epochs | 5 | Both |
| jepa_ctx_weight | 0.2 | Pretrain only |
| num_inference_steps | 20 | Inference |
| inference_t_schedule_power | 3.0 | Inference |
| guidance_scale | 1.0 | Inference |
| val_interval | 5 | Both |
| checkpoint_interval | 50 | Both |

---

## Future Phases

**Flow Matching Predictor Training**: Train `FlowMatchingPredictor` with frozen encoder for denoising diffusion. Cross-attention to encoder features + AdaLN on text/time. Output is `[B, N, H_enc]`.

**Consistency Distillation**: Reduce 20 inference steps to 1-4 for real-time generation. Config flags already exist (`use_consistency_loss`, `consistency_loss_weight=10.0`).

**Long-Horizon Generation**: Sliding window with overlap blending for sequences beyond training horizon (40 frames).

**Multi-Modal Conditioning**: Extend beyond text to music/audio features, trajectory constraints, and style references.

---

## Evaluation Targets

| Metric | Description | Target |
|--------|-------------|--------|
| FID | Distribution similarity to real motion | < 10.0 |
| R-Precision | Text-motion retrieval accuracy | > 0.8 |
| Diversity | Variance across samples from same text | > 5.0 |
| Foot Contact Rate | Realistic foot-ground contact | > 0.95 |
