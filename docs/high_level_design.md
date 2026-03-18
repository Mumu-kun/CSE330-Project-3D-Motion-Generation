# High-Level Architecture Design: Human Motion Generation

This project focuses on generating high-quality 3D human motion sequences from textual descriptions. The architecture is a two-stage pipeline that combines spatiotemporal transformer encoding with continuous flow matching.

## 1. Conceptual Workflow

The generation process follows a "Conditioned Denoising" paradigm:
1. **Context Construction**: Past motion history and text prompts are encoded into a rich latent representation using a spatiotemporal transformer.
2. **Denoising Prediction**: A spatial transformer predicts the velocity field required to transport noisy samples toward the realistic data manifold.
3. **Iterative Refinement**: This prediction is repeated over several steps (ODE integration) to reconstruct a clean motion frame.

---

## 2. Model Architecture

### A. Motion History Encoder (GRU-based Context Encoder)

The encoder uses a **GRU (Gated Recurrent Unit)** architecture to process motion history and text conditioning.

#### Forward Pass Data Pipeline

```
Input: motion_seq(B, T, 271), text_emb(B, 512)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. TEXT PROJECTION                                         │
│     text_proj = Linear(text_emb)           # (B, text_proj_dim)│
│     t_scaled = text_scale * text_proj       # Apply text scale │
│     t_rep = t_scaled.unsqueeze(1).expand(B, T, -1)  # (B,T,D)│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. INITIALIZE GRU HIDDEN STATE                             │
│     h0 = text_to_hidden(text_emb)       # (B, hidden_dim)  │
│     h0 = h0.unsqueeze(0).repeat(num_layers, 1, 1)  # (L,B,H)│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. CONCATENATE MOTION + TEXT                               │
│     gru_input = cat([motion_seq, t_rep], dim=-1)           │
│                                     # (B, T, motion_dim + text_proj_dim)│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. GRU FORWARD PASS                                        │
│     h_seq, h_next = gru(gru_input, h0)  # h_seq: (B,T,H) │
│     h_last = h_seq[:, -1, :]             # Last timestep  │
│                                     # (B, hidden_dim)     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  5. MLP TO PER-JOINT TOKENS                                 │
│     joint_tokens = global_to_joints(h_last)                 │
│                                     # (B, 22 * per_joint_dim)│
│     history_features = joint_tokens.view(B, 22, per_joint_dim)│
│                                     # (B, 22, per_joint_dim)│
└─────────────────────────────────────────────────────────────┘

Output: history_features(B, 22, per_joint_dim)
```

#### Step Function (for AR inference)

For autoregressive generation, the encoder provides a `step()` method:

```
Input: x_t(B, 271), text_emb(B, 512), h(Optional[L, B, H])
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. PREPARE SINGLE TIMESTEP INPUT                          │
│     motion_in = x_t.unsqueeze(1)         # (B, 1, 271)    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RUN GRU BLOCK (same as forward pass)                   │
│     history_features, h_next = _gru_block(motion_in, text_emb, h)│
└─────────────────────────────────────────────────────────────┘

Output: history_features(B, 22, per_joint_dim), h_next(L, B, H)
```

---

### B. Flow Matching Predictor (Spatial Transformer)

The core generation engine that predicts velocity fields in joint space.

#### Forward Pass Data Pipeline

```
Input: history_features(B,22,64), noise_level(B,), 
       noisy_target(B,72), prev_frame_features(B,261)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. NORMALIZATION (if inference mode)                       │
│     if normalizer exists and normalize=True:                │
│         prev_frame_features = normalizer.normalize_prev_frame│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PREV FRAME SPLIT                                        │
│     if prev_frame_features is None:                         │
│         prev_root = null_prev_root.expand(B, 9)             │
│         prev_joints = null_prev_joint.expand(B, 21, 12)     │
│     else:                                                   │
│         prev_root = prev_frame_features[:, :9]    # (B,9)   │
│         prev_joints = prev_frame_features[:, 9:]  # (B,252) │
│         prev_joints = prev_joints.reshape(B, 21, 12)        │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. HISTORY PROJECTION                                      │
│     history_proj = input_proj_history(history_features)     │
│                                    # (B, 22, model_dim)     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. PREV FRAME PROJECTION                                   │
│     prev_root_proj = proj_root(prev_root).unsqueeze(1)      │
│                                    # (B, 1, model_dim)      │
│     prev_joint_proj = proj_joint(prev_joints)               │
│                                    # (B, 21, model_dim)     │
│     prev_proj = cat([prev_root_proj, prev_joint_proj], dim=1)│
│                                    # (B, 22, model_dim)     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  5. CONDITION COMBINATION                                   │
│     cond_proj = history_proj + prev_proj  # (B,22,model_dim)│
│     # Element-wise addition: history context + prev state   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  6. NOISY TARGET PROJECTION                                 │
│     if noisy_target is None:                                │
│         noisy_target = randn(B, 72)                         │
│     noisy_root = noisy_target[:, :9]           # (B, 9)     │
│     noisy_joints = noisy_target[:, 9:].reshape(B, 21, 3)    │
│     noisy_root_proj = proj_noisy_root(noisy_root).unsqueeze(1)│
│     noisy_joint_proj = proj_noisy_joint(noisy_joints)       │
│     noisy_proj = cat([noisy_root_proj, noisy_joint_proj], dim=1)│
│                                    # (B, 22, model_dim)     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  7. COMBINE CONDITION + NOISY                               │
│     x = cond_proj + noisy_proj  # (B, 22, model_dim)        │
│     # Element-wise addition of condition and noisy state    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  8. TIME EMBEDDING                                          │
│     t_emb = sinusoidal_time_embedding(noise_level)          │
│                                    # (B, time_embed_dim)    │
│     t_bias = time_mlp(t_emb)       # (B, model_dim)         │
│     x = x + t_bias.unsqueeze(1)    # Add to all joints      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  9. KINEMATIC BIAS                                          │
│     joint_ids = arange(22)                                  │
│     kinematic_bias = kinematic_encoder(joint_ids) # (22,D)  │
│     x = x + kinematic_bias.unsqueeze(0) # Broadcast to batch│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  10. SPATIAL TRANSFORMER (3 layers)                         │
│      x = spatial_transformer(x)   # (B, 22, model_dim)      │
│      # Bidirectional attention over 22 joints               │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  11. OUTPUT HEADS                                           │
│      root_token = x[:, 0:1, :]        # (B, 1, model_dim)   │
│      joint_tokens = x[:, 1:, :]       # (B, 21, model_dim)  │
│      root_out = root_head(root_token) # (B, 1, 9)           │
│      joint_out = joint_head(joint_tokens) # (B, 21, 3)      │
│      pred_frame = cat([root_out.squeeze(1),                 │
│                        joint_out.reshape(B, -1)], dim=-1)   │
│                                    # (B, 72)                │
└─────────────────────────────────────────────────────────────┘

Output: pred_frame(B, 72) - predicted velocity field
```

---

### C. Human Motion Generator (Pipeline Wrapper)

Integrates the Encoder and Predictor into a single interface for autoregressive generation.

#### Generation Loop Data Pipeline

```
Input: text, num_frames, num_steps, guidance_scale, input_features
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. TEXT ENCODING                                           │
│     if text is str: text = clip_encoder(text) # (1,1,512)   │
│     elif text is list: text = clip_encoder(text)            │
│     B = text.shape[0]                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. HISTORY INITIALIZATION                                  │
│     if input_features is None:                              │
│         root_pos = zeros(B, 3)                              │
│         root_tracker = RootPositionTracker(root_pos)        │
│         feature_history = null_history.expand(B, 1, 271)    │
│     else:                                                   │
│         feature_history = input_features.clone()            │
│         root_tracker = RootPositionTracker.from_history(...)│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  FOR frame_idx in range(num_frames):                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3a. EXTRACT LAST FRAME                               │  │
│  │      last_frame = feature_history[:, -1]  # (B, 271)  │  │
│  │      prev_root_pos = root_tracker.get()   # (B, 3)    │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3b. ENCODE CONTEXT (CFG)                             │  │
│  │      context_cond = encoder(text, feature_history,    │  │
│  │                            normalize=True)[:, -1, :, :]│  │
│  │                         # (B, 22, 64)                 │  │
│  │      context_uncond = encoder(None, feature_history,  │  │
│  │                              normalize=True)[:, -1,:,:]│  │
│  │                         # (B, 22, 64)                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3c. EXTRACT PREV FRAME FEATURES                      │  │
│  │      prev_frame_features = extract_prev_frame_features│  │
│  │                              (last_frame)  # (B, 261)  │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3d. FLOW MATCHING ODE LOOP                           │  │
│  │      x_t = randn(B, 72)                               │  │
│  │      dt = 1.0 / num_steps                             │  │
│  │      for step in range(num_steps):                    │  │
│  │          t = full((B,), step * dt)                    │  │
│  │          v_cond = predictor(context_cond, t, x_t,     │  │
│  │                              prev_frame_features,     │  │
│  │                              normalize=True)          │  │
│  │          v_uncond = predictor(context_uncond, t, x_t, │  │
│  │                                prev_frame_features,   │  │
│  │                                normalize=True)        │  │
│  │          v_t = v_uncond + guidance_scale*(v_cond-v_uncond)│
│  │          x_t = x_t + v_t * dt                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3e. DENORMALIZE OUTPUT                               │  │
│  │      if normalizer:                                   │  │
│  │          x_t = normalizer.denormalize_flow_output(x_t)│  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3f. CONVERT 72D → 271D                               │  │
│  │      new_frame, new_root_pos = flow_output_to_271d(   │  │
│  │          x_t, last_frame, prev_root_pos)              │  │
│  │      # new_frame: (B, 271), new_root_pos: (B, 3)      │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3g. UPDATE HISTORY                                   │  │
│  │      root_tracker.update(new_frame)                   │  │
│  │      feature_history = cat([feature_history,          │  │
│  │                          new_frame.unsqueeze(1)], dim=1)│  │
│  │                      # (B, N+1, 271)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│  END FOR                                                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. CONVERT TO POSITIONS                                    │
│     position_history = features_to_positions(feature_history)│
│                                    # (B, N+num_frames, 22, 3)│
└─────────────────────────────────────────────────────────────┘

Output: position_history(B, N+num_frames, 22, 3)
```

---

## 3. Data Representation

### 271D Feature Format (HumanML3D Custom)

| Index Range | Feature Type     | Description                            |
| :---------- | :--------------- | :------------------------------------- |
| `[0:3]`     | Root Global      | Height Y, Velocity X, Velocity Z       |
| `[3:69]`    | RIC Positions    | 22 joints × 3D relative positions      |
| `[69:201]`  | 6D Rotations     | 22 joints × 6D rotation representation |
| `[201:267]` | Local Velocities | 22 joints × 3D velocities              |
| `[267:271]` | Foot Contacts    | 4 binary contact flags                 |

### Flow Matching Target (72D)

| Index Range | Feature Type  | Description                              |
| :---------- | :------------ | :--------------------------------------- |
| `[0:9]`     | Root Features | Height(1) + Velocity(2) + Rotation_6d(6) |
| `[9:72]`    | Joint RIC     | 21 joints × 3D RIC positions             |

### Previous Frame Features (261D)

| Index Range | Feature Type   | Description                              |
| :---------- | :------------- | :--------------------------------------- |
| `[0:9]`     | Root Features  | Height(1) + Velocity(2) + Rotation_6d(6) |
| `[9:261]`   | Joint Features | 21 joints × 12D (RIC + Rot + Vel)        |

---

## 4. Training Configuration

| Parameter               | Value | Description                                      |
| :---------------------- | :---- | :----------------------------------------------- |
| `encoder_hidden_dim`    | 256   | GRU hidden dimension                             |
| `encoder_num_layers`    | 2     | MotionHistoryEncoder GRU layers                  |
| `encoder_per_joint_dim` | 64    | Per-joint context vector dimension               |
| `predictor_model_dim`   | 128   | FlowMatchingPredictor hidden dimension           |
| `predictor_num_layers`  | 2     | FlowMatchingPredictor spatial transformer layers |
| `batch_size`            | 192   | Training batch size                              |
| `learning_rate`         | 1e-4  | Adam learning rate                               |
| `num_epochs`            | 1000  | Total training epochs                            |
| `dropout`               | 0.1   | Dropout rate                                     |
| `ema_decay`             | 0.999 | EMA decay for validation                         |
| `cfg_dropout`           | 0.1   | CFG dropout probability                          |

### Progressive Horizon Curriculum

Training uses progressive context windows:
- Stage 1: 16 frames
- Stage 2: 32 frames
- Stage 3: 64 frames
- Stage 4: 128 frames (optional)

---

## 5. Data Flow Diagram

```mermaid
graph TD
    Text[Text Prompt] --> CLIP[CLIP Encoder]
    History[Motion History 271D] --> Encoder[MotionHistoryEncoder]
    CLIP --> Encoder
    Encoder --> Context[Context Vectors Bx22x64]
    
    Noise[Gaussian Noise 72D] --> Predictor[FlowMatchingPredictor]
    Context --> Predictor
    PrevFrame[Previous Frame 261D] --> Predictor
    Time[Flow Time t] --> Predictor
    
    Predictor --> Velocity[Velocity Field 72D]
    Velocity --> Euler[Euler ODE Step]
    Euler --> CleanFrame[Clean Frame 72D]
    CleanFrame --> Convert[flow_output_to_271d]
    Convert --> NewFrame[New Frame 271D]
```

---

## 6. Inference Pipeline

```mermaid
sequenceDiagram
    participant Text as Text Prompt
    participant CLIP as CLIP Encoder
    participant MHE as MotionHistoryEncoder
    participant FMP as FlowMatchingPredictor
    participant ODE as ODE Solver
    
    loop For each frame
        Text->>CLIP: Encode text
        CLIP->>MHE: CLIP embeddings
        History->>MHE: Feature history
        MHE->>FMP: Context vectors
        loop N ODE steps
            FMP->>ODE: Predict velocity
            ODE->>FMP: Updated x_t
        end
        ODE->>History: Append new frame
    end
```

---

## 7. Key Implementation Details

### Tensor Shape Transformations Summary

| Stage              | Operation                  | Input Shape                            | Output Shape                     |
| :----------------- | :------------------------- | :------------------------------------- | :------------------------------- |
| Text Projection    | Linear + Expand            | `(B, L_text, 512)`                     | `(B, L_text, 22, D)`             |
| Global Token       | Extract + Linear           | `(B, T, 271)`                          | `(B, T, 1, D)`                   |
| Track Tokens       | Extract + View + Linear    | `(B, T, 271)`                          | `(B, T, 21, D)`                  |
| Motion Assembly    | Concat                     | `(B, T, 1, D)` + `(B, T, 21, D)`       | `(B, T, 22, D)`                  |
| Sequence Concat    | Concat                     | `(B, L_text, 22, D)` + `(B, T, 22, D)` | `(B, L_text+T, 22, D)`           |
| Spatial Attention  | Reshape + Attn             | `(B, T, 22, D)`                        | `(B*T, 22, D)` → `(B, T, 22, D)` |
| Temporal Attention | Transpose + Reshape + Attn | `(B, T, 22, D)`                        | `(B*22, T, D)` → `(B, T, 22, D)` |
| Output Extraction  | Slice                      | `(B, L_text+T, 22, D)`                 | `(B, T, 22, D)`                  |

### Addition Operations (Broadcasting)

| Operation            | Left Shape      | Right Shape     | Result Shape    |
| :------------------- | :-------------- | :-------------- | :-------------- |
| Kinematic Bias       | `(B, T, 22, D)` | `(1, 1, 22, D)` | `(B, T, 22, D)` |
| History + Prev       | `(B, 22, D)`    | `(B, 22, D)`    | `(B, 22, D)`    |
| Cond + Noisy         | `(B, 22, D)`    | `(B, 22, D)`    | `(B, 22, D)`    |
| Time Bias            | `(B, 22, D)`    | `(B, 1, D)`     | `(B, 22, D)`    |
| Kinematic Bias (FMP) | `(B, 22, D)`    | `(1, 22, D)`    | `(B, 22, D)`    |

---

## 8. Why This Architecture?

- **Flow Matching**: Offers stable training dynamics compared to GANs, with faster inference than diffusion models
- **Kinematic Chain Bias**: Ensures anatomically plausible motion by encoding skeletal hierarchy
- **Dual-Stage Design**: Separates context understanding (Encoder) from generation (Predictor)
- **HumanML3D Compatible**: Works with standard datasets and BVH visualization tools
