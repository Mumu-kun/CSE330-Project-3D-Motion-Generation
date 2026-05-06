# High-Level Architecture Design: Human Motion Generation

This document summarizes the architecture that is actually implemented in `src/`. The project is a text-conditioned autoregressive motion generator built around a causal temporal history encoder, a flow-matching spatial transformer, and a frame-by-frame conversion loop between absolute positions and compact motion features.

## 1. System Summary

At a high level, each generated frame is produced in four stages:

1. Encode recent 271D motion history with text conditioning.
2. Assemble predictor inputs by concatenating noisy reduced-state features, per-joint history features, and current-frame causal features.
3. Predict a denoised next-frame state in a reduced 68D representation.
4. Convert that reduced state into absolute global joint positions.
5. Convert the new positions back into a 271D frame and append them to history.

The recurrent state is therefore not a latent-only state. It is a pair of concrete, interpretable histories:
- absolute joint positions
- derived 271D feature frames

## 2. Main Components

### A. Motion History Encoder

Implemented in `src/models.py` as `MotionHistoryEncoder`.

Role:
- reads a variable-length motion window `(B, T, 271)`
- conditions each timestep on pooled CLIP text `(B, 512)`
- emits one context token per joint `(B, 22, D_joint)` or one context tensor per timestep when `return_all=True`

Current data path:

```text
motion history (B, T, 271)
text embedding (B, 512)
    -> frame_projection(motion)
    -> text_projection(text) -> shift, scale
    -> FiLM-style modulation of every timestep
    -> causal temporal self-attention with RoPE
    -> gated MLP blocks
    -> final norm
    -> per-timestep projection to 22 joint tokens
```

Tensor flow details:
- `frame_projection` and `text_projection` act on each timestep independently, then the text branch applies FiLM-style shift and scale by addition and multiplication, not concatenation.
- `step(...)` appends the newest 271D frame to the explicit `frame_buffer` and recomputes the encoder over the buffered sequence.
- The encoder output is a per-joint context tensor, so the predictor receives joint-aligned history rather than a pooled latent.

Why it exists:
- It gives the predictor a temporally informed summary without forcing the predictor itself to model full spatiotemporal attention.
- The encoder is causal, so each context only depends on current and past frames.

Important implementation note:
- `step(...)` exists for generation, but it currently recomputes over the explicit `frame_buffer`.
- `TemporalCacheState` is part of the interface, but there is no real KV-cache acceleration yet.

### B. Flow Matching Predictor

Implemented in `src/models.py` as `FlowMatchingPredictor`.

Role:
- predicts flow in a reduced next-frame state space
- reasons jointly over root and non-root joints using attention across 22 tokens

Prediction tokenization:
- after the temporary text/global token is removed, token `0` is the root and tokens `1..21` are the non-root joints
- the temporary text token exists only to carry global conditioning through the shared transformer stream

Per-token inputs are assembled from three sources:
- noisy reduced-state features
- per-joint history features from the temporal encoder
- current-frame causal features extracted from the latest 271D frame

The input assembly is explicit concatenation:
- root token input = `root_state + root_history + root_current_frame`
- joint token input = `joint_state + joint_history + joint_current_frame`
- text conditioning is added after projection as a global FiLM branch, not concatenated into every token feature vector

The current implementation also prepends a dedicated text/global token before the joint tokens, then prepends a zero kinematic token so the structural prior can align with that sequence layout.

Conditioning path:

```text
time t -> sinusoidal embedding -> MLP
text -> linear projection
time embedding + text projection -> AdaLN conditioning
```

Structure prior:
- A `KinematicChainEncoder` provides a fixed embedding per joint based on chain id and depth.
- The prior is injected by gated addition into the token stream, layer by layer.
- This is not a concatenation path; the joint sequence remains width-stable while the gate modulates how strongly the prior is added.

### C. Frame Conversion Layer

Implemented primarily in `src/utils/motion_utils.py`.

This layer keeps training and inference grounded in explicit motion geometry.

Key responsibilities:
- convert full 271D frames to the 68D predictor target
- reconstruct absolute positions from predicted reduced states
- derive fresh 271D features from generated positions
- optionally canonicalize generated positions with FK-consistent offsets

This conversion layer bridges:
- the model-friendly reduced state used by flow matching
- the motion-friendly 271D representation used by the encoder and dataset

### D. Human Motion Generator

Implemented in `src/models.py` as `HumanMotionGenerator`.

Role:
- orchestrates encoder, predictor, and conversion utilities for autoregressive generation

Important implementation choices:
- the generator keeps absolute positions as the primary autoregressive state
- 271D features are re-derived every step instead of treating the predictor output as a persistent full feature frame
- the current frame is converted to 257D predictor conditioning with feature packing, then the predictor integrates 68D noise toward the next reduced state
- history length is controlled externally by `horizon` or, if omitted, by `config.horizon`

That keeps context management explicit and avoids a hidden encoder-owned context limit.

## 3. Current Data Representations

### 271D frame representation

Used by:
- dataset loading
- temporal history encoding
- feature normalization
- frame-by-frame reconstruction after generation

Layout:

```text
[0:3]    root height y, root vel x, root vel z
[3:69]   22 joints x 3D RIC positions
[69:201] 22 joints x 6D rotations
[201:267]22 joints x 3D root-local velocities
[267:271]4 foot-contact values
```

### 68D reduced predictor state

Used by:
- flow-matching target construction
- ODE integration at inference time

Layout:

```text
[0:5]   root height, root vel x/z, sin(dyaw), cos(dyaw)
[5:68]  21 non-root joints x 3D RIC positions
```

### 257D current-frame conditioning

Used only as predictor conditioning.

Layout:

```text
[0:5]    root height, root vel x/z, sin(yaw), cos(yaw)
[5:257]  21 x (RIC 3 + rot6d 6 + vel 3)
```

## 4. Training Architecture

Training is implemented in `src/utils/train_utils.py` and centers on `Trainer.incremental_flow_loss(...)`.

### Training step

For each batch:

1. Load raw motion and joints from the dataloader.
2. Normalize 271D motion if a `FeatureNormalizer` is configured.
3. Remove the first frame so each prediction has a previous frame.
4. Form:
   - `hist = motion[:, :-1]`
   - `target_motion = motion[:, 1:]`
5. Run `encoder(hist, text_for_encoder, return_all=True)` to get one context per valid history prefix.
6. Flatten batch and time dimensions.
7. Build:
   - 257D current-frame conditioning from `hist`
   - 68D target reduced state from `target_motion`
8. Sample Gaussian noise `x0` and a timestep `t`.
9. Interpolate `xt = t*x1 + (1-t)*x0`.
10. Predict flow and train against `x1 - x0`.
11. Optionally add consistency loss on high-`t` samples.

This is a parallel loss construction over all prediction steps in the cropped sequence window.

### Rollout status

The config still contains rollout scheduling fields, but the active loss path does not currently replace teacher-forced history with model rollouts. The relevant rollout code is commented out.

So today:
- rollout scheduling metadata exists
- rollout execution in `incremental_flow_loss(...)` is inactive
- returned rollout probability is effectively `0.0`

### Validation and EMA

Validation reuses the same loss builder without CFG dropout and can run on the EMA copies of the models.

Implemented support includes:
- EMA copies of encoder and predictor
- best-train and best-validation checkpoints
- loss-vs-t diagnostic CSV and plot artifacts
- optional timing instrumentation

## 5. Inference Architecture

Inference is implemented in `HumanMotionGenerator.generate_sequence(...)`.

### Generation loop

For each output frame:

1. Take the latest history window of normalized 271D frames.
2. Trim that window according to `horizon` or `config.horizon`.
3. Encode that window with the temporal history encoder.
4. Extract current-frame causal features from the latest frame and keep them as a separate conditioning vector.
5. Initialize a noisy reduced state `x_t ~ N(0, I)` in 68D.
6. Concatenate the reduced-state, history, and current-frame branches inside the predictor, then integrate the predictor over `num_steps` using the end-biased Heun ODE solver.
7. Convert the integrated reduced state to absolute positions using the previous root position and root rotation.
8. Convert the new positions back into a 271D frame.
9. Append positions, features, and relative shifts to history.

### Position-first design

One of the most important current design choices is that inference is position-first:

- new positions are the canonical generated artifact
- 271D features are derived from those positions after generation
- relative shifts are derived outputs, not the primary recurrent state

This keeps the loop aligned with geometry-aware utilities such as IK, FK, root-motion integration, and foot-contact recomputation.

The important distinction is that concatenation is used for token assembly, while addition and gating are used for conditioning and structural bias. That separation keeps the geometry path explicit and the model path easy to inspect.

### FK-consistent option

When `use_fk=True`, the generator:
- computes FK offsets from the seed history
- allows `generated_positions_to_271d(...)` to produce FK-consistent positions
- replaces raw predicted positions with those FK-consistent positions when available

That path aims to keep bone structure more stable during long rollouts.

## 6. Why This Architecture Makes Sense

The current design separates concerns cleanly:

- The temporal encoder handles causal accumulation over history.
- The predictor handles spatial coupling across joints for the next frame.
- The motion utilities keep the model anchored to explicit geometry.

That split is useful because the predictor does not need to solve a full-sequence spatiotemporal problem. It only needs to solve a conditioned next-frame denoising problem with strong temporal context already provided.

## 7. Practical Notes

### What is no longer true

Older project notes may still mention:
- a GRU-based history encoder
- an encoder-owned max context length
- track-space relative-shift prediction as the main runtime state
- long-text CLIP chunk averaging as active behavior

Those do not describe the current implementation accurately.

### What to trust first

When docs and comments disagree, use these files as the source of truth:
- `src/models.py`
- `src/utils/motion_utils.py`
- `src/utils/train_utils.py`
- `src/config.py`

This document has been aligned to those files.
