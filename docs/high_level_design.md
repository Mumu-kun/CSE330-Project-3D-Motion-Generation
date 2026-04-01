# High-Level Architecture Design: Human Motion Generation

This document summarizes the architecture that is actually implemented in `src/`. The project is a text-conditioned autoregressive motion generator built around a GRU history encoder, a flow-matching spatial transformer, and a frame-by-frame conversion pipeline between absolute positions and compact motion features.

## 1. System Summary

At a high level, each generated frame is produced in four stages:

1. Encode recent 271D motion history with text conditioning.
2. Predict a denoised next-frame state in a reduced 68D representation.
3. Convert that reduced state into absolute global joint positions.
4. Convert the new positions back into a 271D frame and append them to history.

The recurrent state is therefore not a latent-only state. It is a pair of concrete, interpretable histories:
- absolute joint positions;
- derived 271D feature frames.

## 2. Main Components

### A. Motion History Encoder

Implemented in `src/models.py` as `MotionHistoryEncoder`.

Role:
- reads a variable-length motion window `(B, T, 271)`;
- conditions each timestep on pooled CLIP text `(B, 512)`;
- emits one context token per joint `(B, 22, D_joint)`.

Current data path:

```text
motion history (B, T, 271)
text embedding (B, 512)
    -> text_to_hidden(text) for GRU initial state
    -> text_proj(text) repeated across T
    -> concat(motion, repeated_text)
    -> GRU over time
    -> last hidden state
    -> MLP
    -> reshape to (B, 22, per_joint_out_dim)
```

Why it exists:
- The GRU supplies temporal memory cheaply.
- The predictor can then operate over joints at a single frame rather than running a full spatiotemporal transformer over the whole sequence.

### B. Flow Matching Predictor

Implemented in `src/models.py` as `FlowMatchingPredictor`.

Role:
- predicts flow in a reduced next-frame state space;
- reasons jointly over root and non-root joints using attention across 22 tokens.

Current tokenization:
- token 0: root
- tokens 1-21: non-root joints

Per-token inputs are assembled from three sources:
- noisy reduced-state features;
- per-joint history features from the GRU encoder;
- current-frame causal features extracted from the latest 271D frame.

Conditioning path:

```text
time t -> sinusoidal embedding -> MLP
text -> linear projection
time embedding + text projection -> AdaLN conditioning
```

Structure prior:
- A `KinematicChainEncoder` provides a fixed embedding per joint based on chain id and depth.
- Each transformer layer adds this prior through a learned scalar gate, initialized as a no-op.

### C. Frame Conversion Layer

Implemented primarily in `src/utils/motion_utils.py`.

This layer is what keeps training and inference grounded in explicit motion geometry.

Key responsibilities:
- convert full 271D frames to the 68D predictor target;
- reconstruct absolute positions from predicted reduced states;
- derive fresh 271D features from generated positions;
- optionally canonicalize generated positions with FK-consistent offsets.

This conversion layer is the bridge between:
- the model-friendly reduced state used by flow matching;
- the motion-friendly 271D representation used by the encoder and dataset.

### D. Human Motion Generator

Implemented in `src/models.py` as `HumanMotionGenerator`.

Role:
- orchestrates encoder, predictor, and conversion utilities for autoregressive generation.

Important implementation choice:
- the generator keeps **absolute positions** as the primary autoregressive state;
- 271D features are re-derived every step instead of treating the predictor output as a complete persistent feature frame.

That makes the generation loop easier to reason about and avoids accumulating drift from repeated feature-only updates.

## 3. Current Data Representations

### 271D frame representation

Used by:
- dataset loading;
- GRU history encoding;
- feature normalization;
- frame-by-frame reconstruction after generation.

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
- flow-matching target construction;
- ODE integration at inference time.

Layout:

```text
[0:5]   root height, root vel x/z, sin(dyaw), cos(dyaw)
[5:68]  21 non-root joints x 3D RIC positions
```

Important note:
- the helper that builds this state is still named `subset_271d_to_72d(...)`;
- that name is legacy, but the implemented state is 68D.

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
3. Walk through the sequence one frame at a time with `encoder.gru_step(...)`.
4. For each prediction step:
   - extract current-frame conditioning from the latest frame;
   - build the target reduced next-frame state;
   - sample Gaussian noise `x0`;
   - sample a random noise level `t`;
   - interpolate `xt = t*x1 + (1-t)*x0`;
   - predict the flow target `x1 - x0`.
5. Accumulate MSE across all flattened frame-level samples.

### Optional rollout during training

The training loop can partially replace teacher-forced next inputs with model-generated rollouts.

Current behavior:
- rollout probability is linearly scheduled from `rollout_prob_start` to `rollout_prob_end`;
- rollout generation runs a short ODE loop in reduced-state space;
- the rolled state is converted to positions, then back to a 271D frame before being fed into the GRU on the next step.

This gives the encoder some exposure to its own generated history without abandoning stable teacher-forced supervision.

### Validation and EMA

Validation uses the same incremental loss machinery with optional EMA models.

Implemented support includes:
- EMA copies of encoder and predictor;
- best-train and best-validation checkpoints;
- loss-vs-t diagnostic CSV/plot artifacts;
- optional timing instrumentation for forward/backward and rollout hotspots.

## 5. Inference Architecture

Inference is implemented in `HumanMotionGenerator.generate_sequence(...)`.

### Generation loop

For each output frame:

1. Take the latest history window of 271D frames.
2. Encode that window with the GRU history encoder.
3. Extract current-frame causal features from the latest frame.
4. Initialize a noisy reduced state `x_t ~ N(0, I)` in 68D.
5. Integrate the predictor over `num_steps` Euler updates.
6. Denormalize the reduced state if needed.
7. Convert it to absolute joint positions with:
   - previous root position;
   - previous root rotation.
8. Convert the new positions back into a 271D frame.
9. Append positions, features, and relative shifts to history.

### Position-first design

One of the most important current design choices is that inference is position-first:

- new positions are the canonical generated artifact;
- 271D features are derived from those positions after generation;
- relative shifts are derived outputs, not the primary recurrent state.

This keeps the loop aligned with geometric utilities such as IK, FK, root-motion integration, and foot-contact recomputation.

### FK-consistent option

When `use_fk=True`, the generator:
- computes FK offsets from the seed history;
- allows `generated_positions_to_271d(...)` to produce FK-consistent positions;
- replaces raw predicted positions with those FK-consistent positions when available.

That path aims to keep bone structure more stable during long rollouts.

## 6. Why This Architecture Makes Sense

The current design separates concerns cleanly:

- The GRU handles temporal accumulation over history.
- The transformer handles spatial coupling across joints for the next frame.
- The motion utilities keep the model anchored to explicit geometry.

That split is especially useful here because the predictor does not need to model full-sequence attention. It only needs to solve a well-conditioned next-frame denoising problem with rich contextual inputs.

## 7. Practical Notes

### What is no longer true

Older project notes may still mention:
- a 72D predictor target;
- track-space relative-shift prediction as the main runtime state;
- a 261D previous-frame representation;
- CLIP long-text chunk averaging.

Those do not describe the current implementation accurately.

### What to trust first

When docs and comments disagree, use these files as the source of truth:
- `src/models.py`
- `src/utils/motion_utils.py`
- `src/utils/train_utils.py`
- `src/config.py`

This document has been aligned to those files.
