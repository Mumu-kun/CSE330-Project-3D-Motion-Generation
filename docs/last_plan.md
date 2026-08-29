# Implementation Plan: Restoring Text Authority & Breaking Flow Matching Loss Plateau

## Goal Description
Empirical audits on the active Phase 3 checkpoint (`epoch=2000`) revealed that the model plateaued at a validation flow loss of `0.468` and fails to complete tasks end-to-end because it is **functionally blind to text prompts**:
1. **Caption Shuffle Test:** Replacing *"throwing a ball"* with *"jumping jacks"* altered model loss by only **$0.10\%$**.
2. **Backpropagation Starvation:** Average gradient norm reaching text tokens was **$0.00005$** (virtually zero).
3. **AdaLN Magnitude Drowning:** Timestep embedding norm ($32.0$) overwhelmed text projection norm ($3.2$), causing AdaLN to be **$90.4\%$ time and only $9.6\%$ text**.
4. **Cross-Attention Imbalance:** History frames absorbed **$67.7\%$** of attention weights while text received only **$32.3\%$**.
5. **Inverted Frame Addition:** `masked_cond` injected frame $-39$ into future frame $+1$ at $0.02\%$ norm.

This implementation plan re-architects the conditioning pathways in [`FlowMatchingPredictor`](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/src/utils/models/flow_matching_predictor.py), updates the training pipeline in [`FlowMatchingTrainer`](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/src/utils/models/flow_matching_trainer.py), synchronizes the Kaggle build bundle, and provides immediate local unit tests to verify balanced gradient flow before retraining on Kaggle.

---

## User Review Required

> [!IMPORTANT]
> **Retraining is Required to Learn Text Semantics:** 
> Because the current checkpoint was trained for 2,000 epochs with text gradients near zero ($0.00005$), the existing model weights never learned to map text descriptions to motion trajectories. Applying these architectural fixes will allow gradients to flow into the text pathways, but the model must be retrained on Kaggle (estimated 300–500 epochs with balanced conditioning) to drive validation flow loss from `0.468` down to the target `0.18 - 0.22` convergence band.

---

## Proposed Changes

### Component 1: Predictor Model Architecture (`src/utils/models/flow_matching_predictor.py`)

#### [MODIFY] [flow_matching_predictor.py](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/src/utils/models/flow_matching_predictor.py)

1. **Equalize AdaLN Conditioning Modulation (50% Time / 50% Text):**
   * *Location:* `FlowMatchingPredictor.forward()` around line 341.
   * *Change:* Normalize both `time_cond` and `self.text_proj(text_embedding)` to unit vectors before blending:
     ```python
     time_norm = F.normalize(time_cond, dim=-1)
     text_norm = F.normalize(self.text_proj(text_embedding), dim=-1)
     adaln_cond = 0.5 * time_norm + 0.5 * text_norm
     ```
   * *Rationale:* Eliminates the $10\times$ norm mismatch ($32.0$ vs $3.2$). Text and timestep will have equal modulation authority over all 6 transformer layers.

2. **Remove the Inverted `masked_cond` Residual Addition:**
   * *Location:* `PredictorLayer.forward()` lines 212–255.
   * *Change:* Remove `encoder_cond_proj`, `gate_proj`, and the residual line `hidden_states = hidden_states + (self.self_attn.cond_scale * gate) * masked_cond`.
   * *Rationale:* Eliminates the temporally inverted addition (frame $-39 \to$ frame $+1$) which contributed only $0.02\%$ norm as misaligned noise.

3. **Softmax Cross-Attention Text Prior Bias (+1.5):**
   * *Location:* `PredictorRopeCrossAttention.forward()` around line 130.
   * *Change:* Add an additive logit bias of $+1.5$ to text token positions prior to softmax:
     ```python
     # attn_scores: (B, num_heads, T_query, Total_Keys)
     # Total_Keys = S_text + T_hist
     text_bias = torch.zeros_like(attn_scores)
     text_bias[..., :S_text] = 1.5
     attn_weights = F.softmax(attn_scores + text_bias, dim=-1)
     ```
   * *Rationale:* Counteracts key entropy bias so text tokens achieve $\sim 50\%$ cross-attention budget (up from $32.3\%$).

---

### Component 2: Training Pipeline & Diagnostic Hooks (`src/utils/models/flow_matching_trainer.py`)

#### [MODIFY] [flow_matching_trainer.py](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/src/utils/models/flow_matching_trainer.py)

1. **Consistent Conditioning Interface:**
   * Ensure `history_states=track_features` is cleanly passed to all predictor invocations during training and validation.
2. **Text Gradient Monitoring:**
   * Add a lightweight diagnostic log during validation steps that computes and logs $\|\nabla_{\text{text}}\| / \|\nabla_{\text{history}}\|$ to verify gradient health during training.

---

### Component 3: Build Synchronization

#### [MODIFY] [utils_kaggle.py](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/build/utils_kaggle.py)
* Re-run [`combine_utils.py`](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/combine_utils.py) to compile all updates into the single-file Kaggle training bundle [`build/utils_kaggle.py`](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/build/utils_kaggle.py).

---

### Component 4: Notebook Diagnostic Suite (`src/phase3.ipynb`)

#### [MODIFY] [phase3.ipynb](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/src/phase3.ipynb)
* Update training cells and verification diagnostics to benchmark the updated architecture.

---

## Verification Plan

### Automated Local Unit Tests
Run a dedicated verification script ([`scripts/test_rebalanced_architecture.py`](file:///d:/ml_project/CSE330-Project-3D-Motion-Generation/scripts/test_rebalanced_architecture.py)) on GPU to measure:
1. **AdaLN Ratio Test:** Confirm `adaln_cond` perturbation on prompt swap increases from $14.8\% \to > 50\%$.
2. **Attention Mass Test:** Confirm cross-attention text budget increases from $32.3\% \to \sim 50\%$.
3. **Gradient Attribution Test:** Confirm backward gradient flowing into text tokens increases by at least $10\times$ relative to baseline.
4. **Residual Cleanliness:** Confirm hidden states pass cleanly through self-attention without dead gate parameters.

### Retraining Verification on Kaggle
1. Launch training with the new `utils_kaggle.py`.
2. Monitor training curve: validation flow loss should break below the previous `0.468` plateau within 100 epochs and trend toward `0.20`.
3. In inference, test sequential prompts (*"a person walks forward, turns left, and sits down"*) to confirm complete multi-stage action execution.
