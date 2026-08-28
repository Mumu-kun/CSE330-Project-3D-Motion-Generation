# Audit Log: Failed Experiments & Diagnostic Findings

This document tracks all experimental hypotheses, tested modifications, empirical diagnostic metrics, and root-cause analyses for the 3D Text-to-Motion Flow Matching predictor.

---

## 1. Overview of Experiment: Pure Text Cross-Attention (Decoupled Stream)

### Hypothesis
In the baseline predictor, `cross_attn` attended to `combined_cond = torch.cat([text_seq, track_features], dim=1)` ($28 \text{ text tokens} + 40 \text{ history frames} = 68 \text{ tokens}$).
* **Observation**: In layers 3–5, history frames absorbed $68\%–79\%$ of cross-attention weights, while text tokens absorbed only $20.6\%$.
* **Hypothesis**: Decoupling cross-attention to attend *only* to text tokens (`track_features = text_seq`) while passing history frames *only* to self-attention boundary gating (`history_states = track_features`) would eliminate text dilution and lower Flow Matching loss.

---

## 2. Empirical Results & Direct Comparison

Both models were evaluated on the held-out validation batch ($B=128, T_{\text{target}}=79, H_{\text{enc}}=512$) using identical 20-step ODE integration:

| Diagnostic Metric | Original Architecture (`combined_cond = [text, history]`) | Pure Text Cross-Attention (`track_features = text`) | Verdict |
| :--- | :---: | :---: | :---: |
| **Overall Latent MSE** ($z_1$) | **`1.156694`** | **`1.673237`** | ❌ **Worse (+0.52)** |
| **Ratio (Pred / Random Noise)** | **`0.4961`** ($<0.5 \implies$ learned signal) | **`0.7174`** ($\approx 0.8 \implies$ near random) | ❌ **Worse (+0.22)** |
| **Baseline Zero MSE** | `1.327451` | `1.333249` | Reference |
| **Frame Cosine Similarity (Mean)** | **`0.5125`** | **`0.3268`** | ❌ **Worse (-0.19)** |
| **Frame Cosine Similarity (Min / Max)** | `[-0.1852, 0.9467]` | `[-0.2080, 0.8843]` | ❌ **Worse** |
| **Per-Channel Pearson Correlation ($>0.3$)**| **`142 / 512 channels`** | **`84 / 512 channels`** | ❌ **Worse (-58 channels)**|
| **Velocity CosSim ($t=0.50$)** | **`0.9185`** | **`0.9180`** | Neutral |
| **Single-Step Denoising CosSim ($t=0.80$)** | **`0.9928`** | **`0.9928`** | Neutral |

---

## 3. Why the Pure Text Cross-Attention Approach Failed

### Failure Cause 1: Loss of Kinematic History in Cross-Attention
* In 3D human motion generation, generating the next frame $z_t$ requires knowing the exact spatial configurations, limb velocities, and root trajectory from the preceding 40 frames (`track_features`).
* Self-attention boundary gating alone (which injects a single gated residual at the sequence boundary) is **insufficient** for deep transformer layers to resolve temporal joint dependencies.
* Cross-attention over individual history tokens provides the necessary token-to-token receptive field for preserving physical momentum and smooth limb continuation.

### Failure Cause 2: Cross-Attention Weight Specialization
* The cross-attention projection matrices ($W_Q, W_K, W_V$) in the pre-trained predictor were specialized over 2,000 epochs to process a key sequence of length $S + T_{\text{hist}} = 68$.
* Forcing the input length to $S=28$ disrupted the learned positional and cross-modal attention maps, causing the multi-step ODE integration to drift away from the ground truth manifold ($MSE = 1.15 \to 1.67$).

---

## 4. Key Learnings & Proven Insights

1. **History Dominating Attention Is Normal for Motion**:
   * The fact that history tokens absorb $\sim 70\%$ of cross-attention mass is an expected physical requirement of motion continuation, not an error.
2. **Text Adherence is Controlled via CFG Scale, Not Key Stripping**:
   * Text conditioning is properly amplified at inference via Classifier-Free Guidance:
     $$v_t = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$
   * In the default configuration, `guidance_scale` was set to `1.0` (CFG disabled). Increasing `guidance_scale` to `2.5 - 3.5` amplifies the text guidance vector without destroying history cross-attention.
3. **Logit-Normal Timestep Sampling vs Uniform**:
   * Timestep sampling concentrated in $[0.2, 0.8]$ is valid, but the architecture must retain `combined_cond = [text_seq, track_features]`.

---

## 5. Decision & Action Taken

* **Revert Action**: All source files reverted back to the clean git commit state (`git restore`).
* **Active Checkpoints Maintained**:
  * `checkpoints/phase3/predictor/phase3_predictor_latest_20260802_205537.pt` (Epoch 2000, Best baseline MSE 1.15, CosSim 0.51).
  * `checkpoints/phase3/predictor/phase3_predictor_best_val_20260828_022718.pt` (Converted from Kaggle zip to .pt).
