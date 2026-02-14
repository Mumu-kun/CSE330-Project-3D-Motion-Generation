# Model Sizing Guide for Training on 4000 HumanML3D Samples

## Executive Summary

This guide provides recommendations for adjusting model architecture parameters when training on a subset of 4000 samples from the HumanML3D dataset. The current configuration is designed for the full dataset (~15,000+ samples) and requires adjustment to prevent overfitting and improve training efficiency on the smaller subset.

---

## 1. Current Architecture Analysis

### 1.1 Model Components

The architecture consists of two main components:

#### MotionHistoryEncoder (MHE)
- **Purpose**: Encodes motion history and text conditioning into per-joint context vectors
- **Architecture Flow**:
  1. Text projection: 512D CLIP → 32D latent
  2. Per-joint encoder: 47D → 128D → 64D (MLP)
  3. Temporal GRU: 1408D input → 128D hidden (bidirectional)
  4. Per-joint head: 256D → 256D → 1408D (MLP)
- **Output**: (B, 22, 64) context vectors

#### FlowMatchingPredictor (FMP)
- **Purpose**: Predicts velocity field for flow-based motion generation
- **Architecture Flow**:
  1. Kinematic encoder: Chain/depth embeddings → 128D
  2. Input projection: 81D → 128D
  3. Spatial Transformer: 4 layers, 4 heads, 128D model
  4. Noise head: 128D → 128D → 3D
- **Output**: (B, 22, 3) velocity vectors

### 1.2 Current Configuration

| Parameter | Current Value | Description |
|-----------|---------------|-------------|
| `text_projection_dim` | 32 | Text latent dimension |
| `joint_feature_projection_dim` | 64 | Joint-level latent size |
| `per_joint_out_dim` | 64 | Context vector per joint |
| `model_dim` | 128 | Primary embedding dimension |
| `num_encoder_layers` | 1 | GRU layers in MHE |
| `num_flow_layers` | 4 | Transformer layers in FMP |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.1 | Dropout rate |
| `bidirectional_gru` | True | Bidirectional GRU flag |

### 1.3 Parameter Count Estimation

```
MotionHistoryEncoder:
├── text_projection:      512 × 32 = 16,384
├── per_joint_encoder:    47×128 + 128×64 ≈ 14,400
├── gru (bidirectional):  2 × 3 × (1408×128 + 128×128) ≈ 1,180,000
├── per_joint_head:       256×256 + 256×1408 ≈ 427,000
└── null_tokens:          ~800
                        Total MHE: ~1.64M parameters

FlowMatchingPredictor:
├── kinematic_encoder:    5×64 + 8×64 = 832
├── input_proj:           81×128 = 10,368
├── transformer (4 layers):
│   ├── self_attn:        4 × (128² × 4) ≈ 262,144
│   └── ffn:              4 × (128×256×2) ≈ 262,144
├── noise_head:           128×128 + 128×3 ≈ 16,900
└── null_tokens:          ~265
                        Total FMP: ~550K parameters

GRAND TOTAL: ~2.2M parameters
```

---

## 2. Dataset Characteristics

### 2.1 HumanML3D Data Format

| Property | Value |
|----------|-------|
| Feature dimension | 263D |
| Number of joints | 22 |
| Joint dimension | 3D (x, y, z) |
| Frame rate | 20 FPS |
| Max frames | 200 |
| Min motion length | 40 frames |

### 2.2 Feature Layout (263D)

| Index Range | Feature Type | Dimensions |
|-------------|--------------|------------|
| 0:4 | Root global (rot_vel, lin_vel_xz) | 4 |
| 4:67 | RIC positions (21 joints × 3) | 63 |
| 67:193 | RIC rotations (21 joints × 6) | 126 |
| 193:259 | Local velocities (22 joints × 3) | 66 |
| 259:263 | Foot contacts | 4 |

### 2.3 Sample Size Considerations

With **4000 samples**:
- **Full dataset**: ~15,000+ motion sequences
- **Subset ratio**: ~27% of full dataset
- **Effective training signal**: Significantly reduced

---

## 3. Model Sizing Principles

### 3.1 Parameter-to-Sample Ratio

The fundamental constraint for small datasets:

| Regime | Ratio (samples/parameter) | Risk Level |
|--------|---------------------------|------------|
| Underfitting | > 1000 | Model too small |
| Optimal | 10 - 100 | Balanced |
| Overfitting | < 10 | Model too large |

**Current situation**: 4000 samples / 2.2M params ≈ **0.002 samples/parameter**

This indicates **severe overfitting risk** with current configuration.

### 3.2 Target Parameter Budget

For 4000 samples, recommended parameter budgets:

| Strategy | Target Parameters | Samples/Param |
|----------|-------------------|---------------|
| Conservative | 40K - 100K | 40 - 100 |
| Moderate | 100K - 400K | 10 - 40 |
| Aggressive | 400K - 800K | 5 - 10 |

**Recommendation**: Target **200K - 500K parameters** with strong regularization.

### 3.3 Scaling Strategies

1. **Width Scaling**: Reduce hidden dimensions
2. **Depth Scaling**: Reduce number of layers
3. **Attention Scaling**: Reduce number of heads
4. **Bottleneck Scaling**: Add bottleneck layers

---

## 4. Specific Parameter Recommendations

### 4.1 Recommended Configuration for 4000 Samples

```python
# Recommended Config for 4000 samples
@dataclass
class Config:
    # Model architecture - MotionHistoryEncoder
    text_projection_dim: int = 16      # ↓ from 32
    joint_feature_projection_dim: int = 32  # ↓ from 64
    per_joint_out_dim: int = 32        # ↓ from 64
    
    # Model architecture - General
    model_dim: int = 64                # ↓ from 128
    num_encoder_layers: int = 1        # = same
    num_flow_layers: int = 2           # ↓ from 4
    dropout: float = 0.3               # ↑ from 0.1
    bidirectional_gru: bool = True     # = same
    
    # Model architecture - FlowMatchingPredictor
    num_heads: int = 2                 # ↓ from 4
    
    # Training settings (adjusted for small dataset)
    batch_size: int = 64               # ↓ from 200
    learning_rate: float = 5e-5        # ↓ from 1e-4
    weight_decay: float = 1e-4         # ↑ from 1e-5
```

### 4.2 Parameter-by-Parameter Rationale

#### `text_projection_dim`: 32 → 16

| Aspect | Analysis |
|--------|----------|
| **Current impact** | 16K params in projection layer |
| **Reduction** | ~8K params saved |
| **Rationale** | Text conditioning is auxiliary; 16D provides sufficient semantic signal |
| **Risk** | Low - text is not the primary motion driver |

#### `joint_feature_projection_dim`: 64 → 32

| Aspect | Analysis |
|--------|----------|
| **Current impact** | Affects per-joint encoder MLP |
| **Reduction** | ~7K params saved in encoder |
| **Rationale** | Per-joint features can be compressed; 32D captures local joint context |
| **Risk** | Low - local joint motion is relatively simple |

#### `per_joint_out_dim`: 64 → 32

| Aspect | Analysis |
|--------|----------|
| **Current impact** | Output context dimension, affects FMP input |
| **Reduction** | ~200K params in per-joint head |
| **Rationale** | Context vectors can be compressed; 32D sufficient for conditioning |
| **Risk** | Medium - may reduce expressiveness of joint interactions |

#### `model_dim`: 128 → 64

| Aspect | Analysis |
|--------|----------|
| **Current impact** | Core dimension for GRU and Transformer |
| **Reduction** | ~800K params in GRU, ~300K in Transformer |
| **Rationale** | Most impactful reduction; 64D is common for small datasets |
| **Risk** | Medium - reduces model capacity significantly |

#### `num_flow_layers`: 4 → 2

| Aspect | Analysis |
|--------|----------|
| **Current impact** | Transformer depth in FMP |
| **Reduction** | ~260K params saved |
| **Rationale** | 2 layers sufficient for 22-joint spatial attention |
| **Risk** | Low - spatial relationships are relatively local |

#### `num_heads`: 4 → 2

| Aspect | Analysis |
|--------|----------|
| **Current impact** | Attention granularity |
| **Reduction** | Minimal params, but reduces overfitting |
| **Rationale** | 2 heads can capture global/local patterns |
| **Risk** | Low - 22 joints is small for attention |

#### `dropout`: 0.1 → 0.3

| Aspect | Analysis |
|--------|----------|
| **Current impact** | Regularization strength |
| **Rationale** | Higher dropout critical for small datasets |
| **Risk** | May slow convergence, but prevents overfitting |

### 4.3 Estimated Parameter Count After Reduction

```
MotionHistoryEncoder (reduced):
├── text_projection:      512 × 16 = 8,192
├── per_joint_encoder:    47×64 + 64×32 ≈ 5,200
├── gru (bidirectional):  2 × 3 × (704×64 + 64×64) ≈ 295,000
├── per_joint_head:       128×128 + 128×704 ≈ 106,000
└── null_tokens:          ~800
                        Total MHE: ~415K parameters

FlowMatchingPredictor (reduced):
├── kinematic_encoder:    5×32 + 8×32 = 416
├── input_proj:           49×64 = 3,136
├── transformer (2 layers):
│   ├── self_attn:        2 × (64² × 4) ≈ 32,768
│   └── ffn:              2 × (64×128×2) ≈ 32,768
├── noise_head:           64×64 + 64×3 ≈ 4,300
└── null_tokens:          ~265
                        Total FMP: ~75K parameters

GRAND TOTAL: ~490K parameters
```

**Reduction**: 2.2M → 490K (78% reduction)

---

## 5. Training Adjustments for Small Datasets

### 5.1 Regularization Enhancements

```python
# Additional regularization settings
weight_decay: float = 1e-4        # Increased from 1e-5
gradient_clip: float = 0.5        # Reduced from 1.0
dropout: float = 0.3              # Increased from 0.1

# Optional: Add label smoothing for flow matching
label_smoothing: float = 0.1
```

### 5.2 Learning Rate Schedule

```python
# More conservative learning for small dataset
learning_rate: float = 5e-5       # Reduced from 1e-4
warmup_steps: int = 500           # Reduced from 1000
lr_decay: float = 0.9             # Faster decay
lr_decay_epoch: int = 5           # More frequent decay
```

### 5.3 Early Stopping

```python
# Early stopping configuration
early_stopping_patience: int = 15  # Stop if no improvement for 15 epochs
early_stopping_metric: str = "val_loss"
```

### 5.4 Data Augmentation

For motion data, consider:
- **Temporal cropping**: Random subsequences
- **Gaussian noise**: Add small noise to features
- **Time warping**: Slight speed variations

---

## 6. Alternative Configurations

### 6.1 Conservative Configuration (~200K params)

For maximum generalization:

```python
text_projection_dim: int = 8
joint_feature_projection_dim: int = 24
per_joint_out_dim: int = 24
model_dim: int = 48
num_encoder_layers: int = 1
num_flow_layers: int = 1
num_heads: int = 2
dropout: float = 0.4
bidirectional_gru: bool = False  # Unidirectional for fewer params
```

### 6.2 Balanced Configuration (~500K params)

Recommended default:

```python
text_projection_dim: int = 16
joint_feature_projection_dim: int = 32
per_joint_out_dim: int = 32
model_dim: int = 64
num_encoder_layers: int = 1
num_flow_layers: int = 2
num_heads: int = 2
dropout: float = 0.3
bidirectional_gru: bool = True
```

### 6.3 Aggressive Configuration (~800K params)

If validation metrics show underfitting:

```python
text_projection_dim: int = 24
joint_feature_projection_dim: int = 48
per_joint_out_dim: int = 48
model_dim: int = 96
num_encoder_layers: int = 1
num_flow_layers: int = 3
num_heads: int = 3
dropout: float = 0.25
bidirectional_gru: bool = True
```

---

## 7. Validation Strategy

### 7.1 Overfitting Detection

Monitor these metrics during training:

| Metric | Healthy Sign | Overfitting Sign |
|--------|--------------|------------------|
| Train loss | Decreasing | Decreasing fast |
| Val loss | Decreasing | Increasing or plateau |
| Train/Val gap | < 2× | > 3× |
| Motion quality | Improving | Degraded diversity |

### 7.2 Cross-Validation

For 4000 samples, use **5-fold cross-validation**:
- Each fold: 3200 train, 800 validation
- Report mean and std of metrics

### 7.3 Evaluation Metrics

Use standard motion generation metrics:
- **R-precision**: Text-motion alignment
- **FID**: Distribution matching
- **Diversity**: Motion variety
- **MultiModality**: Generation diversity

---

## 8. Implementation Checklist

### 8.1 Before Training

- [ ] Update config with recommended parameters
- [ ] Implement early stopping
- [ ] Set up validation split (80/20)
- [ ] Configure learning rate scheduler
- [ ] Enable gradient clipping

### 8.2 During Training

- [ ] Monitor train/val loss ratio
- [ ] Check for loss divergence
- [ ] Visualize generated motions periodically
- [ ] Save best checkpoint by validation loss

### 8.3 After Training

- [ ] Evaluate on held-out test set
- [ ] Compare with baseline (current config)
- [ ] Analyze failure cases
- [ ] Document results for future reference

---

## 9. Summary Table

| Parameter | Current | Recommended | Reduction |
|-----------|---------|-------------|-----------|
| `text_projection_dim` | 32 | 16 | 50% |
| `joint_feature_projection_dim` | 64 | 32 | 50% |
| `per_joint_out_dim` | 64 | 32 | 50% |
| `model_dim` | 128 | 64 | 50% |
| `num_encoder_layers` | 1 | 1 | 0% |
| `num_flow_layers` | 4 | 2 | 50% |
| `num_heads` | 4 | 2 | 50% |
| `dropout` | 0.1 | 0.3 | +200% |
| `learning_rate` | 1e-4 | 5e-5 | 50% |
| `weight_decay` | 1e-5 | 1e-4 | +900% |
| **Total params** | ~2.2M | ~490K | 78% |

---

## 10. References

1. **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) - Parameter-data relationships
2. **A Recipe for Training Neural Networks** (Karpathy, 2019) - Practical training advice
3. **HumanML3D** (Guo et al., 2022) - Dataset documentation
4. **MoMask** (Guo et al., 2024) - Reference architecture

---

*Document generated for training on 4000 HumanML3D samples. Adjust based on validation results.*