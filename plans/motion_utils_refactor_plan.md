# Motion Utils Refactoring Plan

## Objective
Refactor `motion_utils.py` to maintain a single canonical feature-to-position conversion function and a single position-to-feature extraction function.

## Use Cases

1. **Dataset Preprocessing** - Convert ground truth joint positions to 271D features
2. **Inference/Motion Generation** - Frame-by-frame feature extraction during generation
3. **Reconstruction** - Convert features back to joint positions

## Final Architecture

### Canonical Functions

```python
# Dataset preprocessing - for ground truth joints
def preprocess_sequence(
    positions: torch.Tensor,
    dataset_type: str = "t2m",
    feet_thre: float = 0.002,
) -> torch.Tensor:
    """
    Preprocess a sequence of joint positions to 271D features.
    Used for dataset preprocessing.
    
    Uses IK + direct RIC for ground truth joints.
    Input: (N, 22, 3) or (B, N, 22, 3)
    Output: (N, 271) or (B, N, 271)
    """

# Reconstruction - features to positions
def features_to_positions(
    features: torch.Tensor,
    dataset_type: str = "t2m",
) -> torch.Tensor:
    """
    Reconstruct positions from 271D features.
    Uses direct RIC transform (perfect reconstruction).
    Input: (N, 271) or (B, N, 271)
    Output: (N, 22, 3) or (B, N, 22, 3)
    """
```

### IncrementalFeatureExtractor (Inference)

```python
class IncrementalFeatureExtractor:
    """
    Stateful incremental feature extractor for frame-by-frame generation.
    Used during inference/motion generation.
    
    Internally uses FK for extracting RIC from newly generated joint positions
    to maintain kinematic consistency.
    """
    
    def process_frame(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame and extract 271D features.
        Uses FK internally for kinematic consistency.
        Input: (B, 22, 3) joint positions for current frame
        Output: (B, 271) feature vectors
        """
```

## Implementation Steps

1. **Remove `extract_features_from_predicted()`** - Not needed as separate function
2. **Keep `preprocess_sequence()`** - For ground truth dataset preprocessing
3. **Keep `features_to_positions()`** - For reconstruction
4. **Keep `IncrementalFeatureExtractor`** - Uses FK internally for RIC extraction

## API Summary

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `preprocess_sequence()` | Dataset preprocessing (ground truth) | positions (N, 22, 3) | features (N, 271) |
| `features_to_positions()` | Reconstruction | features (N, 271) | positions (N, 22, 3) |
| `IncrementalFeatureExtractor` | Frame-by-frame inference (FK internally) | single frame | features |

## Notes

- IncrementalFeatureExtractor uses FK internally for extracting RIC positions from newly generated joints
- This maintains kinematic consistency during autoregressive generation
