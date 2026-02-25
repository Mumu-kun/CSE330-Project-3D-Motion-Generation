# Plan: Update 3D Human Motion Inference Notebook

## Overview

The `misc/3d-human-motion-inference.ipynb` notebook needs to be updated to use the new model interfaces. The main issues are:

1. **FeatureNormalizer** - Not being used, but models now require it for inference
2. **CLIPEncoder output shape** - Returns `(B, 1, 512)` not `(B, 512)`
3. **HumanMotionGenerator.load_from_checkpoint** - Now requires `normalizer` parameter
4. **generate_sequence interface** - Accepts text directly (str or List[str]) or pre-encoded tensor
5. **Dataset text_clip shape** - Now returns `(B, 1, 512)` instead of `(B, 512)`

**Note:** Import paths should remain as-is (without `src.` prefix) for cloud environment compatibility.

## Detailed Changes

### 1. Import Paths (Keep As-Is)

Keep the current import structure without `src.` prefix:
```python
from config import Config
from models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator
from utils.dataset import Text2MotionDataset, text2motion_collate_fn, create_dataloader
from utils.text_encoder import CLIPEncoder
from utils.visualization import visualize_motion
from utils.motion_utils import FeatureNormalizer
```

**Remove unused import:**
```python
# Remove this - not used in inference
from utils.train_utils import train
```

### 2. Update create_dataloader Call

**Current:**
```python
dataloader = create_dataloader(config, split="train", shuffle=True)
```

**New:**
```python
dataloader, normalizer = create_dataloader(config, split="train", shuffle=True)
```

The `create_dataloader` function now returns a tuple `(DataLoader, FeatureNormalizer)`.

### 3. Update HumanMotionGenerator.load_from_checkpoint

**Current:**
```python
generator = HumanMotionGenerator.load_from_checkpoint(
    checkpoint_path=checkpoint_path, config=config, device=str(device)
)
```

**New:**
```python
generator = HumanMotionGenerator.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    config=config,
    device=str(device),
    normalizer=normalizer,  # Pass the FeatureNormalizer
)
```

### 4. Fix CLIPEncoder Usage

**Current:**
```python
clip_encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32")
clip_encoder.to(device)

# Later...
text_embedding = clip_encoder([prompt]).to(device)  # Expected (1, 512)
```

**New:**
```python
clip_encoder = CLIPEncoder()  # Default model is openai/clip-vit-base-patch32
clip_encoder.to(device)

# CLIPEncoder now returns (B, 1, 512) instead of (B, 512)
# No need to call .to(device) on the output - it's already on the correct device
text_embedding = clip_encoder([prompt])  # Returns (1, 1, 512)
```

### 5. Update generate_sequence Calls

The `generate_sequence` method now accepts:
- `text`: str, List[str], or pre-encoded tensor `(B, l_seq, 512)`
- It handles CLIP encoding internally if text is a string/list

**Option A - Pass text directly (simpler):**
```python
joint_positions = generator.generate_sequence(
    text=prompt,  # Pass string directly
    num_frames=100,
    num_steps=25,
    guidance_scale=2.5,
    dataset_type="t2m",
)
```

**Option B - Use pre-encoded embeddings:**
```python
# Dataset provides text_clip as (B, 1, 512)
joint_positions = generator.generate_sequence(
    text=d["text_clip"],  # Pre-encoded tensor
    num_frames=100,
    num_steps=25,
    guidance_scale=2.5,
    dataset_type="t2m",
)
```

### 6. Fix Dataset Text Embedding Shape

The dataset now returns `text_clip` as `(B, 1, 512)` instead of `(B, 512)`.

**Current:**
```python
text_embedding = d["text_clip"].to(device)  # (1, 512)
```

**New:**
```python
text_embedding = d["text_clip"]  # Already (B, 1, 512), no reshape needed
```

### 7. Remove Unused Imports

Remove `train` from imports since it's not used in inference:
```python
# Remove this line
from utils.train_utils import train
```

## Implementation Order

1. Add `FeatureNormalizer` import from `utils.motion_utils`
2. Update `create_dataloader` call to capture `normalizer`
3. Update `HumanMotionGenerator.load_from_checkpoint` to pass `normalizer`
4. Fix `CLIPEncoder` initialization and usage
5. Update `generate_sequence` calls to use new interface
6. Fix text embedding handling from dataset
7. Update visualization import (change `utils.utils` to `utils.visualization`)
8. Remove unused `train` import

## Key Interface Changes Summary

| Component | Old Interface | New Interface |
|-----------|--------------|---------------|
| `create_dataloader()` | Returns `DataLoader` | Returns `(DataLoader, FeatureNormalizer)` |
| `CLIPEncoder()` | Returns `(B, 512)` | Returns `(B, 1, 512)` |
| `load_from_checkpoint()` | No `normalizer` param | Requires `normalizer` param |
| `generate_sequence()` | `text` must be tensor | `text` can be str, List[str], or tensor |
| Dataset `text_clip` | Shape `(B, 512)` | Shape `(B, 1, 512)` |

## Notes

- Import paths remain without `src.` prefix for cloud environment compatibility
- The `FeatureNormalizer` is essential for proper inference - models normalize internally during inference
- The `generate_sequence` method handles text encoding internally, so passing strings directly is the simplest approach
