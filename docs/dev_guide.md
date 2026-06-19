# Developer Guide: Working with the Human Motion Generation Project

This project implements a text-conditioned autoregressive 3D human motion generator using a causal temporal encoder, flow matching predictor, and frame-by-frame geometric conversion.

## Project Structure

```
src/
├── config.py                      # Configuration dataclasses
├── utils/
│   ├── __init__.py
│   ├── motion_utils.py            # 271D feature conversions, IK/FK
│   ├── dataset.py                 # HumanML3D data loading
│   ├── text_encoder.py            # CLIP text embeddings
│   ├── wandb_logger.py            # Weights & Biases logging
│   ├── visualization.py           # Motion visualization
│   ├── quaternion.py              # Quaternion math
│   └── models/
│       ├── __init__.py            # Shared modules (AdaLN, GatedMLP, etc.)
│       ├── motion_history_encoder.py  # Encoder + JepaPredictor + LinearProbe
│       ├── flow_matching_predictor.py # Predictor + LatentDecoder
│       ├── pretrain_trainer.py        # JEPA pretraining
│       ├── finetune_trainer.py        # Flow matching finetuning
│       └── human_motion_generator.py  # Autoregressive inference
```

## Two-Phase Training Pipeline

### Phase 1: JEPA Pretraining

Pretrains the temporal encoder to predict future representations:

```python
from utils.config import Config
from utils.models.pretrain_trainer import train_pretrain

config = Config()
ema_encoder, _, checkpoint_path = train_pretrain(config, wandb_project="motion-pretrain")
```

Key concepts during pretraining:
- `MotionHistoryEncoder` learns motion representations
- `JepaPredictor` predicts masked future encoder outputs
- `LatentDecoder` provides auxiliary reconstruction
- `LinearProbe` aligns features with text embeddings

### Phase 2: Flow Matching Finetuning

Fine-tunes with denoising diffusion objective:

```python
from utils.models.finetune_trainer import train_finetune

ema_encoder, ema_predictor = train_finetune(
    config, train_loader, val_loader, wandb_project="motion-finetune"
)
```

## Inference

Load and generate sequences:

```python
from utils.models.human_motion_generator import HumanMotionGenerator

generator = HumanMotionGenerator.load_from_checkpoint(
    checkpoint_path="checkpoints/pretrain_latest.pt",
    config=config,
    device="cuda",
    normalizer=normalizer,
)

positions, features, shifts = generator.generate_sequence(
    text="a person walking forward",
    num_frames=200,
    num_steps=20,
)
```

## Configuration

The `Config` dataclass consolidates all settings:

```python
from utils.config import Config, MotionHistoryEncoderConfig, FlowMatchingPredictorConfig

# Use defaults
config = Config()

# Or customize
config = Config(
    encoder_config=MotionHistoryEncoderConfig(
        hidden_size=256,
        num_hidden_layers=6,
    ),
    predictor_config=FlowMatchingPredictorConfig(
        hidden_size=128,
    ),
    learning_rate=1e-4,
    batch_size=128,
)
```

## Key Implementation Details

### Motion Representations

- **271D features**: Full motion frame (positions, rotations, velocities, contacts)
- **68D reduced state**: Minimal representation for flow prediction
- **Conversion**: Bidirectional via `x271_to_x68()`, `x68_to_x271()` in `motion_utils.py`

### Model Features

- **AdaLN**: Adaptive LayerNorm for conditioning (DiT-style)
- **RoPE**: Rotary position encoding for temporal attention
- **FiLM-style text conditioning**: Shift/scale modulation per timestep
- **Register tokens**: Prepend learnable tokens for global context
- **Masked prediction**: Random span masking during pretraining

### Training Notes

- Gradient accumulation via `effective_batch_size`
- OneCycleLR with warmup for learning rate scheduling
- EMA weights used for validation and generation
- W&B logging for metrics and checkpointing

## Running Tests

```bash
# End-to-end pipeline test
python -m pytest tests/test_pipeline_e2e.py -v

# Generator tests
python -m pytest tests/test_human_motion_generator.py -v

# Checkpoint tests
python -m pytest tests/test_checkpoint_save_and_load.py -v
```