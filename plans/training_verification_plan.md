# Training Loop and HumanMotionGenerator Verification Plan

## Overview

This plan outlines the verification steps needed to ensure the training loop runs error-free and the HumanMotionGenerator class functions correctly in preparation for Kaggle training.

## Components to Verify

### 1. Training Loop (`src/utils/train_utils.py`)

The training loop involves:
- **Data Loading**: Text2MotionDataset provides RAW 271D features
- **Normalization**: FeatureNormalizer normalizes raw features before passing to models
- **Encoder**: MotionHistoryEncoder processes text + motion history
- **Predictor**: FlowMatchingPredictor predicts velocity field
- **Loss Computation**: MSE loss between predicted and target velocity

#### Key Functions:
- [`extract_prev_frame_features()`](src/utils/train_utils.py:40) - Extracts 261D from 271D frame
- [`extract_clean_target()`](src/utils/train_utils.py:82) - Extracts 72D from 271D frame
- [`train()`](src/utils/train_utils.py:157) - Main training loop
- [`validate()`](src/utils/train_utils.py:638) - Validation function

### 2. HumanMotionGenerator (`src/models.py:777`)

The generator integrates:
- **MotionHistoryEncoder**: Encodes text + motion history context
- **FlowMatchingPredictor**: Predicts next frame via flow matching
- **Autoregressive Loop**: Generates frames sequentially

#### Key Methods:
- [`generate_sequence()`](src/models.py:795) - Main generation method
- [`load_from_checkpoint()`](src/models.py:963) - Load from checkpoint

### 3. Data Flow

```
Dataset (RAW 271D) 
    → FeatureNormalizer.normalize() 
    → MotionHistoryEncoder(normalize=False) 
    → FlowMatchingPredictor(normalize=False) 
    → Loss
```

## Verification Steps

### Step 1: Unit Test for Feature Extraction Helpers

Verify that `extract_prev_frame_features()` and `extract_clean_target()` work correctly:
- Input: 271D frame tensor
- Output: 261D and 72D tensors with correct slicing

### Step 2: Training Loop Smoke Test

Create a minimal test that:
1. Creates mock models (encoder, predictor)
2. Creates mock dataloader with sample data
3. Runs 1-2 training iterations
4. Verifies no errors occur

### Step 3: HumanMotionGenerator Initialization Test

Verify that:
1. Models can be instantiated with correct config
2. `load_from_checkpoint()` works with mock checkpoint
3. All parameters are on correct device

### Step 4: HumanMotionGenerator Generation Test

Verify that:
1. `generate_sequence()` runs without errors
2. Output shape is correct: (B, T, 22, 3)
3. No NaN values in output

### Step 5: End-to-End Integration Test

Verify the complete pipeline:
1. Load sample data
2. Run training iteration
3. Save checkpoint
4. Load checkpoint into HumanMotionGenerator
5. Generate motion sequence

## Test Files to Create

1. `tests/test_training_loop.py` - Training loop verification
2. `tests/test_generator.py` - HumanMotionGenerator verification

## Potential Issues to Check

1. **Shape Mismatches**: Verify all tensor shapes align
2. **Device Mismatches**: Ensure all tensors on same device
3. **Normalization**: Verify RAW vs normalized feature handling
4. **Text Encoding**: Verify CLIP embedding handling
5. **Feature Slicing**: Verify correct indices for 271D features
6. **Autoregressive Loop**: Verify history tracking in generator

## Success Criteria

- All tests pass without errors
- No NaN values in outputs
- Correct output shapes
- Training loss decreases over iterations
- Generated motion has valid joint positions
