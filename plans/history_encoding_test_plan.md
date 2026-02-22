# History Encoding and Rollout Performance Test Plan

## Objective
Test the model's ability to encode ground truth history vs predicted history, and understand long horizon rollout performance.

## Test Design

### Test 1: Ground Truth History Encoding
Use increasing amounts of ground truth seed frames to predict the next frame.

**Setup**:
- Seed frames: 1, 3, 5, 10, 20 ground truth frames
- Predict: 1 frame ahead
- Compare: Predicted frame vs ground truth

**Purpose**: Test if the encoder properly encodes ground truth history.

### Test 2: Autoregressive Rollout with Ground Truth Seeds
Use ground truth seed frames, then autoregressively predict multiple frames.

**Setup**:
- Seed frames: 1, 3, 5, 10 ground truth frames
- Predict: 5-10 frames autoregressively
- Compare: Each predicted frame vs ground truth

**Purpose**: Test how well ground truth history initialization supports rollout.

### Test 3: Teacher Forcing vs Autoregressive
Compare two approaches:
1. **Teacher forcing**: Always use ground truth as history, predict next frame
2. **Autoregressive**: Use predicted frames as history

**Setup**:
- For each frame position, compare:
  - Teacher forcing prediction (using GT history)
  - Autoregressive prediction (using predicted history)

**Purpose**: Isolate encoder quality from rollout error accumulation.

### Test 4: Long Horizon Rollout
Test performance over longer horizons (20-50 frames).

**Setup**:
- Seed frames: 5, 10 ground truth frames
- Predict: 20, 50 frames
- Track: Error per frame, error growth rate

**Purpose**: Understand long horizon performance limits.

## Metrics to Track

1. **Per-frame MSE**: Position error for each predicted frame
2. **RIC MSE**: Pose correctness
3. **Root velocity MSE**: Motion dynamics
4. **Error growth rate**: How fast error accumulates
5. **Teacher forcing gap**: Difference between teacher forcing and autoregressive

## Expected Insights

1. **Encoder quality**: If teacher forcing works well but autoregressive fails, encoder is good but rollout accumulates error
2. **History encoding**: If more seed frames don't help, encoder may not be using history effectively
3. **Prediction horizon**: Identify where error becomes unacceptable
4. **Error source**: Distinguish between encoder errors and predictor errors

## Implementation

```python
def test_ground_truth_history_encoding():
    """
    Test encoder with increasing ground truth seed frames.
    Predict only 1 frame ahead to isolate encoder quality.
    """
    seed_counts = [1, 3, 5, 10, 20]
    for num_seeds in seed_counts:
        # Use ground truth frames as history
        history = gt_frames[:num_seeds]
        
        # Predict next frame
        predicted = predict_one_frame(history)
        
        # Compare with ground truth
        gt_next = gt_frames[num_seeds]
        error = mse(predicted, gt_next)
        
        print(f"Seeds: {num_seeds}, Error: {error}")

def test_teacher_forcing_vs_autoregressive():
    """
    Compare teacher forcing with autoregressive rollout.
    """
    num_seeds = 5
    num_predict = 10
    
    # Teacher forcing: always use GT history
    tf_errors = []
    for i in range(num_predict):
        history = gt_frames[:num_seeds + i]
        predicted = predict_one_frame(history)
        gt_next = gt_frames[num_seeds + i]
        tf_errors.append(mse(predicted, gt_next))
    
    # Autoregressive: use predicted frames as history
    ar_errors = []
    history = gt_frames[:num_seeds]
    for i in range(num_predict):
        predicted = predict_one_frame(history)
        gt_next = gt_frames[num_seeds + i]
        ar_errors.append(mse(predicted, gt_next))
        history = append(history, predicted)  # Use predicted as history
    
    # Compare
    for i in range(num_predict):
        gap = ar_errors[i] - tf_errors[i]
        print(f"Frame {i+1}: TF={tf_errors[i]:.4f}, AR={ar_errors[i]:.4f}, Gap={gap:.4f}")
```

## Output Format

```
=== Ground Truth History Encoding ===
Seeds | 1-Frame Error
------|---------------
    1 | 0.XXXX
    3 | 0.XXXX
    5 | 0.XXXX
   10 | 0.XXXX
   20 | 0.XXXX

=== Teacher Forcing vs Autoregressive ===
Frame | TF Error | AR Error | Gap
------|----------|----------|--------
    1 |   0.XXXX |   0.XXXX | 0.XXXX
    2 |   0.XXXX |   0.XXXX | 0.XXXX
   ...

=== Long Horizon Rollout ===
Frame | GT Seeds 5 | GT Seeds 10
------|------------|-------------
    1 |     0.XXXX |     0.XXXX
   10 |     0.XXXX |     0.XXXX
   20 |     0.XXXX |     0.XXXX
   50 |     0.XXXX |     0.XXXX
```
