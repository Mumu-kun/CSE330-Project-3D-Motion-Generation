# Flow Predictor Comparison Test Plan

## Objective
Compare FlowMatchingPredictor output with ground truth to understand:
1. How well the model predicts within a small prediction horizon
2. How error accumulates across increasing number of predicted frames
3. Proper alignment of predicted frames with ground truth (accounting for seed frames)

## Data Sources
- **Input features**: `sample_data/000070_vec.npy` - 271D motion features (117 frames)
- **Ground truth joints**: `sample_data/000070_joint.npy` - Global joint positions (117 frames, 22 joints, 3D)
- **Text prompt**: `sample_data/000070.txt` - "a person walks one way then backtracks"
- **Checkpoint**: `tests/checkpoints/best.pt`

## Test Design

### 1. Single Frame Prediction Test
Compare model's 1-step prediction with ground truth:
- Use frame N as history
- Predict frame N+1
- Compare with ground truth frame N+1

**Metrics**:
- 72D flow output MSE (predicted vs ground truth 72D)
- Global position MSE (reconstructed joints vs ground truth joints)
- Per-joint error distribution

### 2. Short Horizon Prediction Test (1-10 frames)
Track error accumulation for short predictions:
- Use frames 0 to N-1 as seed (N = 1, 5, 10)
- Predict next 1, 2, 3, ..., 10 frames
- Compare each predicted frame with corresponding ground truth

**Metrics per frame**:
- 72D output MSE
- Position MSE
- Root position error
- Joint position error (mean across joints)

### 3. Frame Alignment Strategy
```
Ground truth:  [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, ...]
Seed frames:   [F0, F1, F2] (N=3 seed frames)
Predicted:           [P3, P4, P5, P6, P7, ...]
Compare:             F3 vs P3, F4 vs P4, F5 vs P5, ...
```

### 4. Error Accumulation Analysis
Plot error vs frame index to visualize:
- How quickly error grows
- Whether model has learned short-term dynamics
- Whether there's a "prediction horizon" where error explodes

## Implementation Steps

### Step 1: Extract Ground Truth 72D Targets
For each frame in the sequence, extract the 72D target using `extract_clean_target()`:
```python
# From 271D frame -> 72D target
ground_truth_72d = extract_clean_target(frame_271d)  # (B, 72)
```

### Step 2: Run Single Frame Prediction
```python
# Setup
history = frames[0:N]  # N seed frames
prev_frame = frames[N-1]  # Last seed frame
prev_features = extract_prev_frame_features(prev_frame)  # 261D

# Encode context
context = encoder(text, history)

# Predict
x_t = randn(72)
for step in range(num_steps):
    t = step / num_steps
    v = predictor(context, t, x_t, prev_features)
    x_t = x_t + v * dt

# Compare
predicted_72d = x_t
ground_truth_72d = extract_clean_target(frames[N])
```

### Step 3: Reconstruct Global Positions
```python
# From predicted 72D
prev_root_pos = ...  # From previous frame
prev_root_rot_6d = ...  # From previous frame
predicted_positions = flow_output_to_positions(predicted_72d, prev_root_pos, prev_root_rot_6d)

# From ground truth 72D
ground_truth_positions = flow_output_to_positions(ground_truth_72d, prev_root_pos, prev_root_rot_6d)

# Compare
position_error = mse(predicted_positions, ground_truth_positions)
```

### Step 4: Multi-Frame Autoregressive Rollout
```python
errors_72d = []
errors_pos = []

for i in range(num_predict_frames):
    # Predict next frame
    predicted_72d = predict_one_frame(history, context)
    
    # Get ground truth
    gt_72d = extract_clean_target(frames[seed_len + i])
    
    # Compare 72D
    errors_72d.append(mse(predicted_72d, gt_72d))
    
    # Reconstruct positions and compare
    predicted_pos = flow_output_to_positions(predicted_72d, prev_root_pos, prev_root_rot_6d)
    gt_pos = ground_truth_joints[seed_len + i]
    errors_pos.append(mse(predicted_pos, gt_pos))
    
    # Update for next iteration
    history = update_history(predicted_72d)
    prev_root_pos, prev_root_rot_6d = extract_from_predicted(predicted_72d)
```

## Expected Outputs

### Console Output
```
=== Single Frame Prediction ===
72D MSE: 0.XXXX
Position MSE: 0.XXXX
Root error: 0.XXXX
Joint error: 0.XXXX

=== Error Accumulation (1-10 frames) ===
Frame 1: 72D MSE=0.XX, Pos MSE=0.XX
Frame 2: 72D MSE=0.XX, Pos MSE=0.XX
...
Frame 10: 72D MSE=0.XX, Pos MSE=0.XX

=== Error Growth Rate ===
Average error increase per frame: X.XX%
```

### Visualization (Optional)
- Plot: Error vs Frame Index
- Plot: Predicted vs Ground Truth trajectory (2D top-down view)

## Test File Structure
```
tests/test_flow_predictor_comparison.py
├── test_single_frame_prediction()
│   └── Compare 1-step prediction with ground truth
├── test_short_horizon_prediction()
│   └── Track error for 1-10 frame predictions
├── test_error_accumulation()
│   └── Analyze how error grows over time
└── helper functions
    ├── extract_ground_truth_72d()
    ├── reconstruct_positions_from_72d()
    └── compute_per_joint_error()
```

## Key Considerations

1. **Seed Frame Alignment**: Ground truth frames used as seed should not be compared against predictions. Only compare predicted frames.

2. **Position Reconstruction Consistency**: Use the same reconstruction path for both predicted and ground truth:
   - `flow_output_to_positions()` for 72D -> positions
   - Ensure same prev_root_pos and prev_root_rot_6d are used

3. **Model State**: The model may only be trained for short horizons, so expect:
   - Low error for first few frames
   - Increasing error as prediction horizon grows
   - Possible "drift" in root position over time

4. **Metric Selection**:
   - 72D MSE: Direct measure of flow predictor accuracy
   - Position MSE: End-to-end reconstruction accuracy
   - Per-joint error: Identify which joints have largest errors
