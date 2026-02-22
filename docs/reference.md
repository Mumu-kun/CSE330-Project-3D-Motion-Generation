PROJECT: Text-to-Motion generation
ARCH: MotionHistoryEncoder(ARFM Transformer) + FlowMatchingPredictor(Transformer)
DATA: HumanML3D 271D features, 22 joints, 20fps

CONSTANTS:
motion_dim:271 num_joints:22 joint_dim:3 fps:20 max_frames:200
feature_slices: 0:3(global_root) 3:69(RIC pos 22x3) 69:201(RIC rot 22x6) 201:267(vel 22x3) 267:271(foot)

271D FEATURE LAYOUT (from motion_utils.py):
[0:3]   Root height Y, Root velocity X, Root velocity Z (velocity form)
[3:69]  RIC positions (22x3=66D) - local positions relative to root
[69:201] 6D rotations (22x6=132D) - auxiliary features from IK
[201:267] Local velocities (22x3=66D) - causal velocities (current - previous)
[267:271] Foot contacts (4D) - binary contact flags

JOINTS: See src/utils/motion_utils.py:t2m_kinematic_chain
chains: 0=L-leg[0,2,5,8,11] 1=R-leg[0,1,4,7,10] 2=spine[0,3,6,9,12,15] 3=R-arm[9,14,17,19,21] 4=L-arm[9,13,16,18,20]
face_joints:[2,1,17,16] foot_r:[8,11] foot_l:[7,10]

MODELS: See src/models.py
MotionHistoryEncoder (ARFM Feature Fusion Transformer):
  Input: text:(B, l_seq, 512) CLIP sequence embeddings, history:(B, T, 271) motion features
  Architecture:
    1. Text Prefix Tokens: CLIP sequence (B, l_seq, 512) -> linear -> (B, l_seq, d_model)
    2. Global Token: per-timestep global features (16D) -> linear -> 1 token/timestep
       - root_height_y(1) + root_vel_x(1) + root_vel_z(1) + root_rot_6d(6) + root_local_vel(3) + foot_contacts(4) = 16D
    3. Track Tokens: per-timestep local features (12D) -> linear -> add kinematic bias -> 21 tokens/timestep
       - RIC positions(3) + rotation_6d(6) + local_velocity(3) = 12D per joint
       - Kinematic embedding added as position bias (not concatenated)
    4. Spatiotemporal Sequence: [Text Prefix (l_seq x 22); Motion (T x 22)] -> (B, l_seq+T, 22, d_model)
    5. Transformer Stack (4 layers): Spatiotemporal blocks with temporal causal + spatial bidirectional attention
    6. Output: Mean pooling over time -> (B, 22, per_joint_out_dim)
  Output: context:(B, 22, per_joint_out_dim)

CLIPEncoder: See src/utils/text_encoder.py
  Input: text:str or List[str]
  Output: (B, l_seq, 512) sequence embeddings (l_seq=77 for CLIP)

FlowMatchingPredictor: (context:Bx22x64, t:B, x_t:Bx72, prev:Bx261, progress) -> v:Bx72
  Inputs:
    - history_features: (B, 22, per_joint_dim) - Context from MotionHistoryEncoder
    - noise_level: (B,) - Flow time t in [0,1]
    - noisy_target: (B, 72) - Current noisy state x_t
      - [0:9] Root features: height(1) + velocity(2) + rotation_6d(6)
      - [9:72] Joint RIC positions: 21 joints x 3D = 63D
    - prev_frame_features: (B, 261) - Previous frame features
      - [0:9] Root features: height(1) + velocity(2) + rotation_6d(6)
      - [9:261] Joint features: 21 joints x 12D = 252D
        - Per joint: RIC position(3) + rotation_6d(6) + local_velocity(3) = 12D
    - temporal_progress: (B,) - Optional normalized frame progress
  Output: pred_frame:(B, 72)
    - [0:9] Root prediction: height(1) + velocity(2) + rotation_6d(6)
    - [9:72] Joint RIC prediction: 21 joints x 3D = 63D
  Architecture:
    1. History projection: (B, 22, per_joint_dim) -> (B, 22, model_dim)
    2. Prev frame projection: root(9D) + joints(252D) -> (B, 22, model_dim)
    3. Noisy target projection: root(9D) + joints(63D) -> (B, 22, model_dim)
    4. Time embedding: sinusoidal -> MLP -> (B, model_dim)
    5. Kinematic bias: KinematicChainEncoder -> (22, model_dim)
    6. Combine: history + prev + noisy + time + kinematic -> (B, 22, model_dim)
    7. Spatial Transformer: 2-4 layers of transformer encoder
    8. Output heads: root_head -> (B, 9), joint_head -> (B, 21, 3)
    9. Concatenate: (B, 72)

Reconstruction Functions (src/utils/motion_utils.py):
  features_to_positions(features:Nx271) -> positions:Nx22x3
    - Reconstruct global joint positions from 271D features
    - Uses cumulative sum along time dimension (dim=0) for root X,Z from velocities
    - Verified: MSE=0.0, perfect reconstruction (tests/verify_reconstruction.py)
  flow_output_to_positions(flow_output:Bx72, prev_root_pos:Bx3, prev_root_rot_6d:Bx6) -> positions:Bx22x3
    - Reconstruct global joint positions from FlowMatchingPredictor output
    - Uses predicted root velocity to update position, rotation for coordinate transform
    - Verified: MSE=0.0, perfect reconstruction (tests/verify_reconstruction.py)
  flow_output_to_displacements(flow_output:Bx72) -> displacements:Bx22x3
    - Extract joint displacements from FlowMatchingPredictor output
    - Simpler interpretation: output as per-joint deltas to add to current positions
    - Verified: Root velocity extraction matches 271D features (tests/verify_reconstruction.py)

HumanMotionGenerator: (text, num_frames, num_steps, guidance_scale, input_features) -> joints:(B, T, 22, 3)
  - Integrates MotionHistoryEncoder and FlowMatchingPredictor
  - Uses IncrementalFeatureExtractor for autoregressive feature extraction
  - Classifier-free guidance: v = v_uncond + scale * (v_cond - v_uncond)
  - Input text: str, List[str], or pre-encoded tensor (B, 1, 512) from CLIPEncoder
  - input_features: Required, shape (B, N, 271) - initial motion history frames
  - load_from_checkpoint: Uses config.max_text_seq_len (not hardcoded 77)

CONFIG: See src/config.py
behavior_params: text_proj:64 joint_proj:64 model_dim:256 transformer_layers:4
                max_text_seq_len:77 heads:4 dropout:0.1
                flow_steps:50 guidance:1.0 batch:200 lr:1e-4 epochs:200
loss_weights: flow:1.0 context:0.1

ALGORITHMS:
FlowMatch: x_t = t*clean + (1-t)*noise, predict v = clean - noise, loss = MSE(v_pred, v_target)
CFG: v = v_uncond + scale*(v_cond - v_uncond), x_t += v*dt
Infer: null_history, for each frame: CFG loop N steps, x_t->joints, extract features, update history

TRAINING: See src/utils/train_utils.py
Progressive Horizon Curriculum:
  Stage 1: 16 frames
  Stage 2: 32 frames
  Stage 3: 64 frames
  Stage 4: 128 frames (optional)
  
Training Loop:
  1. Sample window based on horizon (uses batch["lengths"] for padding awareness)
  2. Extract history: hist = motion[:, start_idx:end_idx]
  3. Extract target: target_frame = motion[:, end_idx]
  4. Build prev_frame_features (261D) and clean_target (72D)
  5. CFG dropout: 10% chance to drop text conditioning
  6. Encode context: encoder(text, hist) -> (B, 22, 64)
  7. Flow matching: sample t, create x_t = t*clean + (1-t)*noise
  8. Predict: predictor(context, t, x_t, prev) -> (B, 72)
  9. Loss: MSE(pred, clean - noise)

Feature Extraction Helpers:
  extract_prev_frame_features(frame:Bx271) -> (B, 261)
    - [0:9] Root: height(1) + vel(2) + rot_6d(6)
    - [9:261] Joints: 21 x 12D (RIC + rot + vel)
  extract_clean_target(frame:Bx271) -> (B, 72)
    - [0:9] Root: height(1) + vel(2) + rot_6d(6)
    - [9:72] Joint RIC: 21 x 3D

EMA Model Management:
  - EMAModel class wraps model with decay=0.999
  - Updated every training step
  - Used for validation and checkpointing
  - Checkpoints save both regular and EMA weights

Key Design Decisions:
  - Full teacher forcing (no scheduled sampling)
  - Fixed learning rate (no scheduling)
  - Simple MSE loss (no auxiliary losses)
  - CFG dropout for conditional generation capability
  - Manual stage advancement (no automatic progression)

DEPENDENCIES:
config -> (none)
models <- config, motion_utils, text_encoder
dataset <- config, motion_utils
motion_utils <- quaternion
train_utils <- models, dataset
visualization <- motion_utils

FILES: See project structure
src/config.py - hyperparameters
src/models.py - MHE, FMP, Generator
src/utils/dataset.py - Text2MotionDataset
src/utils/motion_utils.py - features, IncrementalFeatureExtractor
src/utils/train_utils.py - training loop, EMA
src/utils/quaternion.py - qrot, qmul, qinv
src/utils/text_encoder.py - CLIP encoding

TESTS: update test files on code interface change; remove previous redundant tests if new test is written
tests/test_flow_predictor.py - FlowMatchingPredictor unit tests (I/O shapes: 72D noisy, 261D prev, 72D output)
tests/test_motion_encoder.py - MotionHistoryEncoder unit tests
tests/test_nan_fix.py - HumanMotionGenerator.generate_sequence() NaN fix verification
tests/test_nan_debug.py - Debug test for tracing NaN propagation
tests/test_pose_validation.py - Pose validation unit tests
tests/**

POSE VALIDATION: See src/utils/pose_validation.py
  validate_pose(positions, check_bone_ratios, check_ric_bounds, check_kinematic_chain, report_scaling) -> PoseValidationResult
    - Comprehensive pose validation for generated motion
    - Checks bone length ratios within kinematic chains (handles scaling)
    - Validates RIC positions against dataset-derived bounds
    - Detects joint dislocations and chain discontinuities
    - Reports scaling deviation metrics
  
  validate_pose_sequence(joints, frame_threshold) -> (overall_result, per_frame_results)
    - Frame-by-frame validation of motion sequences
    - Returns aggregated and per-frame results
  
  PoseValidationResult:
    - is_valid: bool
    - bone_ratio_issues: List[BoneRatioIssue]
    - ric_bound_violations: Dict[str, List[int]]
    - kinematic_issues: List[KinematicIssue]
    - scaling_metrics: ScalingMetrics
    - summary() -> str: Human-readable summary
    - to_dict() -> Dict: For logging/serialization
  
  Usage:
    result = validate_pose(joints)  # (B, 22, 3) or (N, 22, 3)
    print(result.summary())
    if not result.is_valid:
        print(f"Found {len(result.kinematic_issues)} kinematic issues")
