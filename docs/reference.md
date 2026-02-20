PROJECT: Text-to-Motion generation
ARCH: MotionHistoryEncoder(ARFM Transformer) + FlowMatchingPredictor(Transformer)
DATA: HumanML3D 271D features, 22 joints, 20fps

CONSTANTS:
motion_dim:271 num_joints:22 joint_dim:3 fps:20 max_frames:200
feature_slices: 0:3(global_root) 3:69(RIC pos 22x3) 69:201(RIC rot 22x6) 201:267(vel 22x3) 267:271(foot)

271D FEATURE LAYOUT (from motion_utils.py):
[0:3]   Global root position (XYZ) - absolute position in world frame
[3:69]  RIC positions (22x3=66D) - local positions relative to root
[69:201] 6D rotations (22x6=132D) - auxiliary features from IK
[201:267] Local velocities (22x3=66D) - causal velocities (current - previous)
[267:271] Foot contacts (4D) - binary contact flags

JOINTS: See src/utils/motion_utils.py:t2m_kinematic_chain
chains: 0=L-leg[0,2,5,8,11] 1=R-leg[0,1,4,7,10] 2=spine[0,3,6,9,12,15] 3=R-arm[9,14,17,19,21] 4=L-arm[9,13,16,18,20]
face_joints:[2,1,17,16] foot_r:[8,11] foot_l:[7,10]

SKELETON: See src/utils/skeleton.py
Root(0)-+->R_Hip(1)->R_Knee(4)->R_Ankle(7)->R_Foot(10)
        +->L_Hip(2)->L_Knee(5)->L_Ankle(8)->L_Foot(11)
        +->Spine(3)->Spine1(6)->Spine2(9)-+->Spine3(12)->Neck(15)
                                        +->R_Collar(14)->R_Shoulder(17)->R_Elbow(19)->R_Wrist(21)
                                        +->L_Collar(13)->L_Shoulder(16)->L_Elbow(18)->L_Wrist(20)

MODELS: See src/models.py
MotionHistoryEncoder (ARFM Feature Fusion Transformer):
  Input: text:(B, l_seq, 512) CLIP sequence embeddings, history:BxTx271 motion features
  Architecture:
    1. Text Prefix Tokens: CLIP sequence (B, l_seq, 512) -> linear -> (B, l_seq, d_model)
    2. Global Token: per-timestep global features (16D) -> linear -> 1 token/timestep
       - root_pos(3) + root_vel(3) + root_rot_6d(6) + foot_contacts(4) = 16D
    3. Track Tokens: per-timestep local features (6D) -> linear -> add kinematic bias -> 22 tokens/timestep
       - RIC positions(3) + local velocities(3) = 6D per joint
       - Kinematic embedding added as position bias (not concatenated)
    4. Spatiotemporal Sequence: [Text Prefix (l_seq×23); Motion (T×23)] -> (B, l_seq+T, 23, d_model)
    5. Positional Encoding: Temporal (sinusoidal) + Spatial (learnable 23 vectors)
    6. Transformer Stack (4 layers): Temporal Causal Attention + Spatial Bidirectional Attention
    7. Output: Last timestep track features -> (B, 22, per_joint_out_dim)
  Output: context:Bx22x64

CLIPEncoder: See src/utils/text_encoder.py
  Input: text:str or List[str]
  Output: (B, l_seq, 512) sequence embeddings (l_seq=77 for CLIP)

FlowMatchingPredictor: (context:Bx22x64, t:B, x_t:Bx22x3, prev:12D, progress) -> v:Bx22x3
  prev_frame_features: pos(3) + rot(6) + vel(3) = 12D per joint
HumanMotionGenerator: (text, num_frames, num_steps, guidance_scale) -> joints:BxTx22x3

CONFIG: See src/config.py
behavior_params: text_proj:64 joint_proj:64 model_dim:256 transformer_layers:4
                max_text_seq_len:77 heads:4 dropout:0.1
                flow_steps:50 guidance:1.0 batch:200 lr:1e-4 epochs:200
loss_weights: flow:1.0 context:0.1

ALGORITHMS:
FlowMatch: x_t = t*clean + (1-t)*noise, predict v = clean - noise, loss = MSE(v_pred, clean)
CFG: v = v_uncond + scale*(v_cond - v_uncond), x_t += v*dt
Train: history->context, sample t, make x_t, predict v, MSE loss, EMA update
Infer: null_history, for each frame: CFG loop N steps, x_t->joints, extract features, update history

DEPENDENCIES:
config -> (none)
models <- config, motion_utils, text_encoder
dataset <- config, motion_utils
motion_utils <- quaternion, skeleton
train_utils <- models, dataset
visualization <- motion_utils

FILES: See project structure
src/config.py - hyperparameters
src/models.py - MHE, FMP, Generator
src/utils/dataset.py - Text2MotionDataset
src/utils/motion_utils.py - features, IncrementalFeatureExtractor
src/utils/train_utils.py - training loop, EMA
src/utils/quaternion.py - qrot, qmul, qinv
src/utils/skeleton.py - Skeleton IK
src/utils/text_encoder.py - CLIP encoding

TESTS: update test files on code interface change; remove previous redundant tests if new test is written
tests/**