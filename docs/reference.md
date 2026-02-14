PROJECT: Text-to-Motion generation
ARCH: MotionHistoryEncoder(GRU) + FlowMatchingPredictor(Transformer)
DATA: HumanML3D 263D features, 22 joints, 20fps

CONSTANTS:
motion_dim:263 num_joints:22 joint_dim:3 fps:20 max_frames:200
feature_slices: 0:4(root) 4:67(RIC pos 21x3) 67:193(RIC rot 21x6) 193:259(vel 22x3) 259:263(foot)

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
MotionHistoryEncoder: (text:512D, history:BxTx263, duration) -> context:Bx22x64
FlowMatchingPredictor: (context:Bx22x64, t:B, x_t:Bx22x3, prev:12D, progress) -> v:Bx22x3
HumanMotionGenerator: (text, num_frames, num_steps, guidance_scale) -> joints:BxTx22x3

CONFIG: See src/config.py
behavior_params: text_proj:32 joint_proj:64 model_dim:128 gru_layers:1 bidir:True
                transformer_layers:4 heads:4 dropout:0.1
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
