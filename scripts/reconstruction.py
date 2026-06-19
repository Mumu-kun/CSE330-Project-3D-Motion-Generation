import importlib.util
import json
import sys
from pathlib import Path

import torch
from ignite.handlers import Checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "build"))

# from build.utils_kaggle import Config, LatentDecoder, MotionHistoryEncoder, create_dataloader, visualize_motion

# Load build/utils_kaggle.py as a module
_spec = importlib.util.spec_from_file_location("build_utils_kaggle", PROJECT_ROOT / "build" / "utils_kaggle.py")
build_utils = importlib.util.module_from_spec(_spec)
sys.modules["build_utils_kaggle"] = build_utils
_spec.loader.exec_module(build_utils)

Config = build_utils.Config
LatentDecoder = build_utils.LatentDecoder
MotionHistoryEncoder = build_utils.MotionHistoryEncoder
MotionHistoryEncoderConfig = build_utils.MotionHistoryEncoderConfig
create_dataloader = build_utils.create_dataloader
visualize_motion = build_utils.visualize_motion


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "test_jepa_pretrain"
CHECKPOINT_NAME = "pretrain_best_eval_20260612_000732_9.pt"
SAVE_PATH = PROJECT_ROOT / "output" / "_".join(CHECKPOINT_NAME.split("_")[0:-1])

config = Config()
# config.load_state_dict(config_dict)
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load checkpoint into the instantiated object
Checkpoint.load_objects(
    to_load={"config": config}, checkpoint=CHECKPOINT_PATH / CHECKPOINT_NAME, map_location=config.device
)

ckpt = torch.load(CHECKPOINT_PATH / CHECKPOINT_NAME, map_location=config.device)

print(ckpt)

exit()

print(json.dumps(config.to_dict(), indent=2))

print(config.encoder_config.frame_feature_dim)

ema_encoder = MotionHistoryEncoder(config.encoder_config)
decoder = LatentDecoder(config)

ema_encoder_state_dict = {}
decoder_state_dict = {}

Checkpoint.load_objects(
    to_load={"encoder": ema_encoder_state_dict, "decoder": decoder_state_dict},
    checkpoint=CHECKPOINT_PATH / CHECKPOINT_NAME,
)

exit()

ema_encoder.load_state_dict(ema_encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)


val_dataloader, normalizer = create_dataloader(config, split="val")

batch = next(iter(val_dataloader))
_motion = batch["motion"].to(config.device)
_joints = batch["joints"].to(config.device)
random_sample = torch.randint(0, _motion.shape[0], (1,))
motion = _motion[random_sample]
with torch.inference_mode():
    # Perform reconstruction or other operations
    prev_pos = _joints[random_sample][:, :-1]
    prev_poss = torch.cat([torch.zeros_like(prev_pos[:, :1]), prev_pos], dim=1).flatten(
        0, 1
    )  # Add zero for the first frame
    prev_frame = motion[:, :1]
    prev_frames = torch.stack([prev_frame, motion[:, :-1]], dim=1).flatten(0, 1)  # (1*seq_len, motion_dim)

    latents = ema_encoder(motion).flatten(0, 1)  # (1*seq_len, latent_dim)
    new_pos, rel_shift, _ = decoder.decode(
        latents, prev_pos=prev_poss, prev_frame=prev_frames, normalizer=normalizer
    )  # (1*seq_len, 22, 3), (1*seq_len, 22, 3), (1*seq_len, motion_dim)


joints = _joints[random_sample].cpu().numpy()[0]
positions = new_pos.numpy()
visualize_motion(
    joints,
    save_path=SAVE_PATH / "gt.mp4",
)
visualize_motion(
    positions,
    save_path=SAVE_PATH / "recon.mp4",
)
