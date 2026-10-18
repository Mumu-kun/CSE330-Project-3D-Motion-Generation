import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from utils.config import Config
from utils.dataset import create_dataloader
from utils.models import _find_latest_checkpoint
from utils.models.flow_matching_trainer import FlowMatchingTrainer
from utils.models.decoder_trainer import FinetuneTrainer
from utils.pipeline import generate_motion_from_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

config = Config()
config.device = device
config.dataset_path = Path("src/dataset/humanml3d-subset")
config.checkpoint_dir = Path("src/checkpoints/phase3")
config.horizon = 80
config.history_length = 40

# Load data & normalizer
val_loader, normalizer = create_dataloader(
    config=config,
    split="val",
    shuffle=False,
)

pretrain_ckpt = "src/checkpoints/pretrain/pretrain_best_val_20260620_205707_6630.pt"
dec_ckpt = "src/checkpoints/phase3/decoder/phase3_decoder_best_val_20260730_221759.pt"
pred_ckpt = "src/checkpoints/phase3/predictor/phase3_predictor_best_val_20260901_005718.pt"

decoder_trainer = FinetuneTrainer(config, pretrained_checkpoint_path=pretrain_ckpt)
decoder_trainer.load_checkpoint(dec_ckpt)

predictor_trainer = FlowMatchingTrainer(config, decoder_trainer)
predictor_trainer.load_checkpoint(pred_ckpt)

val_batch = next(iter(val_loader))

test_prompts = [
    "character imitates throwing a ball with left hand and then steps forward",
    "a person throws a ball with the left hand",
    "a person throws something forward using their left hand, turning their body while they throw",
    "a person is throwing something with their left arm.",
]

for prompt in test_prompts:
    print("\n" + "=" * 70)
    print(f"PROMPT: {prompt}")
    print("=" * 70)
    
    # Test with seed frame from val_batch
    seed_joints = val_batch["joints"][0:1, 0]
    seed_frame = val_batch["motion"][0:1, 0]
    history_zeros = torch.zeros(1, config.history_length, 271, device=device)
    
    positions = generate_motion_from_prompt(
        text_prompt=prompt,
        history_motion=history_zeros,
        initial_joints=seed_joints,
        initial_frame=seed_frame,
        predictor_trainer=predictor_trainer,
        decoder_trainer=decoder_trainer,
        normalizer=normalizer,
        clip_encoder=predictor_trainer.clip_encoder,
        device=device,
        num_inference_steps=20,
        guidance_scale=2.2,
        velocity_scale=1.05,
        horizon=80,
        history_valid_length=0,
        solver="midpoint",
        smooth_output=True,
    )
    
    # Analyze left wrist (joint 18) and left hand (joint 20)
    # vs right wrist (joint 19) and right hand (joint 21)
    lwrist = positions[:, 18] # (80, 3)
    rwrist = positions[:, 19] # (80, 3)
    
    lwrist_range = np.ptp(lwrist, axis=0) # peak-to-peak range in X, Y, Z
    rwrist_range = np.ptp(rwrist, axis=0)
    
    lwrist_vel = np.linalg.norm(np.diff(lwrist, axis=0), axis=-1)
    max_l_vel = np.max(lwrist_vel)
    
    rwrist_vel = np.linalg.norm(np.diff(rwrist, axis=0), axis=-1)
    max_r_vel = np.max(rwrist_vel)
    
    print(f"Left Wrist Range (X, Y, Z) : {lwrist_range.round(3)} | Max velocity: {max_l_vel:.3f}")
    print(f"Right Wrist Range (X, Y, Z): {rwrist_range.round(3)} | Max velocity: {max_r_vel:.3f}")
    print(f"Root position displacement : {np.ptp(positions[:, 0], axis=0).round(3)}")
