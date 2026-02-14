import sys
import os
import torch
import shutil
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from config import Config
from models import MotionHistoryEncoder, FlowMatchingPredictor, HumanMotionGenerator
from utils.train_utils import train
import traceback


class MockDataset(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        T = 60  # Fixed length
        motion = torch.randn(T, 263)
        text_clip = torch.randn(512)
        lengths = torch.tensor(T)
        captions = "a random motion"

        return {
            "motion": motion,
            "text_clip": text_clip,
            "lengths": lengths,
            "captions": captions,
        }


def mock_collate_fn(batch):
    motion = torch.stack([b["motion"] for b in batch])
    text_clip = torch.stack([b["text_clip"] for b in batch])
    lengths = torch.stack([b["lengths"] for b in batch])
    captions = [b["captions"] for b in batch]

    return {
        "motion": motion,
        "text_clip": text_clip,
        "lengths": lengths,
        "captions": captions,
    }


def test_pipeline():
    print("Setting up configuration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Minimal config for fast test
    config = Config(
        device=device,
        model_dim=32,
        num_encoder_layers=1,
        num_flow_layers=1,
        num_heads=2,
        batch_size=2,
        num_epochs=2,
        dataset_path=Path("./tests/mock_dataset"),
        output_path=Path("./tests/output"),
        checkpoint_dir=Path("./tests/checkpoints"),
        motion_dim=263,
        text_embedding_dim=512,
        num_joints=22,
    )

    # Cleanup
    if config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing models...")
    encoder = MotionHistoryEncoder(
        frame_feature_dim=config.motion_dim,
        text_embedding_dim=config.text_embedding_dim,
        joint_feature_projection_dim=config.joint_feature_projection_dim,
        text_projection_dim=config.text_projection_dim,
        per_joint_out_dim=config.per_joint_out_dim,
        joint_count=config.num_joints,
        model_dim=config.model_dim,
        num_layers=config.num_encoder_layers,
        bidirectional=config.bidirectional_gru,
    ).to(device)

    predictor = FlowMatchingPredictor(
        per_joint_dim=config.per_joint_out_dim,
        model_dim=config.model_dim,
        num_layers=config.num_flow_layers,
        joint_count=config.num_joints,
    ).to(device)

    print("Creating mock dataloader...")
    dataset = MockDataset(length=10)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=mock_collate_fn
    )

    print("Starting training loop...")
    train(
        motion_history_encoder=encoder,
        flow_predictor=predictor,
        dataloader=dataloader,
        num_epochs=config.num_epochs,
        save_dir=str(config.checkpoint_dir),
        device=device,
        # allow default save_every=1 from implementation or rely on hardcoded
    )

    print("Checking checkpoints...")
    latest_ckpt = config.checkpoint_dir / "latest.pt"
    best_ckpt = config.checkpoint_dir / "best.pt"

    if latest_ckpt.exists():
        print(f"✓ Found latest.pt at {latest_ckpt}")
    else:
        print("✗ latest.pt MISSING!")
        exit(1)

    if best_ckpt.exists():
        print(f"✓ Found best.pt at {best_ckpt}")
    else:
        print("✗ best.pt MISSING!")
        exit(1)

    print("Testing Model Loading...")
    try:
        generator = HumanMotionGenerator.load_from_checkpoint(
            latest_ckpt, config, device=device
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        exit(1)

    print("Testing Generation...")
    try:
        test_text = torch.randn(1, 512).to(device)  # Mock CLIP embedding
        output = generator.generate_sequence(
            text=test_text, num_frames=20, num_steps=5, guidance_scale=1.0
        )
        print(f"✓ Generation successful. Output shape: {output.shape}")

        expected_shape = (1, 20, 22, 3)  # (B, T, joints, 3)
        if output.shape == expected_shape:
            print("✓ Output shape correct")
        else:
            print(
                f"✗ Output shape mismatch! Expected {expected_shape}, got {output.shape}"
            )

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        print(traceback.format_exc())
        exit(1)

    print("pipeline test PASSED!")


if __name__ == "__main__":
    test_pipeline()
