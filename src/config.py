"""
Configuration file for Human Motion Animation Generation Pipeline.

This configuration uses the custom 271D feature format:
- Input: 271D feature vectors from motion_utils.py
- Output: Joint positions (nframe, 22, 3) → BVH files

Feature Layout (271D) - Updated per normalization plan:
- [0:3]   Root height Y, Root velocity X, Root velocity Z (velocity form)
- [3:69]  22 RIC positions (22 * 3)
- [69:201] 22 6D rotations (22 * 6)
- [201:267] 22 local velocities (22 * 3)
- [267:271] Foot contacts (4D)

Note: Root X,Z are stored as velocities for autoregressive stability.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration class for the motion generation pipeline."""

    # Device settings
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42

    # Data paths and directories
    dataset_path: Path = Path("./dataset/humanml3d-subset")
    output_path: Path = Path("./generation")
    checkpoint_dir: Path = Path("./checkpoints")

    # Motion format settings (271D custom format from motion_utils.py)
    motion_dim: int = 271  # Custom 271D feature dimension
    num_joints: int = 22  # Number of joints in skeleton
    joint_dim: int = 3  # 3D coordinates per joint
    max_motion_length: int = 200  # Maximum motion length in frames
    fps: int = 20  # Frames per second

    # Feature dimension subsetting for training
    feature_dims: tuple = (
        slice(0, 3),  # root height Y, velocity X, velocity Z (3D)
        slice(3, 69),  # RIC positions (22*3 = 66D)
        slice(69, 201),  # 6D rotations (22*6 = 132D)
        slice(201, 267),  # local velocities (22*3 = 66D)
        slice(267, 271),  # foot contacts (4D)
    )

    # Model architecture - MotionHistoryEncoder
    text_embedding_dim: int = 512  # CLIP embedding size
    per_joint_out_dim: int = 64  # Context vector size per joint
    max_text_seq_len: int = 1  # CLIP max sequence length
    model_dim: int = 128  # Primary embedding size
    num_encoder_layers: int = 4  # Transformer layers
    dropout: float = 0.1

    # Model architecture - FlowMatchingPredictor
    num_flow_layers: int = 2  # Spatial Transformer layers
    num_heads: int = 4  # Attention heads
    time_embed_dim: int = 64  # Sinusoidal time embedding dimension

    # Training settings
    batch_size: int = 192
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    ema_decay: float = 0.999

    # CFG (Classifier-Free Guidance) settings
    cfg_dropout: float = 0.1  # Dropout probability for CFG

    # Horizon settings
    horizon: int = 40  # Maximum/target horizon for training

    # Curriculum learning settings
    curriculum_start: Optional[int] = 10  # Initial horizon (None = no curriculum)
    curriculum_step: int = 10  # Frames to increase per step
    curriculum_step_epochs: int = 150  # Epochs per horizon level

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Inference settings
    num_inference_steps: int = 20  # Number of flow matching steps
    guidance_scale: float = 1.0  # CFG scale for inference

    # Validation settings
    val_interval: int = 5  # Run validation every N epochs
    val_batches: int = 20  # Number of validation batches per run (-1 for all)
    val_use_ema: bool = True  # Use EMA models for validation
    save_best_val: bool = True  # Save separate checkpoint for best validation loss

    unit_length = 5

    def __post_init__(self):
        """Create necessary directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Export the configuration as a serializable dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()
        }
