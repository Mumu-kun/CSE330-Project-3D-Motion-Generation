"""
Configuration file for Human Motion Animation Generation Pipeline.

This configuration is compatible with MoMask's input/output format:
- Input: HumanML3D dim-263 feature vectors
- Output: Joint positions (nframe, 22, 3) → BVH files
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for the motion generation pipeline."""
    
    # Device settings
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    
    # Data paths and directories
    dataset_path: Path = Path("./dataset/HumanML3D")
    output_path: Path = Path("./generation")
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Motion format settings (MoMask-compatible)
    motion_dim: int = 263  # HumanML3D feature dimension
    num_joints: int = 22  # Number of joints in skeleton
    joint_dim: int = 3  # 3D coordinates per joint
    max_motion_length: int = 196  # Maximum motion length in frames (rounded by 4)
    fps: int = 20  # Frames per second
    
    # Model architecture - Autoregressive Context Encoder (GRU-based)
    hidden_dim: int = 512
    num_encoder_layers: int = 3  # Number of GRU layers (typically 2-4 for GRU)
    dropout: float = 0.1
    bidirectional_gru: bool = False  # Whether to use bidirectional GRU
    
    # Model architecture - Flow Matching Network
    num_flow_layers: int = 12
    flow_hidden_dim: int = 512
    num_timesteps: int = 1000  # Number of flow matching timesteps
    
    # Training settings
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Training schedule
    warmup_steps: int = 1000
    lr_decay: float = 0.95
    lr_decay_epoch: int = 10
    
    # Loss weights
    flow_loss_weight: float = 1.0
    context_loss_weight: float = 0.1
    
    # Inference settings
    num_inference_steps: int = 50  # Number of flow matching steps during inference
    guidance_scale: float = 1.0  # For classifier-free guidance (if used)
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and checkpointing
    log_interval: int = 100  # Log every N batches
    save_interval: int = 5  # Save checkpoint every N epochs
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Evaluation settings
    num_eval_samples: int = 100
    eval_batch_size: int = 32
    
    def __post_init__(self):
        """Create necessary directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def context_encoder_output_dim(self) -> int:
        """Output dimension of the context encoder."""
        return self.hidden_dim
