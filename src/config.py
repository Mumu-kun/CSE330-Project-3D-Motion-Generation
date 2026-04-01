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
from typing import Optional, Any


@dataclass
class FlowMatchingPredictorConfig:
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    track_dimensionality: int = 3
    global_cond_dim: int = 512  # clip embedding : 512D
    head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class Config:
    """Configuration class for the motion generation pipeline."""

    # Device settings
    device: Any = "cuda"  # "cuda" or "cpu" or torch.device
    seed: int = 42

    # Data paths and directories
    dataset_path: Path = Path("./dataset/humanml3d-subset")
    output_path: Path = Path("./output")
    checkpoint_dir: Path = Path("./checkpoints")

    checkpoint_interval: int = 50  # Save checkpoint every N epochs

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

    # =============================================================================
    # MotionHistoryEncoder (GRU-based) Configuration
    # =============================================================================
    encoder_motion_dim: int = 271  # Input motion feature dimension
    encoder_text_dim: int = 512  # CLIP embedding size
    encoder_text_proj_dim: int = 128  # Text projection dimension
    encoder_hidden_dim: int = 512  # GRU hidden size
    encoder_per_joint_dim: int = 64  # Output per-joint context dimension
    encoder_num_layers: int = 3  # GRU layers
    encoder_num_joints: int = 22  # Number of joints
    encoder_text_scale: float = 1.0  # Text conditioning scale
    encoder_dropout: float = 0.1  # Dropout between GRU layers

    # =============================================================================
    # FlowMatchingPredictor Configuration (Spatial-Only with Flow Matching Timestep)
    # =============================================================================
    # Uses new FlowMatchingPredictorConfig dataclass for structured configuration
    # Time embedding is handled internally via SinusoidalEmbedder(hidden_size)
    predictor_config: FlowMatchingPredictorConfig = field(
        default_factory=lambda: FlowMatchingPredictorConfig(
            hidden_size=96,
            intermediate_size=4 * 96,
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_bias=True,
            attention_dropout=0.1,
            mlp_bias=True,
            track_dimensionality=3,
            head_dim=None,
        )
    )

    # Training settings
    batch_size: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    ema_decay: float = 0.999

    # Curriculum learning settings
    # Set to None to disable curriculum (use fixed horizon from horizon field)
    curriculum: Optional[list[dict[str, int]]] = field(
        default_factory=lambda: [
            {"horizon": 5, "epochs": 100},
            {"horizon": 10, "epochs": 200},
            {"horizon": 20, "epochs": 400},
            {"horizon": 40, "epochs": 800},
        ]
    )

    horizon: int = 40  # Maximum/target horizon for training
    _num_epochs: int = 200

    # CFG (Classifier-Free Guidance) settings
    cfg_dropout: float = 0  # Dropout probability for CFG

    # Training-time timestep sampling
    t_sampling_mode: str = "power"  # "uniform" or "power"
    t_sampling_power: float = 2.0  # Power-law exponent k in p(t)=(k+1)t^k
    t_sampling_power_warmup_fraction: float = (
        0.25  # Fraction of training used to ramp k from 0 to target
    )

    use_fk: bool = False  # Whether to compute FK loss during training
    # Rollout scheduling settings
    rollout_prob_start: float = 0.0  # Rollout probability at first epoch
    rollout_prob_end: float = 0.5  # Rollout probability at final epoch
    rollout_warmup_fraction: float = (
        0.5  # Fraction of training with rollout disabled before schedule starts
    )
    rollout_block_len_start: int = 1  # Rollout block length at schedule start
    rollout_block_len_end: int = 8  # Rollout block length at schedule end
    rollout_integration_steps: int = (
        5  # Number of ODE integration steps for rollout branch
    )
    use_consistency_loss: bool = (
        False  # Enable endpoint consistency loss after no-grad rollout
    )
    consistency_loss_weight: float = 1.0  # Weight for consistency loss in total loss
    use_degenerate_pose_guard: bool = False
    degenerate_bone_ratio_threshold: float = 0.05
    degenerate_across_norm_threshold: float = 1e-4
    degenerate_step_multiplier: float = 5.0
    degenerate_min_valid_samples_for_consistency: int = 1

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Inference settings
    num_inference_steps: int = 20  # Number of flow matching steps
    inference_t_schedule_power: float = (
        3.0  # End-bias power p in t=1-(1-s)^p for inference ODE time boundaries
    )
    guidance_scale: float = 1.0  # CFG scale for inference

    # Validation settings
    val_interval: int = 5  # Run validation every N epochs
    val_batches: int = 20  # Number of validation batches per run (-1 for all)
    val_use_ema: bool = True  # Use EMA models for validation
    save_best_val: bool = True  # Save separate checkpoint for best validation loss

    # Profiling settings
    enable_profiling: bool = False  # Enable timing instrumentation
    timing_log_interval: int = 100  # Log timings every N batches

    unit_length = 5

    def __post_init__(self):
        self.t_sampling_mode = str(self.t_sampling_mode).lower()
        if self.t_sampling_mode not in {"uniform", "power"}:
            raise ValueError(
                "t_sampling_mode must be 'uniform' or 'power', got "
                f"{self.t_sampling_mode!r}"
            )
        self.t_sampling_power = max(0.0, float(self.t_sampling_power))
        self.t_sampling_power_warmup_fraction = min(
            max(float(self.t_sampling_power_warmup_fraction), 0.0),
            1.0,
        )
        self.rollout_warmup_fraction = min(
            max(float(self.rollout_warmup_fraction), 0.0),
            1.0 - 1e-6,
        )
        self.rollout_block_len_start = max(1, int(self.rollout_block_len_start))
        self.rollout_block_len_end = max(
            self.rollout_block_len_start,
            int(self.rollout_block_len_end),
        )
        self.inference_t_schedule_power = float(self.inference_t_schedule_power)
        if self.inference_t_schedule_power <= 0.0:
            raise ValueError(
                "inference_t_schedule_power must be positive, got "
                f"{self.inference_t_schedule_power}"
            )
        self.num_epochs = (
            self.curriculum[-1]["epochs"] if self.curriculum else self._num_epochs
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def get_predictor_feature_size(self) -> int:
        """
        Compute input feature size for FlowMatchingPredictor.

        For tokenized predictor inputs with shape (B, N, F), this returns F
        (the per-joint feature width), not N * F.
        """
        return self.encoder_per_joint_dim

    def to_dict(self) -> dict:
        """Export the configuration as a serializable dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()
        }
