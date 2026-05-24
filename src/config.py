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
from dataclasses import asdict, dataclass, field, is_dataclass
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
class MotionHistoryEncoderConfig:
    frame_feature_dim: int = 271
    text_embedding_dim: int = 512
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    dropout: float = 0.1
    per_joint_output_dim: int = 64
    joint_count: int = 22
    text_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "MotionHistoryEncoderConfig.hidden_size must be divisible by "
                f"num_attention_heads, got {self.hidden_size} and "
                f"{self.num_attention_heads}."
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                "MotionHistoryEncoderConfig requires an even per-head dimension "
                f"for RoPE, got hidden_size={self.hidden_size}, "
                f"num_attention_heads={self.num_attention_heads}, head_dim={head_dim}."
            )
        self.intermediate_size = max(int(self.intermediate_size), self.hidden_size)


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
    # MotionHistoryEncoder Configuration
    # =============================================================================
    encoder_config: MotionHistoryEncoderConfig = field(
        default_factory=lambda: MotionHistoryEncoderConfig(
            frame_feature_dim=271,
            text_embedding_dim=512,
            hidden_size=192,
            intermediate_size=4 * 192,
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attention_bias=True,
            attention_dropout=0.1,
            mlp_bias=True,
            dropout=0.1,
            per_joint_output_dim=64,
            joint_count=22,
            text_scale=1.0,
        )
    )

    # =============================================================================
    # FlowMatchingPredictor Configuration (Spatial-Only with Flow Matching Timestep)
    # =============================================================================
    # Uses new FlowMatchingPredictorConfig dataclass for structured configuration
    # Time embedding is handled internally via SinusoidalEmbedder(hidden_size)
    predictor_config: FlowMatchingPredictorConfig = field(
        default_factory=lambda: FlowMatchingPredictorConfig(
            hidden_size=128,
            intermediate_size=4 * 128,
            num_hidden_layers=3,
            num_attention_heads=4,
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
    learning_rate: float = 0.5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 30.0
    ema_decay: float = 0.999

    # Curriculum learning settings
    # Set to None to disable curriculum (use fixed horizon from horizon field)
    curriculum: Optional[list[dict[str, int]]] = field(
        default_factory=lambda: [
            {"horizon": 5, "epochs": 100},
            {"horizon": 10, "epochs": 200},
            {"horizon": 20, "epochs": 300},
            {"horizon": 40, "epochs": 1000},
        ]
    )

    horizon: int = 40  # Maximum/target horizon for training
    _num_epochs: int = 200

    # CFG (Classifier-Free Guidance) settings
    cfg_dropout: float = 0.1  # Dropout probability for CFG

    # Training-time timestep sampling
    t_sampling_mode: str = "power"  # "uniform" or "power"
    t_sampling_power: float = 3.0  # Power-law exponent k in p(t)=(k+1)t^k
    t_sampling_power_warmup_fraction: float = (
        1  # Fraction of training used to ramp k from 0 to target
    )

    use_fk: bool = False  # Whether to compute FK loss during training
    # Rollout scheduling settings
    rollout_prob_start: float = 0.1  # Rollout probability at first epoch
    rollout_prob_end: float = 0.3  # Rollout probability at final epoch
    rollout_warmup_fraction: float = (
        0.15  # Fraction of training with rollout disabled before schedule starts
    )
    rollout_block_len_start: int = 1  # Rollout block length at schedule start
    rollout_block_len_end: int = 4  # Rollout block length at schedule end
    rollout_integration_steps: int = (
        3  # Number of ODE integration steps for rollout branch
    )
    rollout_subset_fraction: float = 0.25  # Fraction of batch for rollout branch
    rollout_loss_weight: float = 0.25  # Weight of rollout-conditioned loss branch
    rollout_block_len_bias_power: float = (
        2.0  # Power > 1 biases sampled rollout lengths toward the scheduled max
    )

    use_consistency_loss: bool = (
        True  # Enable endpoint consistency loss after no-grad rollout
    )
    consistency_loss_t_threshold: float = (
        0.5  # Only apply consistency loss for t > threshold
    )
    consistency_loss_weight: float = 10  # Weight for consistency loss in total loss

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
    tqdm_log_per_batch: bool = False  # Show per-batch tqdm progress during training

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
        self.tqdm_log_per_batch = bool(self.tqdm_log_per_batch)
        self.rollout_warmup_fraction = min(
            max(float(self.rollout_warmup_fraction), 0.0),
            1.0 - 1e-6,
        )
        self.rollout_block_len_start = max(1, int(self.rollout_block_len_start))
        self.rollout_block_len_end = max(
            self.rollout_block_len_start,
            int(self.rollout_block_len_end),
        )
        self.rollout_subset_fraction = min(
            max(float(self.rollout_subset_fraction), 0.0),
            1.0,
        )
        self.rollout_loss_weight = max(float(self.rollout_loss_weight), 0.0)
        self.rollout_block_len_bias_power = float(self.rollout_block_len_bias_power)
        if self.rollout_block_len_bias_power <= 1.0:
            raise ValueError(
                "rollout_block_len_bias_power must be greater than 1, got "
                f"{self.rollout_block_len_bias_power}"
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

        # self.encoder_config = MotionHistoryEncoderConfig(
        #     frame_feature_dim=271,
        #     text_embedding_dim=512,
        #     hidden_size=192,
        #     intermediate_size=2 * 192,
        #     num_hidden_layers=5,
        #     num_attention_heads=8,
        #     hidden_act="silu",
        #     layer_norm_eps=1e-05,
        #     attention_bias=True,
        #     attention_dropout=0.1,
        #     mlp_bias=True,
        #     dropout=0.1,
        #     per_joint_output_dim=64,
        #     joint_count=22,
        #     text_scale=1.0,
        # )

        # self.predictor_config = FlowMatchingPredictorConfig(
        #     hidden_size=128,
        #     intermediate_size=4 * 128,
        #     num_hidden_layers=3,
        #     num_attention_heads=4,
        #     hidden_act="silu",
        #     rms_norm_eps=1e-06,
        #     attention_bias=True,
        #     attention_dropout=0.1,
        #     mlp_bias=True,
        #     track_dimensionality=3,
        #     head_dim=None,
        # )

    def get_predictor_feature_size(self) -> int:
        """
        Compute input feature size for FlowMatchingPredictor.

        For tokenized predictor inputs with shape (B, N, F), this returns F
        (the per-joint feature width), not N * F.
        """
        return self.encoder_config.per_joint_output_dim

    def to_dict(self) -> dict:
        """Export the configuration as a serializable dictionary."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value) and not isinstance(value, type):
                return {k: _convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return {k: _convert(v) for k, v in self.__dict__.items()}
