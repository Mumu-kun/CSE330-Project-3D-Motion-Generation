"""
Configuration for Human Motion Animation Generation Pipeline.

Uses the custom 271D feature format:
  Input:  271D feature vectors from motion_utils.py
  Output: Joint positions (nframe, 22, 3) → BVH files

Feature Layout (271D):
  [0:3]     Root height Y, Root velocity X, Root velocity Z
  [3:69]    22 RIC positions (22 * 3)
  [69:201]  22 6D rotations (22 * 6)
  [201:267] 22 local velocities (22 * 3)
  [267:271] Foot contacts (4D)

Note: Root X,Z are stored as velocities for autoregressive stability.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


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
    global_cond_dim: int = 512  # CLIP embedding: 512D
    head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class JepaPredictorConfig:
    hidden_size: int = 192
    intermediate_size: int = 512
    num_hidden_layers: int = 2


@dataclass
class LatentDecoderConfig:
    hidden_size: int = 512
    intermediate_size: int = 2 * 512
    dropout: float = 0.0
    num_layers: int = 4


@dataclass
class MotionHistoryEncoderConfig:
    frame_feature_dim: int = 271
    text_embedding_dim: int = 512
    hidden_size: int = 512
    intermediate_size: int = 2 * 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.1
    mlp_bias: bool = True
    dropout: float = 0.1
    joint_count: int = 22
    num_registers: int = 2
    jp_config: JepaPredictorConfig = field(default_factory=JepaPredictorConfig)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads, "
                f"got {self.hidden_size} and {self.num_attention_heads}."
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"Per-head dimension must be even for RoPE, got head_dim={head_dim} "
                f"(hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads})."
            )
        self.intermediate_size = max(int(self.intermediate_size), self.hidden_size)


@dataclass
class PretrainConfig:
    """Configuration for pretraining the motion predictor."""

    effective_batch_size: int = 400
    batch_size: int = 200
    learning_rate: float = 0.5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 30.0
    ema_decay: float = 0.999
    lr_warmup_epochs: int = 5
    lr_scheduler: str = "cosine"
    jepa_ctx_weight: float = 0.2
    cfg_dropout: float = 0.1

    schedules: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {
            "mask_num_spans": [
                (0, 2),
                (0.5, 4),
                (1, 5),
            ],
        }
    )

    # --- Masking ---
    mask_min_span: int = 5
    mask_max_span: int = 20


@dataclass
class Config:
    """Configuration for the motion generation pipeline."""

    # --- Device ---
    device: Any = "cuda"
    seed: int = 42

    # --- Paths ---
    dataset_path: Path = Path("./dataset/humanml3d-subset")
    output_path: Path = Path("./output")
    checkpoint_dir: Path = Path("./checkpoints")
    checkpoint_interval: int = 50

    # --- Motion format (271D) ---
    motion_dim: int = 271
    num_joints: int = 22
    joint_dim: int = 3
    max_motion_length: int = 200
    fps: int = 20

    # --- Sub-configs ---
    encoder_config: MotionHistoryEncoderConfig = field(default_factory=MotionHistoryEncoderConfig)
    predictor_config: FlowMatchingPredictorConfig = field(default_factory=FlowMatchingPredictorConfig)
    decoder_config: LatentDecoderConfig = field(default_factory=LatentDecoderConfig)
    text_embedding_dim: int = 512

    # --- PreTraining ---
    pre_conf: PretrainConfig = field(default_factory=PretrainConfig)
    effective_batch_size: int = 400
    batch_size: int = 200
    learning_rate: float = 0.5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 30.0
    ema_decay: float = 0.999
    lr_warmup_epochs: int = 5
    lr_scheduler: str = "cosine"
    jepa_ctx_weight: float = 0.2
    cfg_dropout: float = 0.1
    use_fk: bool = False

    # --- Curriculum ---
    curriculum: Optional[list[dict[str, int]]] = field(
        default_factory=lambda: [
            {"horizon": 5, "epochs": 100},
            {"horizon": 10, "epochs": 200},
            {"horizon": 20, "epochs": 300},
            {"horizon": 40, "epochs": 1000},
        ]
    )
    horizon: int = 40
    _num_epochs: int = 2000

    # --- Timestep sampling ---
    t_sampling_mode: str = "power"  # "uniform" or "power"
    t_sampling_power: float = 3.0
    t_sampling_power_warmup_fraction: float = 1.0

    # --- Rollout scheduling ---
    rollout_prob_start: float = 0.1
    rollout_prob_end: float = 0.3
    rollout_warmup_fraction: float = 0.15
    rollout_block_len_start: int = 1
    rollout_block_len_end: int = 4
    rollout_integration_steps: int = 3
    rollout_subset_fraction: float = 0.25
    rollout_loss_weight: float = 0.25
    rollout_block_len_bias_power: float = 2.0

    # --- Consistency loss ---
    use_consistency_loss: bool = True
    consistency_loss_t_threshold: float = 0.5
    consistency_loss_weight: float = 10.0

    # --- Data loading ---
    num_workers: int = 4
    pin_memory: bool = True

    # --- Inference ---
    num_inference_steps: int = 20
    inference_t_schedule_power: float = 3.0
    guidance_scale: float = 1.0

    # --- Validation ---
    val_interval: int = 5
    val_batches: int = 20
    val_use_ema: bool = True
    save_best_val: bool = True

    # --- Profiling ---
    enable_profiling: bool = False
    timing_log_interval: int = 100

    def __post_init__(self) -> None:
        # Timestep sampling
        self.t_sampling_mode = str(self.t_sampling_mode).lower()
        if self.t_sampling_mode not in {"uniform", "power"}:
            raise ValueError(f"t_sampling_mode must be 'uniform' or 'power', got {self.t_sampling_mode!r}")
        self.t_sampling_power = max(0.0, float(self.t_sampling_power))
        self.t_sampling_power_warmup_fraction = min(max(float(self.t_sampling_power_warmup_fraction), 0.0), 1.0)

        # Rollout
        self.rollout_warmup_fraction = min(max(float(self.rollout_warmup_fraction), 0.0), 1.0 - 1e-6)
        self.rollout_block_len_start = max(1, int(self.rollout_block_len_start))
        self.rollout_block_len_end = max(self.rollout_block_len_start, int(self.rollout_block_len_end))
        self.rollout_subset_fraction = min(max(float(self.rollout_subset_fraction), 0.0), 1.0)
        self.rollout_loss_weight = max(float(self.rollout_loss_weight), 0.0)
        self.rollout_block_len_bias_power = float(self.rollout_block_len_bias_power)
        if self.rollout_block_len_bias_power <= 1.0:
            raise ValueError(f"rollout_block_len_bias_power must be > 1, got {self.rollout_block_len_bias_power}")

        # Inference
        self.inference_t_schedule_power = float(self.inference_t_schedule_power)
        if self.inference_t_schedule_power <= 0.0:
            raise ValueError(f"inference_t_schedule_power must be positive, got {self.inference_t_schedule_power}")

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def get_num_epochs(self) -> int:
        """Return total training epochs, accounting for curriculum."""
        return self.curriculum[-1]["epochs"] if self.curriculum else self._num_epochs

    def to_dict(self) -> dict:
        """Export config as a serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                self._convert_paths_to_strings(v)
        return d

    def _convert_paths_to_strings(self, d: dict) -> None:
        """Recursively convert Path objects to strings in a dict."""
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                self._convert_paths_to_strings(v)

    def state_dict(self) -> dict:
        """Serialize for Ignite's Checkpoint handler."""
        return self.to_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        from dataclasses import fields
        from pathlib import Path

        from cattrs import Converter

        converter = Converter()
        converter.register_structure_hook(Path, lambda d, _: Path(d) if isinstance(d, str) else d)

        loaded = converter.structure(state_dict, Config)
        for f in fields(loaded):
            setattr(self, f.name, getattr(loaded, f.name))
