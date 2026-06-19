"""
Model architectures for Human Motion Animation Generation.

This module contains:
- AutoregressiveContextEncoder: Encodes motion context sequentially
- FlowMatchingNetwork: Generates motion sequences using flow matching

Compatible with 271D custom feature format from motion_utils.py:
- [0:3]   Root height Y, Root velocity X, Root velocity Z (velocity form)
- [3:69]  22 RIC positions (22 * 3)
- [69:201] 22 6D rotations (22 * 6)
- [201:267] 22 local velocities (22 * 3)
- [267:271] Foot contacts (4D)

Note: Root X,Z are stored as velocities for autoregressive stability.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, OrderedDict, Tuple, TypeVar, Union
from typing import Mapping as MappingABC

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, State
from scipy import interpolate as Interp
from transformers.activations import ACT2FN

from utils.config import Config, FlowMatchingPredictorConfig, MotionHistoryEncoderConfig


class KinematicChainEncoder(nn.Module):
    """
    Encodes the kinematic hierarchy of the skeleton.
    Each joint is mapped to a unique (chain_id, depth) pair based on the T2M skeleton.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        # t2m_kinematic_chain:
        # 0: [0, 2, 5, 8, 11] (Root -> R-Leg)
        # 1: [0, 1, 4, 7, 10] (Root -> L-Leg)
        # 2: [0, 3, 6, 9, 12, 15] (Root -> Spine -> Head)
        # 3: [9, 14, 17, 19, 21] (Neck -> R-Arm)
        # 4: [9, 13, 16, 18, 20] (Neck -> L-Arm)

        joint_to_chain = [0] * 22
        joint_to_depth = [0] * 22

        # Trace and assign:
        # Chain 0: Root + Right Leg
        for d, j in enumerate([0, 2, 5, 8, 11]):
            joint_to_chain[j], joint_to_depth[j] = 0, d
        # Chain 1: Left Leg
        for d, j in enumerate([1, 4, 7, 10], 1):
            joint_to_chain[j], joint_to_depth[j] = 1, d
        # Chain 2: Spine + Head
        for d, j in enumerate([3, 6, 9, 12, 15], 1):
            joint_to_chain[j], joint_to_depth[j] = 2, d
        # Chain 3: Right Arm (starts from joint 9, depth 3)
        for d, j in enumerate([14, 17, 19, 21], 4):
            joint_to_chain[j], joint_to_depth[j] = 3, d
        # Chain 4: Left Arm (starts from joint 9, depth 3)
        for d, j in enumerate([13, 16, 18, 20], 4):
            joint_to_chain[j], joint_to_depth[j] = 4, d

        self.register_buffer("joint_to_chain", torch.tensor(joint_to_chain))
        self.register_buffer("joint_to_depth", torch.tensor(joint_to_depth))

        # Type annotations for Pylance (converts Module buffers to indexed tensors)
        self.joint_to_chain: torch.Tensor
        self.joint_to_depth: torch.Tensor

        self.chain_emb = nn.Embedding(5, model_dim // 2)
        self.depth_emb = nn.Embedding(8, model_dim // 2)

    def forward(self, joint_ids: torch.Tensor) -> torch.Tensor:
        # joint_ids: (n_joints,)
        chains = self.joint_to_chain[joint_ids]
        depths = self.joint_to_depth[joint_ids]
        return torch.cat([self.chain_emb(chains), self.depth_emb(depths)], dim=-1)  # (n_joints, model_dim)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Applies conditioning BEFORE an operation (attention or FFN)
    by modulating the normalized input with learned scale and shift.

    Following DiT (Peebles & Xie, 2023) AdaLN-Zero formulation:
      - scale, shift, gate are all predicted from the condition
      - all three are zero-initialized → identity at step 0
      - gate is applied AFTER the operation as a residual scale
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()

        # No learnable affine params — AdaLN supplies them externally
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Predicts scale (γ), shift (β), gate (α) for one sub-layer
        # Zero-init → at step 0: scale=0, shift=0, gate=0
        #   effective scale = 1 + 0 = 1 (identity norm)
        #   effective gate  = 0         (zero residual contribution)
        self.proj = nn.Linear(d_cond, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, d_model]
        cond: torch.Tensor,  # [B, d_cond]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_modulated: AdaLN(x, cond) — feed this into attention/FFN
            gate:        [B, 1, d_model] — multiply with op output
                         before adding residual
        """
        # Project condition to scale, shift, gate
        # SiLU activation on condition before projection — standard
        params = self.proj(torch.nn.functional.silu(cond))  # [B, 3*d]
        scale, shift, gate = params.chunk(3, dim=-1)  # each [B, d]

        # Unsqueeze over T for broadcasting
        scale = scale.unsqueeze(1)  # [B, 1, d_model]
        shift = shift.unsqueeze(1)  # [B, 1, d_model]
        gate = gate.unsqueeze(1)  # [B, 1, d_model]

        # Modulated norm — applied before attention/FFN
        x_modulated = self.norm(x) * (1 + scale) + shift

        # Gate returned separately — applied after attention/FFN
        return x_modulated, gate


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: Optional[int] = None,
        *,
        bias: bool,
        activation: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        if output_size is None:
            output_size = hidden_size

        self.down_proj = nn.Linear(intermediate_size, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = ACT2FN[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TemporalRoPEAttention(nn.Module):
    def __init__(self, config: MotionHistoryEncoderConfig | FlowMatchingPredictorConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.config = config
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"TemporalRoPEAttention requires an even per-head dimension, got head_dim={self.head_dim}."
            )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = config.attention_dropout

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _build_rope(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = torch.exp(
            torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / self.head_dim)
        )
        freqs = positions[:, None] * inv_freq[None, :]
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)
        return cos.view(1, 1, seq_len, self.head_dim), sin.view(1, 1, seq_len, self.head_dim)

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)


@dataclass
class TemporalLayerCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


@dataclass
class TemporalCacheState:
    layers: List[TemporalLayerCache]


def init_weights(module: nn.Module, linear_init: str = "xavier_normal", linear_std: float = 0.02) -> None:
    """Apply standard weight initialization to a module and all sub-modules.

    Args:
        module: The module to initialize (typically called as init_weights(self) from __init__).
        linear_init: Initialization scheme for nn.Linear weights.
                     Use "xavier_normal" for flow/predictor, "trunc_normal" for encoder.
        linear_std: Standard deviation for trunc_normal init (ignored for xavier_normal).
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if linear_init == "trunc_normal":
                nn.init.trunc_normal_(m.weight, std=linear_std)
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class PretrainState(State):
    """Custom Ignite State for JEPA pretraining."""

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = 40
        self.metrics: Dict[str, Any] = {
            "global_step": 0,
            "best_train_loss": float("inf"),
        }
        self.schedules = config.pre_conf.schedules
        self.num_epochs = config.get_num_epochs()

    def epoch_progress(self) -> float:
        """Return progress through current epoch as a float in [0, 1]."""
        return self.epoch / self.num_epochs if self.num_epochs > 0 else 0.0

    def get_schedule_value(self, name: str, default: float, interp: str = "previous") -> float:
        """Get current value of a scheduled parameter based on epoch progress."""
        schedule = self.schedules.get(name)

        if not schedule:
            return default

        t = self.epoch_progress()
        t_list, v_list = zip(*schedule)
        t_list = np.array(t_list)
        v_list = np.array(v_list)
        t_list = t_list / np.max(t_list)  # Normalize to [0, 1]

        f = Interp.interp1d(t_list, v_list, kind=interp, assume_sorted=True)

        return float(f(t))

    def state_dict(self) -> dict:
        """Return a dictionary containing the state of the trainer."""
        super_dict: OrderedDict = super().state_dict()
        super_dict.update(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "metrics": self.metrics,
                "schedules": self.schedules,
                "num_epochs": self.num_epochs,
                "horizon": self.horizon,
            }
        )
        return super_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the trainer from a dictionary."""
        super().load_state_dict(state_dict)
        self.epoch = state_dict.get("epoch", 0)
        self.iteration = state_dict.get("iteration", 0)
        self.metrics = state_dict.get("metrics", {})
        self.schedules = state_dict.get("schedules", {})
        self.num_epochs = state_dict.get("num_epochs", 0)
        self.horizon = state_dict.get("horizon", 40)


class PretrainEngine(Engine):
    """Custom Ignite Engine for JEPA pretraining."""

    def __init__(self, process_function: Any, config: Config) -> None:
        super().__init__(process_function)
        self.state = PretrainState(config=config)
        self.config = config

    def get_metric(self, name: str, *args, **kwargs):
        """Get a metric value by name."""
        return self.state.metrics.get(name, *args, **kwargs)

    def get_metrics(self, names: list[str], prefix: str = "") -> dict[str, Any]:
        """Get specified metrics as a dict."""
        return {f"{prefix}{name}": self.state.metrics.get(name) for name in names}

    def set_metrics(self, pairs: list[tuple[str, Any]]) -> None:
        """Set multiple metrics at once."""
        for name, value in pairs:
            self.state.metrics[name] = value

    def clear_metrics(self, names: list[str]) -> None:
        """Clear specified metrics."""
        for name in names:
            self.state.metrics.pop(name, None)

    def scale_metrics(self, names: list[str], scaler: float) -> None:
        """Scale specified metrics by a factor."""
        for name in names:
            self.state.metrics[name] *= scaler

    def csa_op_metrics(self, pairs: list[tuple[str, float]], scalers: float | list[float], clear: bool = False) -> None:
        """Add a value to an existing metric (useful for running totals)."""
        if clear:
            self.clear_metrics([name for name, _ in pairs])
        if not isinstance(scalers, list):
            scalers = [scalers for _ in pairs]
        for (name, value), scaler in zip(pairs, scalers):
            self.state.metrics[name] = self.state.metrics.get(name, 0.0) + value * scaler


def estimate_time_remaining(engine: PretrainEngine, step_time: float, config: Config) -> float:
    """Estimate remaining training time based on current progress and elapsed time."""
    curriculum = config.curriculum
    num_epochs = config.get_num_epochs()
    epoch = engine.state.epoch
    global_step = engine.get_metric("global_step")
    steps_per_epoch = global_step / epoch
    total_steps = num_epochs * steps_per_epoch
    remaining_steps = total_steps - global_step

    remaining_time = remaining_steps * step_time
    return remaining_time


T = TypeVar("T", bound=nn.Module)


class EMAModel(Generic[T]):
    """
    Exponential Moving Average model wrapper.

    Maintains an EMA copy of a model for more stable evaluation.
    EMA is used for validation, sampling, and checkpointing.
    """

    def __init__(self, model: T, decay: float = 0.999):
        """
        Initialize EMA model.

        Args:
            model: The model to create EMA copy of
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.model: T = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def update(self, model: T) -> None:
        """
        Update EMA weights.

        Args:
            model: The source model to update from
        """
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device: Union[str, torch.device]) -> "EMAModel[T]":
        """Move EMA model to device."""
        self.model.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return state dict of wrapped model for checkpointing."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: MappingABC) -> None:
        """Load state dict into wrapped model."""
        self.model.load_state_dict(state_dict)


class CheckpointMetadata:
    """Wrapper for non-stateful metadata to be saved with checkpoints.

    Ignite's Checkpoint handler requires all values in the to_save dict to have
    ``state_dict`` / ``load_state_dict`` methods. This wrapper lets us store
    simple metadata (like session_id) alongside model checkpoints without
    triggering infinite recursion in ignite's _tree_map (which would happen
    with bare strings since they are Sequences of single-char strings).
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def state_dict(self) -> dict[str, Any]:
        return self.data

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data = state_dict


def _find_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> Path | None:
    """Find the latest checkpoint file with given prefix."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)
