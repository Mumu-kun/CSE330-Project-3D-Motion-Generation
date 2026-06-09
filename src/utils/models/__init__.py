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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from utils.config import FlowMatchingPredictorConfig, MotionHistoryEncoderConfig


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
