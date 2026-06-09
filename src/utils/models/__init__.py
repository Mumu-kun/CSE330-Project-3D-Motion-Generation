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

import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Union, cast
from .motion_history_encoder import (
    MotionHistoryEncoder,
    GatedMLP,
    TemporalCacheState,
)
from utils.config import Config, FlowMatchingPredictorConfig, MotionHistoryEncoderConfig
from transformers.activations import ACT2FN

from utils.motion_utils import (
    get_fk_offsets,
    sequence_joints_to_features,
    generated_positions_to_271d,
    FeatureNormalizer,
    extract_prev_frame_features,
    flow_output_to_positions,
)


@contextmanager
def _windows_checkpoint_path_compat():
    """Allow checkpoints pickled with PosixPath to load on Windows."""
    original_posix_path = pathlib.PosixPath
    should_patch_posix = os.name == "nt"
    if should_patch_posix:
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        if should_patch_posix:
            pathlib.PosixPath = original_posix_path


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
        return torch.cat(
            [self.chain_emb(chains), self.depth_emb(depths)], dim=-1
        )  # (n_joints, model_dim)


class SinusoidalEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        if half == 0:
            return torch.zeros((t.shape[0], dim), device=t.device, dtype=torch.float32)

        max_period_tensor = torch.tensor(
            max_period, device=t.device, dtype=torch.float32
        )
        freqs = torch.exp(
            -torch.log(max_period_tensor)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        out = self.mlp(t_freq)
        return out


class SpatialTrackMLP(GatedMLP):
    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation="silu",
            dropout=0.0,
        )


class SpatialTrackLayer(nn.Module):
    """
    Simplified transformer layer with AdaLN conditioning using PyTorch-native MultiheadAttention
    """

    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # PyTorch-native MultiheadAttention (batch_first=True for (B, N, H) format)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            bias=config.attention_bias,
            batch_first=True,  # Uses (B, N, H) format
        )

        # Match the denoising predictor MLP behavior (gated + configurable activation).
        self.mlp = SpatialTrackMLP(config)

        # Normalization layers
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # AdaLN modulation (core innovation preserved)
        self.adaln_linear = nn.Linear(
            config.hidden_size, 6 * config.hidden_size, bias=True
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            self.adaln_linear,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N, H)
        adaln_conditioning: torch.Tensor,  # (B, H) - made required
        output_attentions: bool = False,
        **kwargs,  # Ignore attention_mask, position_ids, etc.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Store residual for first connection
        residual = hidden_states

        # AdaLN modulation for attention block
        adaln_out = self.adaln_modulation(adaln_conditioning)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaln_out.chunk(6, dim=-1)
        )
        # Broadcasting: (B,H) -> (B,1,H) -> (B,N,H) works automatically

        # Pre-attention normalization + AdaLN
        normed = self.input_layernorm(hidden_states)
        normed = normed * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # PyTorch MultiheadAttention - NO MASKING
        attn_output, attn_weights = self.self_attn(
            query=normed, key=normed, value=normed, need_weights=output_attentions
        )

        # Output gating + residual connection
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = residual + attn_output

        # Store residual for second connection
        residual = hidden_states

        # Post-attention normalization + AdaLN
        normed = self.post_attention_layernorm(hidden_states)
        normed = normed * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        # MLP
        mlp_output = self.mlp(normed)
        mlp_output = gate_mlp.unsqueeze(1) * mlp_output
        hidden_states = residual + mlp_output

        if output_attentions:
            return hidden_states, attn_weights
        else:
            return hidden_states, None


class FlowMatchingPredictor(nn.Module):
    """
    Flow Matching Predictor - Simplified DenoisingJointTrackPredictor

    Designed for:
    - T=1 (single frame prediction)
    - External preprocessing (positions, shifts, etc. provided externally)
    - No attention masking (all tracks assumed valid)
    - PyTorch-native transformer components
    - Preserves AdaLN conditioning mechanism for rectified flow

    Predicts velocity field for rectified flow-based motion prediction
    """

    def __init__(
        self,
        feature_size: int,  # Size of track features per track (from encoder/detector)
        config: FlowMatchingPredictorConfig,  # Model configuration
        out_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        del out_channels, kwargs
        self.feature_size = feature_size
        self.root_state_dim = 5
        self.joint_state_dim = 3
        self.root_frame_dim = 5
        self.joint_frame_dim = 12
        self.joint_count = 22
        self.non_root_joint_count = self.joint_count - 1
        self.flow_dim = self.root_state_dim + (
            self.non_root_joint_count * self.joint_state_dim
        )
        self.current_frame_feature_dim = self.root_frame_dim + (
            self.non_root_joint_count * self.joint_frame_dim
        )

        self.root_input_projection = nn.Linear(
            self.root_state_dim + feature_size + self.root_frame_dim,
            config.hidden_size,
        )
        self.joint_input_projection = nn.Linear(
            self.joint_state_dim + feature_size + self.joint_frame_dim,
            config.hidden_size,
        )

        self.global_adaln_projection = nn.Linear(
            config.global_cond_dim, config.hidden_size
        )

        self.global_film_projection = nn.Linear(
            config.global_cond_dim, 2 * config.hidden_size, bias=True
        )

        self.text_token_projection = nn.Linear(
            config.global_cond_dim, config.hidden_size, bias=True
        )

        # Time embedding for denoising timestep
        self.time_embedder = SinusoidalEmbedder(config.hidden_size)

        # Kinematic chain embedding
        self.kinematic_encoder = KinematicChainEncoder(config.hidden_size)
        self.register_buffer("joint_ids", torch.arange(22, dtype=torch.long))  # (22,)
        self.kinematic_token_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # Stage-2 structural fusion: layer-wise scalar gates, initialized to no-op.
        self.structural_layer_gates = nn.Parameter(
            torch.zeros(config.num_hidden_layers, dtype=torch.float32)
        )

        # Transformer layers with AdaLN
        self.layers = nn.ModuleList(
            [SpatialTrackLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Output prediction head
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.output_adaln_linear = nn.Linear(
            config.hidden_size, 2 * config.hidden_size, bias=True
        )
        self.output_adaln = nn.Sequential(
            nn.SiLU(),
            self.output_adaln_linear,
        )
        self.root_output_head = nn.Linear(config.hidden_size, self.root_state_dim)
        self.joint_output_head = nn.Linear(config.hidden_size, self.joint_state_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP.
        nn.init.normal_(cast(nn.Linear, self.time_embedder.mlp[0]).weight, std=0.02)
        nn.init.normal_(cast(nn.Linear, self.time_embedder.mlp[2]).weight, std=0.02)
        # Zero-out adaln modulation layers in DiT blocks:
        for layer in self.layers:
            if isinstance(layer, SpatialTrackLayer):
                nn.init.constant_(layer.adaln_linear.weight, 0)
                if layer.adaln_linear.bias is not None:
                    nn.init.constant_(layer.adaln_linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.output_adaln_linear.weight, 0)
        nn.init.constant_(self.output_adaln_linear.bias, 0)
        nn.init.constant_(self.root_output_head.weight, 0)
        nn.init.constant_(self.root_output_head.bias, 0)
        nn.init.constant_(self.joint_output_head.weight, 0)
        nn.init.constant_(self.joint_output_head.bias, 0)

    def _split_noisy_features(
        self, noisy_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noisy_features.ndim != 2 or noisy_features.shape[-1] != self.flow_dim:
            raise ValueError(
                f"Expected noisy_features shape (B, {self.flow_dim}), got {tuple(noisy_features.shape)}"
            )
        root_state = noisy_features[:, : self.root_state_dim]
        joint_state = noisy_features[:, self.root_state_dim :].reshape(
            noisy_features.shape[0], self.non_root_joint_count, self.joint_state_dim
        )
        return root_state, joint_state

    def _split_current_frame_features(
        self, current_frame_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            current_frame_features.ndim != 2
            or current_frame_features.shape[-1] != self.current_frame_feature_dim
        ):
            raise ValueError(
                "Expected current_frame_features shape "
                f"(B, {self.current_frame_feature_dim}), got {tuple(current_frame_features.shape)}"
            )
        root_features = current_frame_features[:, : self.root_frame_dim]
        joint_features = current_frame_features[:, self.root_frame_dim :].reshape(
            current_frame_features.shape[0],
            self.non_root_joint_count,
            self.joint_frame_dim,
        )
        return root_features, joint_features

    def forward(
        self,
        noisy_features: torch.Tensor,  # (B, 68) - noised reduced-state features
        timesteps: torch.Tensor,  # (B,) or (B, 1) - denoising timesteps in [0,1]
        text_embedding: torch.Tensor,  # (B, F) - Text embedding for global conditioning
        track_features: torch.Tensor,  # (B, 22, F) - per-joint features from MotionHistoryEncoder
        current_frame_features: torch.Tensor,  # (B, 257) - current frame state features
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,  # Ignore attention_mask, position_ids, etc.
    ) -> tuple[
        torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]
    ]:
        """
        Forward pass of the Flow Matching Predictor

        Args:
            track_features: (B, 22, F) - Per-joint context features from MotionHistoryEncoder
            noisy_features: (B, 68) - Noised reduced-state motion features to predict flow for
            timesteps: (B,) or (B, 1) - Denoising timesteps t in [0,1]
            current_frame_features: (B, 257) - Current frame root/joint conditioning features
            text_embedding: (B, F) - Text embedding for global conditioning
            output_attentions: Whether to return attention weights (not implemented in this simplified version)
            output_hidden_states: Whether to return hidden states (not implemented in this simplified version)

        Returns:
            flow_prediction: (B, 68) - Predicted velocity field for denoising
        """

        if track_features.ndim != 3 or track_features.shape[1] != self.joint_count:
            raise ValueError(
                f"Expected track_features shape (B, {self.joint_count}, F), got {tuple(track_features.shape)}"
            )
        if track_features.shape[0] != noisy_features.shape[0]:
            raise ValueError(
                "Batch size mismatch between noisy_features and track_features: "
                f"{tuple(noisy_features.shape)} vs {tuple(track_features.shape)}"
            )
        if current_frame_features.shape[0] != noisy_features.shape[0]:
            raise ValueError(
                "Batch size mismatch between noisy_features and current_frame_features: "
                f"{tuple(noisy_features.shape)} vs {tuple(current_frame_features.shape)}"
            )

        root_state, joint_state = self._split_noisy_features(noisy_features)
        root_current, joint_current = self._split_current_frame_features(
            current_frame_features
        )
        root_track_features = track_features[:, :1, :]
        joint_track_features = track_features[:, 1:, :]

        root_inputs = torch.cat(
            [root_state.unsqueeze(1), root_track_features, root_current.unsqueeze(1)],
            dim=-1,
        )
        joint_inputs = torch.cat(
            [joint_state, joint_track_features, joint_current],
            dim=-1,
        )

        root_hidden = self.root_input_projection(root_inputs)
        joint_hidden = self.joint_input_projection(joint_inputs)
        hidden_states = torch.cat([root_hidden, joint_hidden], dim=1)  # (B, 22, H)

        text_token = self.text_token_projection(text_embedding).unsqueeze(
            1
        )  # (B, 1, H)
        hidden_states = torch.cat([text_token, hidden_states], dim=1)  # (B, 23, H)

        # Build static per-joint kinematic tokens once and reuse across layers.
        B, N, H = hidden_states.shape
        expected_token_count = self.joint_count + 1
        if N != expected_token_count:
            raise ValueError(
                f"Expected hidden_states to contain text token + {self.joint_count} joint tokens, got {N} tokens."
            )
        kin_tokens = self.kinematic_encoder(self.joint_ids)  # (22, H)
        if kin_tokens.shape[0] != self.joint_count:
            raise ValueError(
                f"Joint count mismatch: predictor expected {self.joint_count} joints, got {kin_tokens.shape[0]}."
            )
        kin_tokens = self.kinematic_token_norm(kin_tokens)
        kin_tokens = kin_tokens.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        kin_tokens = torch.cat(
            [
                torch.zeros((1, H), device=kin_tokens.device, dtype=kin_tokens.dtype),
                kin_tokens,
            ],
            dim=0,
        )  # Add zero token for text/global conditioning
        kin_tokens = kin_tokens.unsqueeze(0).expand(
            B, expected_token_count, H
        )  # (B, joint_count + 1, H)

        # 3. Time conditioning
        # Handle both (B,) and (B, 1) timestep formats
        time_cond = self.time_embedder(
            timesteps.squeeze(-1) if timesteps.dim() > 1 else timesteps
        )  # (B, H)

        global_cond_proj = self.global_adaln_projection(text_embedding)  # (B, H)
        global_film = self.global_film_projection(text_embedding)  # (B, 2H)
        global_film_shift, global_film_scale = global_film.chunk(2, dim=-1)
        hidden_states = hidden_states * (
            1 + global_film_scale.unsqueeze(1)
        ) + global_film_shift.unsqueeze(1)

        # Combine required global conditioning with time embedding.
        adaln_conditioning = time_cond + global_cond_proj  # (B, H)

        # 4. Transformer processing (NO ATTENTION MASKING)
        all_hidden_states: list[torch.Tensor] = []
        all_self_attns: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.layers):
            # Bounded signed gate allows add/subtract structural prior per layer.
            layer_gate = torch.tanh(self.structural_layer_gates[layer_idx]).to(
                dtype=hidden_states.dtype
            )
            hidden_states = hidden_states + layer_gate * kin_tokens

            layer_outputs = layer(
                hidden_states,
                adaln_conditioning=adaln_conditioning,
                output_attentions=output_attentions,
                # attention_mask, position_ids, etc. all ignored as per requirements
            )

            hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if output_attentions:
                all_self_attns.append(layer_outputs[1])

        hidden_states = hidden_states[
            :, 1:, :
        ]  # Remove text/global token before output

        # 5. Output prediction
        normed_states = self.output_norm(hidden_states)  # (B, N, H)
        shift, scale = self.output_adaln(adaln_conditioning).chunk(2, dim=-1)
        normed_states = normed_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        root_prediction = self.root_output_head(normed_states[:, 0])
        joint_prediction = self.joint_output_head(normed_states[:, 1:]).reshape(
            B, self.non_root_joint_count * self.joint_state_dim
        )
        flow_prediction = torch.cat([root_prediction, joint_prediction], dim=-1)

        return (
            flow_prediction,
            all_hidden_states if output_hidden_states else None,
            all_self_attns if output_attentions else None,
        )


def _build_inference_time_boundaries(
    num_steps: int,
    *,
    power: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build end-biased ODE boundaries on [0, 1] using t = 1 - (1 - s)^p."""
    steps = max(1, int(num_steps))
    s = torch.linspace(0.0, 1.0, steps=steps + 1, device=device, dtype=dtype)
    return 1.0 - (1.0 - s).pow(power)


def integrate_flow_ode(
    *,
    predictor: FlowMatchingPredictor,
    track_features: torch.Tensor,
    current_frame_features: torch.Tensor,
    text_embedding: torch.Tensor,
    num_steps: int,
    time_schedule_power: float = 2.0,
    initial_state: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    unconditional_text_embedding: Optional[torch.Tensor] = None,
    unconditional_track_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Integrate the inference-time flow ODE using an end-biased power grid and Heun.
    """
    batch_size = track_features.shape[0]
    device = track_features.device
    dtype = track_features.dtype
    if time_schedule_power <= 0.0:
        raise ValueError(
            "time_schedule_power must be positive, got " f"{time_schedule_power}"
        )

    if initial_state is None:
        x_t = torch.randn(
            (batch_size, predictor.flow_dim),
            device=device,
            dtype=dtype,
        )
    else:
        if initial_state.shape != (batch_size, predictor.flow_dim):
            raise ValueError(
                "Expected initial_state shape "
                f"({batch_size}, {predictor.flow_dim}), got {tuple(initial_state.shape)}"
            )
        x_t = initial_state.to(device=device, dtype=dtype)

    tau = _build_inference_time_boundaries(
        num_steps,
        power=float(time_schedule_power),
        device=device,
        dtype=dtype,
    )

    use_cfg = float(guidance_scale) != 1.0 and (
        unconditional_text_embedding is not None
        or unconditional_track_features is not None
    )
    if unconditional_text_embedding is not None:
        if unconditional_text_embedding.shape != text_embedding.shape:
            raise ValueError(
                "Expected unconditional_text_embedding shape "
                f"{tuple(text_embedding.shape)}, got "
                f"{tuple(unconditional_text_embedding.shape)}"
            )
        unconditional_text_embedding = unconditional_text_embedding.to(
            device=device,
            dtype=text_embedding.dtype,
        )
    if unconditional_track_features is not None:
        if unconditional_track_features.shape != track_features.shape:
            raise ValueError(
                "Expected unconditional_track_features shape "
                f"{tuple(track_features.shape)}, got "
                f"{tuple(unconditional_track_features.shape)}"
            )
        unconditional_track_features = unconditional_track_features.to(
            device=device,
            dtype=track_features.dtype,
        )

    def _predict_velocity(
        noisy_features: torch.Tensor,
        timesteps_batch: torch.Tensor,
    ) -> torch.Tensor:
        cond_velocity = predictor(
            track_features=track_features,
            noisy_features=noisy_features,
            timesteps=timesteps_batch,
            current_frame_features=current_frame_features,
            text_embedding=text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        if not use_cfg:
            return cond_velocity

        cfg_text_embedding = (
            unconditional_text_embedding
            if unconditional_text_embedding is not None
            else text_embedding
        )
        cfg_track_features = (
            unconditional_track_features
            if unconditional_track_features is not None
            else track_features
        )
        uncond_velocity = predictor(
            track_features=cfg_track_features,
            noisy_features=noisy_features,
            timesteps=timesteps_batch,
            current_frame_features=current_frame_features,
            text_embedding=cfg_text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        return uncond_velocity + float(guidance_scale) * (
            cond_velocity - uncond_velocity
        )

    for step in range(tau.shape[0] - 1):
        t_start = tau[step]
        t_end = tau[step + 1]
        dt = t_end - t_start

        t_start_batch = t_start.expand(batch_size)
        k1 = _predict_velocity(x_t, t_start_batch)

        x_euler = x_t + dt * k1
        t_end_batch = t_end.expand(batch_size)
        k2 = _predict_velocity(x_euler, t_end_batch)

        x_t = x_t + 0.5 * dt * (k1 + k2)

    return x_t


class HumanMotionGenerator:
    """
    Top-level wrapper for the Human Motion Generation pipeline.
    Integrates MotionHistoryEncoder (Context) and FlowMatchingPredictor (Spatial Generation).

    Updated to use 271D features and proper reduced-state → 271D conversion for autoregressive generation.
    """

    def __init__(
        self,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
        config: Config,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalizer = normalizer
        self.config = config

    def eval(self) -> "HumanMotionGenerator":
        """Set models to evaluation mode."""
        self.encoder.eval()
        self.predictor.eval()
        return self

    def train(self, mode: bool = True) -> "HumanMotionGenerator":
        """Set models to training mode."""
        self.encoder.train(mode)
        self.predictor.train(mode)
        return self

    def parameters(self):
        """Yield parameters from both encoder and predictor."""
        for p in self.encoder.parameters():
            yield p
        for p in self.predictor.parameters():
            yield p

    def to(self, device):
        """Move models to device."""
        self.encoder = self.encoder.to(device)
        self.predictor = self.predictor.to(device)
        return self

    def generate_sequence(
        self,
        text: Union[str, List[str], torch.Tensor],  # type: ignore
        num_frames: int = 200,
        num_steps: int = 10,
        horizon: int | None = None,
        input_positions: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        guidance_drop_text: bool = True,
        guidance_drop_context: bool = False,
        dataset_type: str = "t2m",
        use_fk=True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate n consecutive animation frames autoregressively.

        Uses absolute global joint positions as the autoregressive state and
        generated_positions_to_271d for incremental feature extraction. This ensures
        O(n) complexity and Markov-safe autoregressive generation.

        Args:
            text: Text prompt(s) - str, List[str], or pre-encoded tensor (B, 1, 512)
            num_frames: Number of frames to generate
            num_steps: Number of flow matching ODE steps
            horizon: Number of recent frames to use for context encoding
            input_positions: Optional initial global positions (B, N, 22, 3) or (B, 22, 3)
            total_duration: Optional duration tensor (not used)
            guidance_scale: CFG strength used during inference
            guidance_drop_text: Zero the predictor text embedding in the CFG branch
            guidance_drop_context: Zero the encoder context in the CFG branch
            dataset_type: Dataset type for feature extraction
            use_fk: Whether to use FK positions for relative shift computation
        Returns:
            position_history: (B, N+num_frames, 22, 3) - Absolute global joint positions including initial history
            feature_history: (B, N+num_frames, 271) - 271D features derived from position history
            relative_shift_history: (B, N+num_frames, 22, 3) - Relative shifts between frames
        """
        self.eval()
        with torch.no_grad():
            # Handle text encoding
            if isinstance(text, str):
                # Single string - encode and use
                from utils.text_encoder import CLIPEncoder

                clip_encoder = CLIPEncoder()
                text = clip_encoder(text)  # (1, 1, 512)
                B = 1
            elif isinstance(text, list):
                # List of strings - encode all
                from utils.text_encoder import CLIPEncoder

                clip_encoder = CLIPEncoder()
                text: torch.Tensor = clip_encoder(text)  # (B, 1, 512)
                B = text.shape[0]
            else:
                # Already a tensor
                if text.ndim != 3 or text.shape[1] != 1:
                    raise ValueError(
                        f"Pre-encoded text must have shape (B, 1, 512); got {tuple(text.shape)}"
                    )
                B = text.shape[0]
            device = next(self.parameters()).device
            text = text.to(device=device)

            # ========================================
            # History Initialization
            # ========================================
            # Keep absolute global positions as the primary AR state.
            # Feature history is derived incrementally from position history.

            if input_positions is None:
                # Cold start from a zero pose frame.
                joint_count = int(self.config.num_joints)
                joint_dim = int(self.config.joint_dim)
                position_history = torch.zeros(
                    (B, 1, joint_count, joint_dim),
                    device=device,
                )
                feature_history = sequence_joints_to_features(
                    position_history, dataset_type=dataset_type
                )  # (B, 1, 271)
            else:
                input_positions = input_positions.to(device=device)
                if input_positions.ndim == 3:
                    # Single frame (B, N, 3)
                    seed_positions = input_positions.unsqueeze(
                        1
                    ).clone()  # (B, 1, N, 3)
                elif input_positions.ndim == 4:
                    # Sequence (B, T, N, 3)
                    seed_positions = input_positions.clone()  # (B, T, N, 3)
                else:
                    raise ValueError(
                        "input_positions must be shape (B, N, 3) or (B, T, N, 3)"
                    )

                position_history = seed_positions
                # Seeded: convert global positions to 271D feature history
                feature_history = sequence_joints_to_features(
                    seed_positions, dataset_type=dataset_type
                )  # (B, T, 271)

            fk_offsets = (
                get_fk_offsets(position_history) if use_fk else None
            )  # (B, 22, 3)

            feature_history = (
                self.normalizer.normalize(feature_history)
                if self.normalizer is not None
                else feature_history
            )

            relative_shift_history = torch.zeros(
                (B, 1, self.config.num_joints, self.config.joint_dim),
                device=device,
            )

            if position_history.shape[1] > 1:
                relative_shift_history = torch.cat(
                    [
                        relative_shift_history,
                        position_history[:, 1:] - position_history[:, :-1],
                    ],
                    dim=1,
                )

            text_emb = text[:, 0, :]
            use_cfg = float(guidance_scale) != 1.0
            if use_cfg and not (guidance_drop_text or guidance_drop_context):
                raise ValueError(
                    "guidance_scale requires at least one unconditional branch input; "
                    "set guidance_drop_text and/or guidance_drop_context."
                )
            predictor_uncond_text_emb = (
                torch.zeros_like(text_emb) if use_cfg and guidance_drop_text else None
            )
            frame_buffer = (
                feature_history[:, :-1] if feature_history.shape[1] > 1 else None
            )
            cache_state: Optional[TemporalCacheState] = None

            for frame_idx in range(num_frames):
                # ========================================
                # Step A: Extract last frame from position history
                # ========================================
                current_positions = position_history[:, -1]  # (B, 22, 3)
                current_frame = feature_history[:, -1]

                # ========================================
                # Step B: Encode context from last horizon frames
                # Feature history buffer is maintained outside the encoder.
                # ========================================
                if frame_buffer is not None:
                    if horizon is not None:
                        frame_buffer = frame_buffer[:, -horizon - 1 :]

                context_cond, frame_buffer, cache_state = self.encoder.step(
                    current_frame,
                    text_emb,
                    frame_buffer=frame_buffer,
                    cache_state=cache_state,
                )
                predictor_uncond_context = (
                    torch.zeros_like(context_cond)
                    if use_cfg and guidance_drop_context
                    else None
                )
                current_frame_features = extract_prev_frame_features(
                    current_frame,
                    normalizer=self.normalizer,
                    normalize_output=self.normalizer is not None,
                )

                # ========================================
                # Step C: Flow matching ODE loop
                # x_t starts as random noise in normalized reduced flow space.
                # ========================================
                x_t = integrate_flow_ode(
                    predictor=self.predictor,
                    track_features=context_cond,
                    current_frame_features=current_frame_features,
                    text_embedding=text_emb,
                    num_steps=num_steps,
                    time_schedule_power=self.config.inference_t_schedule_power,
                    guidance_scale=guidance_scale,
                    unconditional_text_embedding=predictor_uncond_text_emb,
                    unconditional_track_features=predictor_uncond_context,
                )

                # ========================================
                # Step E: Convert reduced-state prediction -> positions -> 271D (incremental)
                # ========================================
                flow_output_raw = (
                    self.normalizer.denormalize_flow_output(x_t)
                    if self.normalizer is not None
                    else x_t
                )
                current_frame_raw = (
                    self.normalizer.denormalize(current_frame)
                    if self.normalizer is not None
                    else current_frame
                )
                new_positions = flow_output_to_positions(
                    flow_output_raw,
                    prev_root_pos=current_positions[:, 0],
                    prev_root_rot_6d=current_frame_raw[:, 69:75],
                )
                relative_shift = new_positions - current_positions
                new_frame, _, fk_positions = generated_positions_to_271d(
                    new_positions=new_positions,
                    prev_positions=current_positions,
                    dataset_type=dataset_type,
                    normalizer=self.normalizer,
                    fk_offsets=fk_offsets,
                )  # (B, 271), (B, 3), (B, 22, 3)

                if fk_positions is not None:
                    new_positions = fk_positions  # Override with FK-corrected positions if available
                    relative_shift = (
                        new_positions - current_positions
                    )  # Recompute relative shift after FK correction

                # ========================================
                # Step F: Update tracker and history
                # ========================================
                position_history = torch.cat(
                    [position_history, new_positions.unsqueeze(1)], dim=1
                )  # (B, T+1, 22, 3)
                feature_history = torch.cat(
                    [feature_history, new_frame.unsqueeze(1)], dim=1
                )  # (B, N+1, 271)
                relative_shift_history = torch.cat(
                    [relative_shift_history, relative_shift.unsqueeze(1)], dim=1
                )  # (B, T, 22, 3)

                if (frame_idx + 1) % 50 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            return position_history, feature_history, relative_shift_history

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Config,  # type: ignore
        device: str = "cpu",
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> "HumanMotionGenerator":
        """
        Load the generator from a checkpoint file.
        Prefers EMA weights if available.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Config object with model configuration
            device: Device to load model on
            normalizer: Optional FeatureNormalizer for raw feature normalization
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        with _windows_checkpoint_path_compat():
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

        if "config" in checkpoint:
            config: Config = checkpoint["config"]

        encoder = MotionHistoryEncoder(config.encoder_config).to(device)

        # Initialize Flow Matching Predictor with new config-based interface
        predictor_config = config.predictor_config
        feature_size = config.get_predictor_feature_size()

        predictor = FlowMatchingPredictor(
            feature_size=feature_size,
            config=predictor_config,
        ).to(device)

        # Load weights (Prefer EMA)
        if "encoder_ema" in checkpoint and "predictor_ema" in checkpoint:
            print("Loading EMA weights for generation...")
            encoder.load_state_dict(checkpoint["encoder_ema"])
            predictor.load_state_dict(checkpoint["predictor_ema"])
        else:
            print("Loading standard weights (EMA not found)...")
            encoder.load_state_dict(checkpoint["encoder"])
            predictor.load_state_dict(checkpoint["predictor"])

        encoder.to(device)
        predictor.to(device)
        encoder.eval()
        predictor.eval()

        return cls(encoder, predictor, config, normalizer=normalizer)


# Export all public symbols defined in this module.
__all__ = sorted(  # pyright: ignore[reportUnsupportedDunderAll]
    name
    for name, obj in globals().items()
    if not name.startswith("_") and getattr(obj, "__module__", None) == __name__
)
