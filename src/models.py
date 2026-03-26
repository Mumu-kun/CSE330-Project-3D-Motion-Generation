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

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple, Union, cast
from config import Config, FlowMatchingPredictorConfig
from transformers.activations import ACT2FN

from utils.motion_utils import (
    get_fk_offsets,
    sequence_joints_to_features,
    generated_positions_to_271d,
    FeatureNormalizer,
)


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


class MotionHistoryEncoder(nn.Module):
    def __init__(
        self,
        frame_feature_dim: int,  # e.g. 271
        text_embedding_dim: int,  # e.g. 512
        text_proj_dim: int,  # e.g. 128
        model_dim: int,  # GRU hidden size H
        per_joint_out_dim: int,  # D_joint, must match FlowMatchingPredictor model_dim
        num_layers: int = 2,
        joint_count: int = 22,
        text_scale: float = 1.0,
        dropout: float = 0.0,  # dropout between GRU layers if num_layers > 1
        normalizer: Optional[
            FeatureNormalizer
        ] = None,  # Feature normalizer for normalization
    ) -> None:
        super().__init__()

        self.frame_feature_dim = frame_feature_dim
        self.text_embedding_dim = text_embedding_dim
        self.text_proj_dim = text_proj_dim
        self.model_dim = model_dim
        self.per_joint_out_dim = per_joint_out_dim
        self.num_layers = num_layers
        self.joint_count = joint_count
        self.text_scale = text_scale
        self.normalizer = normalizer

        # Text → initial hidden state
        self.text_to_hidden = nn.Linear(text_embedding_dim, model_dim)

        self.text_proj = nn.Linear(text_embedding_dim, text_proj_dim)

        # GRU over time, input is motion + text at each frame
        self.gru = nn.GRU(
            input_size=frame_feature_dim + text_proj_dim,
            hidden_size=model_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Shared MLP: global GRU hidden (B, H) → all joints (B, 22 * D_joint)
        # self.global_to_joints = nn.Sequential(
        #     nn.Linear(model_dim, model_dim),
        #     nn.ReLU(),
        #     nn.Linear(model_dim, joint_count * per_joint_out_dim),
        # )

    def init_hidden(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb: (B, text_dim)
        Returns h0: (num_layers, B, hidden_dim)
        """
        h0 = self.text_to_hidden(text_emb)  # (B, H)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0  # (L, B, H)

    def _gru_block(
        self,
        motion_in: torch.Tensor,  # (B, T_step, motion_dim)
        text_emb: torch.Tensor,  # (B, text_dim)
        h: Optional[torch.Tensor],  # (L, B, H) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shared core: runs GRU on motion_in with text conditioning and returns:
          history_features: (B, 22, per_joint_dim) from the LAST timestep in this block
          h_next: (L, B, H)
        """
        B, T_step, _ = motion_in.shape

        if h is None:
            h = self.init_hidden(text_emb)  # (L, B, H)

        # Repeat scaled text over time
        text_proj = self.text_proj(text_emb)
        t_rep = self.text_scale * text_proj  # (B, text_proj_dim)
        t_rep = t_rep.unsqueeze(1).expand(B, T_step, -1)  # (B, T_step, text_proj_dim)

        # GRU input
        gru_in = torch.cat([motion_in, t_rep], dim=-1)  # (B, T_step, motion+text)

        # GRU forward
        h_seq, h_next = self.gru(gru_in, h)  # h_seq: (B, T_step, H)
        h_t = h_seq[:, -1, :]  # last timestep in this block, (B, H)

        # Shared MLP to per-joint tokens
        # joint_tokens = self.global_to_joints(h_t)  # (B, 22 * D_joint)
        # history_features = joint_tokens.view(
        #     B, self.joint_count, self.per_joint_out_dim
        # )  # (B, 22, D_joint)
        history_features = h_t.unsqueeze(1).expand(
            B, self.joint_count, self.model_dim
        )  # (B, 22, H)

        return history_features, h_next

    def forward(self, motion_seq: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        motion_seq: (B, T, motion_dim)  full history window
        text_emb:   (B, text_dim)       global text embedding
        Returns:
          history_features: (B, 22, per_joint_dim)
        """
        history_features, _ = self._gru_block(motion_seq, text_emb, h=None)
        return history_features

    def gru_step(
        self,
        x_t: torch.Tensor,  # (B, motion_dim)  single frame features
        text_emb: torch.Tensor,  # (B, text_dim)
        h: Optional[torch.Tensor],  # (L, B, H) or None
        use_normalization: bool = False,  # Whether to apply normalization to x_t
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step update for AR inference.
        Returns:
          history_features: (B, 22, per_joint_dim)  summary up to this frame
          h_next: (L, B, H)  next hidden state to carry forward
        """
        if use_normalization and self.normalizer is not None:
            x_t = self.normalizer.normalize(x_t)

        motion_in = x_t.unsqueeze(1)  # (B, 1, motion_dim)
        history_features, h_next = self._gru_block(motion_in, text_emb, h)
        return history_features, h_next

    @property
    def output_dim(self) -> int:
        # Per-joint dimension exposed to FlowMatchingPredictor
        return self.per_joint_out_dim


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


class SpatialTrackMLP(nn.Module):
    def __init__(self, config: FlowMatchingPredictorConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN["silu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
        use_relative_shift: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = (
            out_channels or config.track_dimensionality
        )  # Usually 2 (x,y)
        self.use_relative_shift = use_relative_shift

        # Input projection: concatenate all features -> hidden_size
        input_dim = (
            config.track_dimensionality  # noised_tracks (x,y coordinates to denoise)
            + feature_size  # track_features (from video encoder, etc.)
            + (
                config.track_dimensionality if use_relative_shift else 0
            )  # relative_shifts (if used, provided externally)
        )
        self.input_projection = nn.Linear(input_dim, config.hidden_size)

        self.global_cond_projection = nn.Linear(
            config.global_cond_dim, config.hidden_size
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
        self.output_projection = nn.Linear(config.hidden_size, self.out_channels)

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
        nn.init.constant_(self.output_projection.weight, 0)
        nn.init.constant_(self.output_projection.bias, 0)

    def forward(
        self,
        noised_tracks: torch.Tensor,  # (B, N, D) - noised trajectory coordinates to denoise
        timesteps: torch.Tensor,  # (B,) or (B, 1) - denoising timesteps in [0,1]
        text_embedding: torch.Tensor,  # (B, F) - Text embedding for global conditioning
        track_features: torch.Tensor,  # (B, N, F) - per-track features from encoder/detector
        prev_relative_shifts: Optional[
            torch.Tensor
        ] = None,  # (B, N, D) - precomputed relative shifts (optional)
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,  # Ignore attention_mask, position_ids, etc.
    ) -> tuple[
        torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]
    ]:
        """
        Forward pass of the Flow Matching Predictor

        Args:
            track_features: (B, N, F) - Features per track (e.g., from video encoder)
            noised_tracks: (B, N, D) - Noised trajectory coordinates to predict flow for
            timesteps: (B,) or (B, 1) - Denoising timesteps t in [0,1]
            relative_shifts: (B, N, D) - Precomputed relative position shifts (optional, default None)
            text_embedding: (B, F) - Text embedding for global conditioning
            output_attentions: Whether to return attention weights (not implemented in this simplified version)
            output_hidden_states: Whether to return hidden states (not implemented in this simplified version)

        Returns:
            flow_prediction: (B, N, D) - Predicted velocity field for denoising
        """

        # 1. Feature concatenation along feature dimension
        features_to_concat = [noised_tracks, track_features]
        if self.use_relative_shift:
            if prev_relative_shifts is None:
                prev_relative_shifts = torch.zeros_like(noised_tracks)
            features_to_concat.append(prev_relative_shifts)

        concatenated_features = torch.cat(
            features_to_concat, dim=-1
        )  # (B, N, F+E+[2]+D)

        # 2. Input projection to transformer dimension
        hidden_states = self.input_projection(concatenated_features)  # (B, N, H)

        # Build static per-joint kinematic tokens once and reuse across layers.
        B, N, H = hidden_states.shape
        kin_tokens = self.kinematic_encoder(self.joint_ids)  # (22, H)
        if kin_tokens.shape[0] != N:
            raise ValueError(
                f"Joint count mismatch: predictor got N={N}, "
                f"but kinematic table has {kin_tokens.shape[0]} joints."
            )
        kin_tokens = self.kinematic_token_norm(kin_tokens)
        kin_tokens = kin_tokens.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        kin_tokens = kin_tokens.unsqueeze(0).expand(B, N, H)  # (B, N, H)

        # 3. Time conditioning
        # Handle both (B,) and (B, 1) timestep formats
        time_cond = self.time_embedder(
            timesteps.squeeze(-1) if timesteps.dim() > 1 else timesteps
        )  # (B, H)

        global_cond_proj = self.global_cond_projection(text_embedding)  # (B, H)

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

        # 5. Output prediction
        normed_states = self.output_norm(hidden_states)  # (B, N, H)
        shift, scale = self.output_adaln(adaln_conditioning).chunk(2, dim=-1)
        normed_states = normed_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        flow_prediction = self.output_projection(normed_states)  # (B, N, D)

        return (
            flow_prediction,
            all_hidden_states if output_hidden_states else None,
            all_self_attns if output_attentions else None,
        )


class HumanMotionGenerator:
    """
    Top-level wrapper for the Human Motion Generation pipeline.
    Integrates MotionHistoryEncoder (Context) and FlowMatchingPredictor (Spatial Generation).

    Updated to use 271D features and proper 72D → 271D conversion for autoregressive generation.
    """

    def __init__(
        self,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
        config: Config,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalizer = encoder.normalizer
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
            dataset_type: Dataset type for feature extraction
            use_fk: Whether to use FK positions for relative shift computation
        Returns:
            position_history: (B, N+num_frames, 22, 3) - Absolute global joint positions including initial history
            feature_history: (B, N+num_frames, 271) - 271D features derived from position history
            prev_relative_shifts: (B, N+num_frames, 22, 3) - Relative shifts from previous frames
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
                position_history = torch.zeros(
                    (B, 1, self.encoder.joint_count, self.predictor.out_channels),
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

            prev_relative_shifts = torch.zeros(
                (B, 1, self.encoder.joint_count, self.predictor.out_channels),
                device=device,
            )

            if position_history.shape[1] > 1:
                prev_relative_shifts = torch.cat(
                    [
                        prev_relative_shifts,
                        position_history[:, 1:] - position_history[:, :-1],
                    ],
                    dim=1,
                )

            for frame_idx in range(num_frames):
                # ========================================
                # Step A: Extract last frame from position history
                # ========================================
                current_positions = position_history[:, -1]  # (B, 22, 3)

                # ========================================
                # Step B: Encode context from last horizon frames
                # Feature history is used only for context encoding.
                # ========================================
                # Slice to last horizon frames
                if horizon is not None:
                    horizon_frames = min(horizon, feature_history.shape[1])
                    encoder_input = feature_history[
                        :, -horizon_frames:, :
                    ]  # (B, horizon, 271)
                else:
                    encoder_input = feature_history

                # Strict text shape policy: (B, 1, 512) at entry, (B, 512) for encoder/predictor.
                text_emb = text[:, 0, :]

                context_cond = self.encoder(
                    encoder_input,
                    text_emb,
                )  # (B, 22, per_joint_dim)

                # ========================================
                # Step C: Flow matching ODE loop
                # x_t starts as random noise in track space (B, 22, 3)
                # ========================================
                x_t = torch.randn(
                    (B, self.encoder.joint_count, self.predictor.out_channels),
                    device=device,
                )
                dt = 1.0 / num_steps

                # N-step flow matching in tokenized track space.
                for step in range(num_steps):
                    t = torch.full((B,), step * dt, device=device)
                    relative_shifts = (
                        prev_relative_shifts[:, -1]
                        if prev_relative_shifts.shape[1] > 0
                        else None
                    )

                    # Predict velocity using new forward signature
                    flow_output = self.predictor.forward(
                        track_features=context_cond,
                        noised_tracks=x_t,
                        timesteps=t,
                        prev_relative_shifts=relative_shifts,
                        text_embedding=text_emb,
                        output_attentions=False,
                        output_hidden_states=False,
                    )

                    # Unpack tuple: (flow_prediction, hidden_states, attentions)
                    pred = flow_output[0]
                    x_t = x_t + pred * dt

                # ========================================
                # Step E: Apply displacement and convert positions → 271D (incremental)
                # ========================================
                relative_shift = x_t  # (B, 22, 3)
                new_positions = current_positions + relative_shift  # (B, 22, 3)
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
                prev_relative_shifts = torch.cat(
                    [prev_relative_shifts, relative_shift.unsqueeze(1)], dim=1
                )  # (B, T, 22, 3)

                if (frame_idx + 1) % 50 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            return position_history, feature_history, prev_relative_shifts

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
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        if "config" in checkpoint:
            config: Config = checkpoint["config"]

        # Initialize Motion History Encoder (GRU-based)
        encoder = MotionHistoryEncoder(
            frame_feature_dim=config.encoder_motion_dim,
            text_embedding_dim=config.encoder_text_dim,
            text_proj_dim=config.encoder_text_proj_dim,
            model_dim=config.encoder_hidden_dim,
            per_joint_out_dim=config.encoder_per_joint_dim,
            num_layers=config.encoder_num_layers,
            joint_count=config.encoder_num_joints,
            text_scale=config.encoder_text_scale,
            dropout=config.encoder_dropout,
            normalizer=normalizer,
        ).to(device)

        # Initialize Flow Matching Predictor with new config-based interface
        predictor_config = config.predictor_config
        feature_size = config.get_predictor_feature_size()

        predictor = FlowMatchingPredictor(
            feature_size=feature_size,
            config=predictor_config,
            out_channels=None,
            use_relative_shift=True,
            normalizer=normalizer,
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

        return cls(encoder, predictor, config)
