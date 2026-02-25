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
from typing import Optional, List, Tuple, Union
from config import Config

from utils.motion_utils import (
    features_to_positions,
    flow_output_to_positions,
    flow_output_to_271d,
    RootPositionTracker,
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

        self.chain_emb = nn.Embedding(5, model_dim // 2)
        self.depth_emb = nn.Embedding(8, model_dim // 2)

    def forward(self, joint_ids: torch.Tensor) -> torch.Tensor:
        # joint_ids: (n_joints,)
        chains = self.joint_to_chain[joint_ids]
        depths = self.joint_to_depth[joint_ids]
        return torch.cat([self.chain_emb(chains), self.depth_emb(depths)], dim=-1)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        # x: (B, H, T, D)
        seq_len = x.shape[2]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self.cos_cached = emb.cos()[None, None, :, :]  # (1,1,T,D)
            self.sin_cached = emb.sin()[None, None, :, :]

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class TemporalAttentionWithRoPE(nn.Module):
    def __init__(self, model_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        assert model_dim % nhead == 0, "model_dim must be divisible by nhead"

        self.model_dim = model_dim
        self.nhead = nhead
        self.head_dim = model_dim // nhead

        # QKV projection
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)

        # Rotary embedding
        self.rope = Rotary(self.head_dim)

        self.dropout = dropout

    def forward(self, x, causal_mask=None):
        """
        x: (B, T, D)
        causal_mask: (T, T) bool mask where True = masked
        """

        B, T, D = x.shape

        # ----------------------------
        # QKV projection
        # ----------------------------
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Hd)

        q, k, v = qkv.unbind(0)  # each: (B, H, T, Hd)

        # ----------------------------
        # Apply RoPE to Q and K
        # ----------------------------
        cos, sin = self.rope(q)  # (1,1,T,Hd)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ----------------------------
        # Scaled Dot-Product Attention
        # ----------------------------
        # scaled_dot_product_attention expects:
        # (B, H, T, Hd)
        # and supports bool mask directly

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )  # (B, H, T, Hd)

        # ----------------------------
        # Merge heads
        # ----------------------------
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, D)

        out = self.out_proj(attn_output)

        return out


class TemporalTransformerBlock(nn.Module):
    def __init__(self, model_dim, nhead, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = TemporalAttentionWithRoPE(model_dim, nhead, dropout)

        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # PreNorm Attention
        x = x + self.dropout(self.attn(self.norm1(x), causal_mask))

        # PreNorm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class SpatiotemporalBlock(nn.Module):
    def __init__(self, model_dim, nhead, dropout):
        super().__init__()

        # ================= SPATIAL =================
        self.spatial_norm1 = nn.LayerNorm(model_dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_dropout = nn.Dropout(dropout)

        self.spatial_norm2 = nn.LayerNorm(model_dim)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

        # ================= TEMPORAL =================
        self.temporal_block = TemporalTransformerBlock(
            model_dim=model_dim,
            nhead=nhead,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: (B, T, S, D)
        """

        B, T, S, D = x.shape

        # ============================================
        # 1️⃣ Spatial Attention (within frame)
        # ============================================

        x_spatial = x.reshape(B * T, S, D)

        # --- Attention ---
        residual = x_spatial
        x_spatial = self.spatial_norm1(x_spatial)

        attn_out, _ = self.spatial_attn(
            x_spatial,
            x_spatial,
            x_spatial,
            need_weights=False,
        )

        x_spatial = residual + self.spatial_dropout(attn_out)

        # --- FFN ---
        residual = x_spatial
        x_spatial = self.spatial_norm2(x_spatial)
        x_spatial = residual + self.spatial_dropout(self.spatial_ffn(x_spatial))

        x = x_spatial.reshape(B, T, S, D)

        # ============================================
        # 2️⃣ Temporal Attention (per joint stream)
        # ============================================

        x = x.transpose(1, 2)  # (B, S, T, D)
        x_temporal = x.reshape(B * S, T, D)

        x_temporal = self.temporal_block(x_temporal)

        x = x_temporal.reshape(B, S, T, D)
        x = x.transpose(1, 2)

        return x


class MotionHistoryEncoder(nn.Module):
    """
    ARFM Feature Fusion Transformer - Motion History Encoder.

    Replaces GRU-based encoder with a spatiotemporal transformer architecture:
    1. Input Preparation:
       - CLIP Text Sequence (B, l_seq, 512) → Linear → Text Prefix Tokens (l_seq tokens)
       - Per-timestep Global Features → Linear → Global Token (1 per timestep)
       - Per-timestep Per-track Local Features + KinematicChainEncoder → Linear → Track Tokens (22 per timestep)
    2. Spatiotemporal Sequence: Concat [Text Prefix (l_seq×22); per timestep: [Global (1); Track (21)]] → (B, l_seq + L_past, 22, d_model)
    3. Positional Encoding: Temporal (RoPE/sinusoidal) + Spatial (learnable)
    4. Transformer Stack (4 layers): Temporal Causal + Spatial Bidirectional + Shared FFN
    5. Output: Last timestep track features  (B, 22, d_model)
    """

    def __init__(
        self,
        frame_feature_dim: int,  # Custom Feature Dimension 271D
        text_embedding_dim: int,  # CLIP embedding dimension (e.g., 512)
        per_joint_out_dim: int,
        joint_count: int = 22,
        model_dim: int = 256,
        num_layers: int = 4,  # Transformer layers (default 4)
        max_text_seq_len: int = 1,  # CLIP max sequence length
        dropout: float = 0.1,
        normalizer: Optional[
            "FeatureNormalizer"
        ] = None,  # For normalizing raw features
    ) -> None:
        super().__init__()

        self.frame_feature_dim = frame_feature_dim
        self.text_embedding_dim = text_embedding_dim
        self.per_joint_out_dim = per_joint_out_dim

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.joint_count = joint_count
        self.max_text_seq_len = max_text_seq_len

        # Store normalizer for raw feature normalization
        self.normalizer = normalizer

        # Attention config
        self.nhead = model_dim // 64
        self.head_dim = model_dim // self.nhead
        assert (
            self.head_dim * self.nhead == model_dim
        ), "model_dim must be divisible by nhead"

        # ========== 1. Input Projections ==========

        # Text projection: CLIP sequence embedding (B, l_seq, 512) → (B, l_seq, model_dim)
        self.text_projection = nn.Linear(text_embedding_dim, model_dim)

        # Global/root features projection: global/root features → Global Token
        # 271D format global features (16D):
        # - root_height_y (1D) from [0]
        # - root_vel_x (1D) from [1]
        # - root_vel_z (1D) from [2]
        # - root_rot_6d (6D) from [69:75]
        # - root_local_vel (3d) from [201:204]
        # - foot_contacts (4D) from [267:271]
        self.global_feature_dim = 16
        self.global_proj = nn.Linear(self.global_feature_dim, model_dim)

        # Non Root Track features projection: local features → Track Tokens
        # 271D format track features:
        # - RIC positions (3D) from [6:69] for 21 joints
        # - Rotations (6D) from [75:201] for 21 joints
        # - Local velocities (3D) from [204:267] for 21 joints
        # Kinematic embedding is added as bias after projection (not concatenated)
        self.track_feature_dim = 12  # local features only
        self.track_proj = nn.Linear(self.track_feature_dim, model_dim)

        # ========== 2. Kinematic Chain Encoder (reuse existing) ==========
        self.kinematic_encoder = KinematicChainEncoder(model_dim)

        # ========== 4. Transformer Stack ==========

        self.blocks = nn.ModuleList(
            [
                SpatiotemporalBlock(
                    model_dim=model_dim,
                    nhead=self.nhead,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms
        self.final_norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        # ========== 5. Output Projection ==========
        # Project to per_joint_out_dim
        self.output_proj = nn.Linear(model_dim, per_joint_out_dim)

        # Null tokens for zero-shot generation
        self.null_history = nn.Parameter(torch.zeros(1, 1, frame_feature_dim))
        self.null_text_embedding = nn.Parameter(
            torch.zeros(1, max_text_seq_len, text_embedding_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear and embedding weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        text: Optional[torch.Tensor],
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        return_all_timesteps: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        ARFM Feature Fusion Transformer forward pass.

        Args:
            text:           (B, l_seq, 512) CLIP sequence embeddings or None
            input_features: (B, T_hist, frame_feature_dim) - Optional motion history (RAW features)
            total_duration: (B, 1) Normalized total frames (Optional, not used in this version)
            batch_size:     Optional batch size for null history case
            normalize:      If True and normalizer is set, normalize input_features (default: True)
                            Set to False during training when normalization is handled externally

        Returns:
            joint_features: (B, 22, per_joint_out_dim) - Encoded per-joint context
        """
        # 1) Determine Batch Size
        if input_features is not None:
            B = input_features.shape[0]
        elif text is not None:
            B = text.shape[0]
        elif batch_size is not None:
            B = batch_size
        else:
            B = 1

        device = next(self.parameters()).device

        # 2) Handle text conditioning (now expects sequence embeddings)
        if text is None:
            # Use null text embedding sequence
            text = self.null_text_embedding.to(device).expand(
                B, -1, -1
            )  # (B, l_seq, 512)
        else:
            # text should be (B, l_seq, 512) sequence embeddings
            if text.shape[0] != B:
                if text.shape[0] == 1:
                    text = text.expand(B, -1, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch: text tensor({text.shape[0]}) vs batch({B})"
                    )

        # Get actual text sequence length
        L_text = text.shape[1]

        # 3) Handle motion features (use null token if none)
        if input_features is None:
            T = 1
            input_features = self.null_history.to(device).expand(B, T, -1)
        else:
            T = input_features.shape[1]

        # Normalize raw features if normalizer is provided and normalize=True
        if self.normalizer is not None and normalize:
            input_features = self.normalizer.normalize(input_features)

        S = self.joint_count

        # ========== INPUT PREPARATION ==========

        # --- Text Prefix Tokens (from CLIP sequence) ---
        # Project CLIP embeddings to model dimension
        text_tokens = self.text_projection(text)  # (B, L_text, D)
        text_tokens = text_tokens.unsqueeze(2).expand(
            B, L_text, S, self.model_dim
        )  # (B, L_text, 22, D)

        # ----- Global Features (16D) -----
        root_height_y = input_features[:, :, 0:1]
        root_vel_x = input_features[:, :, 1:2]
        root_vel_z = input_features[:, :, 2:3]
        root_rot_6d = input_features[:, :, 69:75]
        root_local_vel = input_features[:, :, 201:204]
        foot_contacts = input_features[:, :, 267:271]

        global_features = torch.cat(
            [
                root_height_y,
                root_vel_x,
                root_vel_z,
                root_rot_6d,
                root_local_vel,
                foot_contacts,
            ],
            dim=-1,
        )  # (B, T, 16)

        global_token = self.global_proj(global_features)  # (B, T, D)
        global_token = global_token.unsqueeze(2)  # (B, T, 1, D)

        # --- Per-timestep Per-track Local Features ---
        ric = input_features[:, :, 6:69].view(B, T, 21, 3)
        rot = input_features[:, :, 75:201].view(B, T, 21, 6)
        vel = input_features[:, :, 204:267].view(B, T, 21, 3)

        track_features = torch.cat([ric, rot, vel], dim=-1)  # (B, T, 21, 12)
        track_tokens = self.track_proj(track_features)  # (B, T, 21, D)

        # Combine global and track tokens
        motion_tokens = torch.cat([global_token, track_tokens], dim=2)  # (B, T, 22, D)

        # Add kinematic chain embeddings as bias (position bias)
        joint_ids = torch.arange(self.joint_count, device=device)
        kinematic_emb = self.kinematic_encoder(joint_ids)  # (22, D)
        motion_tokens = motion_tokens + kinematic_emb.unsqueeze(0).unsqueeze(
            0
        )  # (B, T_hist, 22, D)

        x = torch.cat([text_tokens, motion_tokens], dim=1)
        # Shape: (B, L_text + T, 22, D)

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)  # (B, L_text + T, 22, D)

        x = x[:, -T:]  # (B, T, 22, D)

        out = self.output_proj(x)  # (B, T, 22, out_dim)

        return out

    @property
    def output_dim(self) -> int:
        """Output dimension of the context encoder."""
        return self.per_joint_out_dim


class FlowMatchingPredictor(nn.Module):
    """
    Spatial Transformer-based Flow Matching Predictor (ARFM-style).

    Input:
    - history_features: (B, 22, per_joint_dim)
        - Conditional context from MotionHistoryEncoder
        - Provides temporal and spatial history for all joints including root

    - noisy_target: (B, 72)
        - Root features (first 9D):
            - 1D root height
            - 2D root local velocity (rotation-invariant)
            - 6D absolute rotation
        - Joint features (next 63D):
            - 21 non-root joints, each with 3D RIC positions
        - Represents the current noisy state x_t in flow matching

    - prev_frame_features: (B, 261)
        - Root features (first 9D):
            - 1D root height
            - 2D root local velocity (rotation-invariant)
            - 6D absolute rotation
        - Joint features (next 252D):
            - 21 non-root joints x 12D each:
                - 3D RIC positions
                - 6D rotations
                - 3D local velocities

    - noise_level: (B,)
        - Scalar flow time t in [0,1] used to compute sinusoidal embedding

    - temporal_progress: (B,)
        - Optional normalized frame progress in sequence (not used yet)

    Output:
    - pred_frame: (B, 72)
        - Concatenated predictions for a single frame:
            - Root token output (first 9D):
                1D height + 2D velocity + 6D rotation
            - Joint tokens output (next 63D):
                21 joints x 3D RIC positions
        - Matches the target frame representation used for flow matching / denoising
    """

    def __init__(
        self,
        per_joint_dim: int = 32,  # history embedding dimension from MotionHistoryEncoder
        model_dim: int = 64,  # spatial transformer hidden dim
        num_layers: int = 2,  # number of spatial transformer layers
        joint_count: int = 22,  # total joints including root
        time_embed_dim: int = 64,  # sinusoidal time embedding
        dropout: float = 0.1,  # dropout rate
        normalizer: Optional[FeatureNormalizer] = None,  # For normalizing raw features
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.joint_count = joint_count
        self.time_embed_dim = time_embed_dim

        # Store normalizer for raw feature normalization
        self.normalizer = normalizer

        # --- Hierarchical Kinematic Encoder ---
        self.kinematic_encoder = KinematicChainEncoder(model_dim)

        # --- Sinusoidal time embedding ---
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # --- Project history / prev frame separately ---
        self.input_proj_history = nn.Linear(
            per_joint_dim, model_dim
        )  # only history embeddings

        # Prev frame projections (root + joints)
        self.input_proj_prev_root = nn.Linear(9, model_dim)  # root: height + vel + rot
        self.input_proj_prev_joint = nn.Linear(
            12, model_dim
        )  # joints: RIC + rot + velocity

        # --- Project noisy target separately ---
        self.input_proj_noisy_root = nn.Linear(9, model_dim)  # noisy root features
        self.input_proj_noisy_joint = nn.Linear(
            3, model_dim
        )  # noisy joint RIC positions

        # --- Spatial Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )

        # --- Separate output heads ---
        self.root_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 9),  # 1D height + 2D velocity + 6D rotation
        )
        self.joint_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 3),  # 3D RIC positions
        )

        # --- Null tokens for prev frame ---
        self.null_prev_root = nn.Parameter(torch.zeros(1, 9))
        self.null_prev_joint = nn.Parameter(torch.zeros(1, joint_count - 1, 12))

    def _sinusoidal_time_embedding(self, t: torch.Tensor, max_positions=10000):
        """Sinusoidal embedding for flow time t in [0,1]."""
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_positions))
            / (half_dim - 1)
            * torch.arange(half_dim, device=t.device)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_embed_dim % 2 != 0:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, time_embed_dim)

    def forward(
        self,
        history_features: torch.Tensor,  # (B, 22, per_joint_dim)
        noise_level: torch.Tensor,  # (B,)
        noisy_target: Optional[torch.Tensor] = None,  # (B, 72) - in normalized space
        prev_frame_features: Optional[torch.Tensor] = None,  # (B, 261) - RAW features
        temporal_progress: Optional[torch.Tensor] = None,
        normalize: bool = True,  # If True, normalize prev_frame_features (for inference)
    ):
        """
        Forward pass for FlowMatchingPredictor.

        Args:
            history_features: (B, 22, per_joint_dim) - Context from MotionHistoryEncoder
            noise_level: (B,) - Flow time t in [0,1]
            noisy_target: (B, 72) - Current noisy state x_t (already in normalized space)
            prev_frame_features: (B, 261) - Previous frame features (RAW if normalize=True)
            temporal_progress: (B,) - Optional normalized frame progress
            normalize: If True and normalizer is set, normalize prev_frame_features
                       Set to False during training when normalization is handled externally

        Returns:
            pred_frame: (B, 72) - Predicted velocity field (in normalized space)
        """
        B, J, _ = history_features.shape
        device = history_features.device

        # --- Normalize raw prev_frame_features if normalizer is provided and normalize=True ---
        # Note: noisy_target is already in normalized space (from ODE integration)
        if self.normalizer is not None and normalize:
            if prev_frame_features is not None:
                prev_frame_features = self.normalizer.normalize_prev_frame_features(
                    prev_frame_features
                )

        # --- Handle optional prev frame ---
        if prev_frame_features is None:
            prev_root = self.null_prev_root.expand(B, 9)
            prev_joints = self.null_prev_joint.expand(B, J - 1, 12)
        else:
            # Split root / joints
            prev_root = prev_frame_features[:, :9]  # (B,9)
            prev_joints = prev_frame_features[:, 9:]  # (B, (J-1)*12)
            prev_joints = prev_joints.reshape(B, J - 1, 12)

        # --- Project history and prev frame separately ---
        history_proj = self.input_proj_history(history_features)  # (B,22,model_dim)

        prev_root_proj = self.input_proj_prev_root(prev_root).unsqueeze(
            1
        )  # (B,1,model_dim)
        prev_joint_proj = self.input_proj_prev_joint(prev_joints)  # (B,21,model_dim)
        prev_proj = torch.cat(
            [prev_root_proj, prev_joint_proj], dim=1
        )  # (B,22,model_dim)

        # --- Combine history + prev frame ---
        cond_proj = history_proj + prev_proj  # (B,22,model_dim)

        # --- Handle noisy target ---
        if noisy_target is None:
            noisy_target = torch.randn(B, 72, device=device)

        noisy_root_in = noisy_target[:, :9]  # (B,9)
        noisy_joints_in = noisy_target[:, 9:].reshape(B, J - 1, 3)  # (B,21,3)

        noisy_root_proj = self.input_proj_noisy_root(noisy_root_in).unsqueeze(1)
        noisy_joint_proj = self.input_proj_noisy_joint(noisy_joints_in)

        noisy_proj = torch.cat([noisy_root_proj, noisy_joint_proj], dim=1)

        # --- Combine condition + noisy ---
        x = cond_proj + noisy_proj  # (B,22,model_dim)

        # --- Add time embedding ---
        t_emb = self._sinusoidal_time_embedding(noise_level)
        t_bias = self.time_mlp(t_emb)
        x = x + t_bias.unsqueeze(1)

        # --- Add kinematic bias ---
        joint_ids = torch.arange(J, device=device)
        kinematic_bias = self.kinematic_encoder(joint_ids)
        x = x + kinematic_bias.unsqueeze(0)

        # --- Spatial Transformer ---
        x = self.spatial_transformer(x)  # (B,22,model_dim)

        # --- Split root and joint tokens for separate heads ---
        root_token = x[:, 0:1, :]
        joint_tokens = x[:, 1:, :]
        root_out = self.root_head(root_token)  # (B,1,9)
        joint_out = self.joint_head(joint_tokens)  # (B,21,3)

        # --- Flatten and concatenate to final 72D ---
        pred_frame = torch.cat([root_out.squeeze(1), joint_out.reshape(B, -1)], dim=-1)
        return pred_frame


class HumanMotionGenerator(nn.Module):
    """
    Top-level wrapper for the Human Motion Generation pipeline.
    Integrates MotionHistoryEncoder (Context) and FlowMatchingPredictor (Spatial Generation).

    Updated to use 271D features and proper 72D → 271D conversion for autoregressive generation.
    """

    def __init__(
        self,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalizer = encoder.normalizer

    def generate_sequence(
        self,
        text: Union[str, List[str], torch.Tensor],  # type: ignore
        num_frames: int = 200,
        num_steps: int = 10,
        guidance_scale: float = 2.5,
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        dataset_type: str = "t2m",
    ) -> torch.Tensor:
        """
        Generate n consecutive animation frames autoregressively.

        Uses flow_output_to_271d for incremental 72D → 271D conversion and
        RootPositionTracker for absolute root position tracking. This ensures
        O(n) complexity and Markov-safe autoregressive generation.

        Args:
            text: Text prompt(s) - str, List[str], or pre-encoded tensor (B, l_seq, 512)
            num_frames: Number of frames to generate
            num_steps: Number of flow matching ODE steps
            guidance_scale: Classifier-free guidance scale
            input_features: Optional initial motion history (B, N, 271) or (B, 271)
            total_duration: Optional duration tensor (not used)
            dataset_type: Dataset type for feature extraction

        Returns:
            position_history: (B, N+num_frames, 22, 3) - Global joint positions including initial history
        """
        from utils.train_utils import extract_prev_frame_features

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
                B = text.shape[0]
            device = next(self.parameters()).device

            # ========================================
            # History Initialization
            # ========================================
            # Use RootPositionTracker for absolute root position tracking
            # and feature_history for context encoding

            if input_features is None:
                root_pos = torch.zeros((B, 3), device=device)  # (B, 3) - absolute XYZ
                root_tracker = RootPositionTracker(root_pos)
            else:
                # Seeded: use input_features
                if input_features.ndim == 2:
                    # Single frame (B, 271)
                    feature_history = input_features.unsqueeze(1).clone()  # (B, 1, 271)
                else:
                    # Sequence (B, N, 271)
                    feature_history = input_features.clone()  # (B, N, 271)
                # Initialize RootPositionTracker from feature history
                root_tracker = RootPositionTracker.from_history(feature_history)

            for frame_idx in range(num_frames):
                # ========================================
                # Step A: Extract last frame from feature history
                # ========================================
                last_frame = feature_history[:, -1]  # (B, 271) - RAW
                prev_root_pos = root_tracker.get()  # (B, 3)

                # ========================================
                # Step B: Encode context from FULL feature history (for CFG)
                # normalize=True (default) - models normalize RAW features internally
                # ========================================
                context_cond = self.encoder(
                    batch_size=B,
                    text=text,
                    input_features=feature_history,  # RAW features
                    normalize=True,  # Normalize internally for inference
                )[
                    :, -1, :, :
                ]  # (B, 22, out_dim)

                context_uncond = self.encoder(
                    batch_size=B,
                    text=None,
                    input_features=feature_history,  # RAW features
                    normalize=True,  # Normalize internally for inference
                )[
                    :, -1, :, :
                ]  # (B, 22, out_dim)

                # ========================================
                # Step C: Extract prev_frame_features (261D) from RAW features
                # ========================================
                prev_frame_features = extract_prev_frame_features(last_frame)

                # ========================================
                # Step D: Flow matching ODE loop
                # x_t starts as random noise (already in normalized space conceptually)
                # ========================================
                x_t = torch.randn((B, 72), device=device)
                dt = 1.0 / num_steps

                for step in range(num_steps):
                    t = torch.full((B,), step * dt, device=device)

                    v_cond = self.predictor(
                        history_features=context_cond,
                        noise_level=t,
                        noisy_target=x_t,
                        prev_frame_features=prev_frame_features,
                        normalize=True,  # Normalize prev_frame_features internally
                    )

                    v_uncond = self.predictor(
                        history_features=context_uncond,
                        noise_level=t,
                        noisy_target=x_t,
                        prev_frame_features=prev_frame_features,
                        normalize=True,  # Normalize prev_frame_features internally
                    )

                    # Classifier-free guidance
                    v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
                    x_t = x_t + v_t * dt

                if self.normalizer:
                    x_t = self.normalizer.denormalize_flow_output(x_t)

                # ========================================
                # Step E: Convert 72D → 271D (incremental)
                # ========================================
                new_frame, new_root_pos = flow_output_to_271d(
                    flow_output=x_t,
                    prev_frame=last_frame,
                    prev_root_pos=prev_root_pos,
                    dataset_type=dataset_type,
                )  # (B, 271), (B, 3)

                # ========================================
                # Step F: Update tracker and history
                # ========================================
                root_tracker.update(new_frame)
                feature_history = torch.cat(
                    [feature_history, new_frame.unsqueeze(1)], dim=1
                )  # (B, N+1, 271)

                if (frame_idx + 1) % 50 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            # Convert full feature_history to positions for return
            position_history = features_to_positions(
                feature_history, dataset_type=dataset_type
            )  # (B, N+num_frames, 22, 3)

            return position_history

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Config,
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
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize Motion History Encoder (Transformer-based)
        encoder = MotionHistoryEncoder(
            frame_feature_dim=config.motion_dim,
            text_embedding_dim=config.text_embedding_dim,
            per_joint_out_dim=config.per_joint_out_dim,
            joint_count=config.num_joints,
            model_dim=config.model_dim,
            num_layers=config.num_encoder_layers,
            max_text_seq_len=config.max_text_seq_len,
            dropout=config.dropout,
            normalizer=normalizer,
        ).to(device)

        # Initialize Flow Matching Predictor
        predictor = FlowMatchingPredictor(
            per_joint_dim=config.per_joint_out_dim,
            model_dim=config.model_dim,
            num_layers=config.num_flow_layers,
            joint_count=config.num_joints,
            time_embed_dim=config.time_embed_dim,
            dropout=config.dropout,
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

        return cls(encoder, predictor)
