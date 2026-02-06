"""
Model architectures for Human Motion Animation Generation.

This module contains:
- AutoregressiveContextEncoder: Encodes motion context sequentially
- FlowMatchingNetwork: Generates motion sequences using flow matching

Compatible with MoMask input format: dim-263 feature vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Callable, Any


class MotionHistoryEncoder(nn.Module):
    """
    Motion History Encoder for motion generation using GRU.

    Processes motion sequences sequentially to encode contextual information.
    Input: dim-137 subset feature vectors of (HumanML3D format) + text embedding
    Output: Context embeddings for flow matching
    """

    def __init__(
        self,
        frame_feature_dim: int,  # 137 for your subset
        text_embedding_dim: int,  # e.g., 512 for CLIP
        joint_feature_projection_dim: int,
        text_projection_dim: int,
        per_joint_out_dim: int,
        joint_count: int = 22,
        model_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        text_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.frame_feature_dim = frame_feature_dim
        self.text_embedding_dim = text_embedding_dim
        self.joint_feature_projection_dim = joint_feature_projection_dim
        self.text_projection_dim = text_projection_dim
        self.per_joint_out_dim = per_joint_out_dim

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.text_encoder = text_encoder
        self.joint_count = joint_count

        self.text_projection = nn.Linear(text_embedding_dim, text_projection_dim)

        # 4) MLP to encode per-joint features before temporal processing
        joint_input_dim = 3 + 3 + 8 + text_projection_dim
        self.per_joint_encoder = nn.Sequential(
            nn.Linear(joint_input_dim, joint_feature_projection_dim * 2),
            nn.SiLU(),
            nn.Linear(joint_feature_projection_dim * 2, joint_feature_projection_dim),
        )

        self.gru = nn.GRU(
            input_size=joint_feature_projection_dim * joint_count,
            hidden_size=model_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # 6) MLP head to project final hidden state to per-joint features
        num_directions = 2 if bidirectional else 1
        gru_hidden_dim = num_layers * num_directions * model_dim
        intermediate_dim = model_dim * 2
        self.per_joint_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, joint_count * per_joint_out_dim),
        )

    def _encode_text(
        self, text: Union[str, List[str]], batch_size: int
    ) -> torch.Tensor:
        """
        Encodes text and ensures it matches the batch size through broadcasting or validation.
        """
        if self.text_encoder is None:
            raise ValueError(
                "text_encoder must be provided during initialization to use text inputs."
            )

        if isinstance(text, str):
            # Encode single string and broadcast to batch size
            single_embedding = self.text_encoder(text)  # (1, dim) or (dim,)
            if single_embedding.dim() == 1:
                single_embedding = single_embedding.unsqueeze(0)
            return single_embedding.expand(batch_size, -1)
        elif isinstance(text, list):
            if len(text) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: text list length ({len(text)}) does not match motion batch size ({batch_size})."
                )
            return self.text_encoder(text)  # (B, dim)
        else:
            raise TypeError(f"Expected text to be str or List[str], got {type(text)}")

    def forward(
        self,
        input_features: torch.Tensor,
        text: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Input:
          - input_features: (B, T_hist, frame_feature_dim)
          - text: (B, string)
        Returns:
            history_context: (B, T_hist, H) per-frame GRU outputs
            final_hidden:    (B, num_layers * num_directions * H) - batch-first flat vector
        """
        B, T_hist, _ = input_features.shape

        # 1) Handle text input: Encode and ensure shape (B, text_embedding_dim)
        text_embeddings = self._encode_text(text, B)
        text_projected: torch.Tensor = self.text_projection(
            text_embeddings
        )  # (B, text_projection_dim)

        # Expand text across time and joints
        text_projected_expanded = (
            text_projected.unsqueeze(1)
            .unsqueeze(2)
            .expand(B, T_hist, self.joint_count, -1)
        )  # (B, T_hist, 22, text_projection_dim)

        # 2) Handle motion features
        # - ric position (Indices 4:67 for 21 joints)
        ric_joints = input_features[:, :, 4:67]  # (B, T_hist, 63)
        ric_joints = ric_joints.view(
            B, T_hist, self.joint_count - 1, 3
        )  # (B, T_hist, 21, 3)
        root_ric = torch.zeros((B, T_hist, 1, 3), device=ric_joints.device)
        ric_joints = torch.cat([root_ric, ric_joints], dim=2)  # (B, T_hist, 22, 3)

        # - ric velocities (Indices 193:259 for 22 joints)
        ric_vel = input_features[:, :, 193:259]  # (B, T_hist, 66)
        ric_vel = ric_vel.view(B, T_hist, self.joint_count, 3)  # (B, T_hist, 22, 3)

        # - root angular velocity | linear velocity | root height | foot contact
        # Indices: [0:4] and [259:263]
        global_features = torch.cat(
            [input_features[:, :, 0:4], input_features[:, :, 259:263]],
            dim=-1,
        )  # (B, T_hist, 8)

        global_features = global_features.unsqueeze(-2).expand(
            B, T_hist, self.joint_count, -1
        )  # (B, T_hist, 22, 8)

        # 3) Concatenate all features
        motion_tokens = torch.cat(
            [ric_joints, ric_vel, global_features, text_projected_expanded], dim=-1
        )  # (B, T_hist, 22, 3 + 3 + 8 + text_projection_dim)

        # 4) Project per joint via MLP encoder
        fused_tokens: torch.Tensor = self.per_joint_encoder(
            motion_tokens
        )  # (B, T_hist, 22, joint_feature_projection_dim)

        # 5) Flatten joint dimension for GRU
        fused_tokens = fused_tokens.view(
            B, T_hist, -1
        )  # (B, T_hist, 22 * joint_feature_projection_dim)

        # 6) Run GRU over time
        # gru_out shape: (B, T_hist, model_dim), hidden shape: (num_layers*num_dirs, B, model_dim)
        _, final_hidden_raw = self.gru(fused_tokens)

        # 7) Transform final_hidden to batch-first (B, L*D*model_dim)
        final_hidden_vec = final_hidden_raw.transpose(0, 1).reshape(B, -1)

        # 8) Project to per-joint features via MLP head
        joint_features = self.per_joint_head(
            final_hidden_vec
        )  # (B, 22 * per_joint_out_dim)
        joint_features = joint_features.view(
            B, self.joint_count, -1
        )  # (B, 22, per_joint_out_dim)

        return joint_features

    @property
    def output_dim(self) -> int:
        """Output dimension of the context encoder (H)."""
        return self.model_dim * (2 if self.bidirectional else 1)


class FlowMatchingNetwork(nn.Module):
    """
    Flow matching predictor for ARFM-style motion prediction.

    Inputs:
        history_ctx: (B, C)          - history context (GRU + text, etc.)
        x_t:        (B, J, D_x)     - current noised displacement per joint
        v_prev:     (B, J, D_v)     - previous local velocity (or other local per-joint features)
        t:          (B,) or (B, 1)  - flow time in [0, 1]

    Output:
        v_hat:      (B, J, D_out)   - predicted flow / velocity for each joint
    """

    def __init__(
        self,
        history_ctx_dim: int,  # C
        x_dim: int = 3,  # D_x, e.g., 3D displacement
        v_dim: int = 3,  # D_v, e.g., 3D velocity
        t_embed_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 3,
        out_dim: int = 3,  # D_out, usually 3 for 3D flow
        dropout: float = 0.0,
    ):
        super().__init__()

        self.history_ctx_dim = history_ctx_dim
        self.x_dim = x_dim
        self.v_dim = v_dim
        self.t_embed_dim = t_embed_dim

        # Time embedding: simple MLP on scalar t
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        # Project global context to match feature space
        self.global_proj = nn.Linear(history_ctx_dim, hidden_dim)

        # First layer input dimension
        in_dim = hidden_dim + x_dim + v_dim + t_embed_dim

        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, history_ctx, x_t, v_prev, t):
        """
        history_ctx: (B, C)
        x_t:        (B, J, D_x)
        v_prev:     (B, J, D_v)
        t:          (B,) or (B, 1)
        """
        B, J, _ = x_t.shape

        # Ensure shapes
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        # Time embedding
        t_emb = self.t_mlp(t)  # (B, t_embed_dim)
        t_emb = t_emb.unsqueeze(1).expand(B, J, -1)  # (B, J, t_embed_dim)

        # Global context projected and broadcast to joints
        g = self.global_proj(history_ctx)  # (B, hidden_dim)
        g = g.unsqueeze(1).expand(B, J, -1)  # (B, J, hidden_dim)

        # Concatenate all conditioning and local features
        h = torch.cat(
            [g, x_t, v_prev, t_emb], dim=-1
        )  # (B, J, hidden + x_dim + v_dim + t_emb)

        # Flatten joints into batch for MLP
        h = h.view(B * J, -1)  # (B*J, F)
        h = self.mlp(h)  # (B*J, hidden_dim)
        v_hat = self.out_layer(h)  # (B*J, out_dim)

        # Reshape back to (B, J, out_dim)
        v_hat = v_hat.view(B, J, -1)
        return v_hat
