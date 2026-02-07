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
    Motion History Encoder (GRU-based Context Encoder).

    Processes motion sequences sequentially to encode contextual information
    and text conditioning into a dense history vector.
    """

    def __init__(
        self,
        frame_feature_dim: int,  # HumanML3D dimension (e.g., 263 or 137 subset)
        text_embedding_dim: int,  # CLIP embedding dimension (e.g., 512)
        joint_feature_projection_dim: int,
        text_projection_dim: int,
        per_joint_out_dim: int,
        joint_count: int = 22,
        model_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        text_encoder: Optional[nn.Module] = None,
    ) -> None:
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

        # Text conditioning projection
        self.text_projection = nn.Linear(text_embedding_dim, text_projection_dim)

        # Dual-MLP Logic Start:
        # Per-joint encoder: Projects raw motion features + text + duration into joint-latent space
        # Input dim: pos(3) + vel(3) + global(8) + duration(1) + text(text_proj_dim) = 15 + text_proj_dim
        joint_input_dim = 3 + 3 + 8 + 1 + text_projection_dim
        self.per_joint_encoder = nn.Sequential(
            nn.Linear(joint_input_dim, joint_feature_projection_dim * 2),
            nn.SiLU(),
            nn.Linear(joint_feature_projection_dim * 2, joint_feature_projection_dim),
        )

        # Temporal core (GRU)
        self.gru = nn.GRU(
            input_size=joint_feature_projection_dim * joint_count,
            hidden_size=model_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Per-joint head: Projects temporal context back to joint-specific features
        num_directions = 2 if bidirectional else 1
        gru_hidden_dim = num_layers * num_directions * model_dim
        intermediate_dim = model_dim * 2
        self.per_joint_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, joint_count * per_joint_out_dim),
        )

        # Null tokens for zero-shot generation (when no history exists)
        self.null_history = nn.Parameter(torch.zeros(1, 1, frame_feature_dim))
        self.null_duration = nn.Parameter(torch.zeros(1, 1))
        self.null_text_embedding = nn.Parameter(torch.zeros(1, text_embedding_dim))

    def _encode_text(
        self, text: Union[str, List[str], torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        """Encodes text or returns tensors and ensures it matches the batch size."""
        if isinstance(text, torch.Tensor):
            if text.shape[0] != batch_size:
                # Handle case where text is a single embedding being broadcasted
                if text.shape[0] == 1:
                    return text.expand(batch_size, -1)
                raise ValueError(
                    f"Batch mismatch: text tensor({text.shape[0]}) vs batch({batch_size})"
                )
            return text

        if self.text_encoder is None:
            raise ValueError("text_encoder must be provided when passing string text.")

        if isinstance(text, str):
            # Encode single string and broadcast to batch size
            single_embedding = self.text_encoder(text)
            if single_embedding.dim() == 1:
                single_embedding = single_embedding.unsqueeze(0)
            return single_embedding.expand(batch_size, -1)
        elif isinstance(text, list):
            if len(text) != batch_size:
                raise ValueError(
                    f"Batch mismatch: text({len(text)}) vs batch({batch_size})"
                )
            return self.text_encoder(text)
        else:
            raise TypeError(
                f"Expected text to be str, List[str], or torch.Tensor, got {type(text)}"
            )

    def forward(
        self,
        text: Union[str, List[str], torch.Tensor],
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text:           (B, 512) tensor, list of strings, or single string
            input_features: (B, T_hist, frame_feature_dim) - Optional history
            total_duration: (B, 1) Normalized total frames (Optional)

        Returns:
            joint_features: (B, 22, per_joint_out_dim) - Encoded per-joint context
        """
        # 1) Determine Batch Size from text input
        if isinstance(text, list):
            B = len(text)
        else:
            B = 1

        # 2) Handle text conditioning
        if text is None:
            text_embeddings = self.null_text_embedding.expand(B, -1)
        else:
            text_embeddings = self._encode_text(text, B)
        text_projected: torch.Tensor = self.text_projection(text_embeddings)

        # 3) Handle optional motion features (Null-token if none)
        if input_features is None:
            # Start of sequence: use one frame of learned "start" features
            input_features = self.null_history.expand(B, 1, -1)

        _, T_hist, _ = input_features.shape

        # Expand text across time and joints for per-joint encoding
        text_projected_expanded = (
            text_projected.unsqueeze(1)
            .unsqueeze(2)
            .expand(B, T_hist, self.joint_count, -1)
        )

        # 4) Handle duration conditioning (Null-token if none)
        if total_duration is None:
            duration_expanded = self.null_duration.expand(
                B, T_hist, self.joint_count, 1
            )
        else:
            duration_expanded = (
                total_duration.unsqueeze(1)
                .unsqueeze(2)
                .expand(B, T_hist, self.joint_count, 1)
            )

        # 3) Extract motion features (263D Standard Layout)
        # - RIC position (Indices 4:67 for 21 joints)
        ric_joints = input_features[:, :, 4:67]
        ric_joints = ric_joints.view(B, T_hist, self.joint_count - 1, 3)
        root_ric = torch.zeros((B, T_hist, 1, 3), device=ric_joints.device)
        ric_joints = torch.cat([root_ric, ric_joints], dim=2)  # (B, T_hist, 22, 3)

        # - Local velocities (Indices 193:259 for 22 joints)
        ric_vel = input_features[:, :, 193:259]
        ric_vel = ric_vel.view(B, T_hist, self.joint_count, 3)  # (B, T_hist, 22, 3)

        # - Global features (Root Rot Vel, Lin Vel, Height, Foot Contacts)
        global_features = torch.cat(
            [input_features[:, :, 0:4], input_features[:, :, 259:263]], dim=-1
        )  # (B, T_hist, 8)
        global_features = global_features.unsqueeze(-2).expand(
            B, T_hist, self.joint_count, -1
        )

        # 4) Concatenate and encode per-joint
        motion_tokens = torch.cat(
            [
                ric_joints,
                ric_vel,
                global_features,
                duration_expanded,
                text_projected_expanded,
            ],
            dim=-1,
        )  # (B, T_hist, 22, input_dim)

        fused_tokens: torch.Tensor = self.per_joint_encoder(motion_tokens)

        # 4) Temporal processing via GRU
        # Flatten joint dimension into the temporal feature vector
        temporal_input = fused_tokens.view(B, T_hist, -1)
        _, final_hidden_raw = self.gru(temporal_input)

        # 5) Transform final hidden state to batch-first
        final_hidden_vec = final_hidden_raw.transpose(0, 1).reshape(B, -1)

        # 6) Project back to per-joint feature space
        output: torch.Tensor = self.per_joint_head(final_hidden_vec)
        joint_features = output.view(
            B, self.joint_count, self.per_joint_out_dim
        )  # (B, 22, per_joint_out_dim)

        return joint_features

    @property
    def output_dim(self) -> int:
        """Output dimension of the context encoder (H)."""
        return self.model_dim * (2 if self.bidirectional else 1)


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


class FlowMatchingPredictor(nn.Module):
    """
    Spatial Transformer-based Flow Matching Predictor (ARFM-style).

    This model predicts the noise/velocity vector to be subtracted from
    noisy targets to perform flow matching in joint space.
    """

    def __init__(
        self,
        per_joint_dim: int = 32,
        model_dim: int = 64,
        num_layers: int = 2,
        joint_count: int = 22,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.joint_count = joint_count

        # Hierarchical Kinematic Encoder
        self.kinematic_encoder = KinematicChainEncoder(model_dim)

        # Fuse condition (history) + spatial data (pos/rot/diffs) + noisy state + progress + noise time
        # Dim: history(32) + pos(3) + rot(6) + diffs(3) + noisy(3) + progress(1) + t(1) = 49
        self.input_proj = nn.Linear(per_joint_dim + 3 + 6 + 3 + 3 + 1 + 1, model_dim)

        # Spatial transformer (22 joints) - ARFM core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )

        # Predict flow/noise per joint
        self.noise_pred = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 3)
        )

        # Null tokens for zero-shot and unspecified signals
        self.null_prev_frame = nn.Parameter(torch.zeros(1, 22, 12))
        self.null_progress = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(
        self,
        history_features: torch.Tensor,
        noise_level: torch.Tensor,
        noisy_target_diffs: Optional[torch.Tensor] = None,
        prev_frame_features: Optional[torch.Tensor] = None,
        temporal_progress: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_features:     (B, 22, per_joint_dim) - Context from History Encoder
            noise_level:          (B,)                   - Flow time t in [0, 1]
            noisy_target_diffs:   (B, 22, 3)             - The noisy state x_t (Optional)
            prev_frame_features:  (B, 22, 12)            - [pos(3) + rot(6) + diffs(3)] (Optional)
            temporal_progress:    (B,)                   - Normalized progress [0, 1] (Optional)

        Returns:
            pred_noise:           (B, 22, 3)             - Predicted flow/velocity v_t
        """
        B, joint_count, _ = history_features.shape

        # 1) Handle noisy state x_t (Sample if None for zero-shot inference)
        if noisy_target_diffs is None:
            noisy_target_diffs = torch.randn(
                (B, joint_count, 3), device=history_features.device
            )

        # 2) Handle temporal progress (Optional: use learned null token)
        if temporal_progress is None:
            p = self.null_progress.expand(B, joint_count, 1)
        else:
            p = temporal_progress.view(B, 1, 1).expand(B, joint_count, 1)

        # 3) Handle previous frame context (Optional: use learned null token)
        if prev_frame_features is None:
            prev_frame_features = self.null_prev_frame.expand(B, joint_count, 12)

        # 4) Construct concatenated conditioning vector
        # Sequence: history(32) + spatial(12) + noisy(3) + progress(1) + time(1) = 49
        x = torch.cat(
            [history_features, prev_frame_features, noisy_target_diffs], dim=-1
        )  # [B, 22, 47]

        t = noise_level.view(B, 1, 1).expand(B, joint_count, 1)
        cond = torch.cat([x, p, t], dim=-1)  # [B, 22, 49]

        x = self.input_proj(cond)  # [B, 22, model_dim]

        # Add learnable kinematic hierarchical bias
        joint_ids = torch.arange(joint_count, device=x.device)
        kinematic_bias = self.kinematic_encoder(joint_ids)  # [22, model_dim]
        x = x + kinematic_bias.unsqueeze(0)  # [B, 22, model_dim]

        # Spatial attention across all 22 joints
        x = self.spatial_transformer(x)  # [B, 22, model_dim]

        # Predict noise to subtract
        pred_noise: torch.Tensor = self.noise_pred(x)  # [B, 22, 3]

        return pred_noise


class HumanMotionGenerator(nn.Module):
    """
    Top-level wrapper for the Human Motion Generation pipeline.
    Integrates MotionHistoryEncoder (Context) and FlowMatchingPredictor (Spatial Generation).
    """

    def __init__(
        self,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def generate(
        self,
        text: Union[str, List[str], torch.Tensor],
        num_steps: int = 10,
        guidance_scale: float = 2.5,
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        prev_frame_features: Optional[torch.Tensor] = None,
        temporal_progress: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Iterative generation loop (Inference).
        Refines noise into clean motion features for the next frame(s).

        Args:
            text:           Text prompt or pre-encoded CLIP embedding
            num_steps:      Number of integration steps (t=0 to t=1)
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger text adherence)
            input_features: (B, T_hist, 263) - Optional history
            ...
        """
        self.eval()
        with torch.no_grad():
            if isinstance(text, list):
                B = len(text)
            elif isinstance(text, torch.Tensor):
                B = text.shape[0]
            else:
                B = 1
            device = next(self.parameters()).device

            # 1. Encode Context once per frame-generation step
            # For CFG, we encode both:
            # - Conditional: using the provided text
            # - Unconditional: using None (which triggers null_text_embedding)
            context_cond = self.encoder(
                text=text, input_features=input_features, total_duration=total_duration
            )

            # Unconditional branch: maintains temporal context but drops text
            context_uncond = self.encoder(
                text=None, input_features=input_features, total_duration=total_duration
            )

            # 2. Initialize noisy state x_0 (Pure Gaussian Noise)
            # Shape: (B, 22, 3) - predicting the next displacement/delta
            x_t = torch.randn((B, 22, 3), device=device)

            # 3. Iterative Refinement (Euler ODE Solver with CFG)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                # t goes from 0 to 1
                t = torch.full((B,), step * dt, device=device)

                # Conditional velocity
                v_cond = self.predictor(
                    history_features=context_cond,
                    noise_level=t,
                    noisy_target_diffs=x_t,
                    prev_frame_features=prev_frame_features,
                    temporal_progress=temporal_progress,
                )

                # Unconditional velocity
                v_uncond = self.predictor(
                    history_features=context_uncond,
                    noise_level=t,
                    noisy_target_diffs=x_t,
                    prev_frame_features=prev_frame_features,
                    temporal_progress=temporal_progress,
                )

                # CFG Interpolation: v = v_uncond + s * (v_cond - v_uncond)
                v_t = v_uncond + guidance_scale * (v_cond - v_uncond)

                # Euler step: x_{t+dt} = x_t + v_t * dt
                x_t = x_t + v_t * dt

            return x_t
