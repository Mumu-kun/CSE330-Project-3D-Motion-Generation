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
from pathlib import Path
from typing import Optional, List, Tuple, Union, Callable, Any
from config import Config

from utils.motion_utils import (
    feature_to_joints,
    get_dataset_config,
    IncrementalFeatureExtractor,
)


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
        self.joint_count = joint_count

        # Text conditioning projection
        self.text_projection = nn.Linear(text_embedding_dim, text_projection_dim)

        # Dual-MLP Logic Start:
        # Per-joint encoder: Projects raw motion features + text + duration into joint-latent space
        # Input dim: pos(3) + vel(3) + global(8) + duration(1) + text(text_proj_dim) = 15 + text_proj_dim
        joint_input_dim = 3 + 3 + 8 + text_projection_dim
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

    def forward(
        self,
        text: Optional[torch.Tensor],
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            text:           (B, 512) tensor or None
            input_features: (B, T_hist, frame_feature_dim) - Optional history
            total_duration: (B, 1) Normalized total frames (Optional)

        Returns:
            joint_features: (B, 22, per_joint_out_dim) - Encoded per-joint context
        """
        # 1) Determine Batch Size
        if input_features is not None:
            B = input_features.shape[0]
        elif text is not None:
            B = text.shape[0]
        elif total_duration is not None:
            B = total_duration.shape[0]
        elif batch_size is not None:
            B = batch_size
        else:
            B = 1

        # 2) Handle text conditioning
        if text is None:
            text_embeddings = self.null_text_embedding.expand(B, -1)
        else:
            # Handle broadcasting if text is (1, D) but B > 1
            if text.shape[0] != B:
                if text.shape[0] == 1:
                    text_embeddings = text.expand(B, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch: text tensor({text.shape[0]}) vs batch({B})"
                    )
            else:
                text_embeddings = text

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
        )  # (B, T_hist, 22, text_projection_dim)

        # 4) Handle duration conditioning (Null-token if none)
        # if total_duration is None:
        #     duration_expanded = self.null_duration.expand(
        #         B, T_hist, self.joint_count, 1
        #     )
        # else:
        #     # total_duration is (B, 1), expand to (B, T_hist, joint_count, 1)
        #     duration_expanded = total_duration.view(B, 1, 1, 1).expand(
        #         B, T_hist, self.joint_count, 1
        #     )

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
                # duration_expanded,
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
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.joint_count = joint_count
        self.time_embed_dim = time_embed_dim

        # Hierarchical Kinematic Encoder
        self.kinematic_encoder = KinematicChainEncoder(model_dim)

        # Sinusoidal time embedding for noise_level (flow time t)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Fuse condition (history) + spatial data (pos/rot/diffs) + noisy state
        # Dim: history(32) + pos(3) + rot(6) + diffs(3) + noisy(3) = 47
        self.input_proj = nn.Linear(per_joint_dim + 3 + 6 + 3 + 3, model_dim)

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

    def _sinusoidal_time_embedding(
        self, t: torch.Tensor, max_positions=10000
    ) -> torch.Tensor:
        """
        Compute sinusoidal time embedding for flow time t.

        Args:
            t: (B,) tensor of time values in [0, 1]

        Returns:
            (B, time_embed_dim) tensor of sinusoidal embeddings
        """
        half_dim = self.time_embed_dim // 2
        # Compute frequencies: exp(log(10000) * (2i / d))
        freqs = torch.exp(
            -torch.log(torch.tensor(max_positions))
            / (half_dim - 1)
            * torch.arange(half_dim, device=t.device)
        )
        # t: (B,) -> (B, half_dim)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        # Sin/cos embedding
        emb = torch.cat(
            [torch.sin(args), torch.cos(args)], dim=-1
        )  # (B, time_embed_dim)

        if self.time_embed_dim % 2 != 0:
            emb = nn.functional.pad(emb, (0, 1), mode="constant")

        return emb

    def forward(
        self,
        history_features: torch.Tensor,
        noise_level: torch.Tensor,
        noisy_target: Optional[torch.Tensor] = None,
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

        assert (
            B == noise_level.shape[0]
        ), f"noise level {noise_level.shape} does not match batch size {B}"
        if temporal_progress is not None:
            assert (
                B == temporal_progress.shape[0]
            ), f"temporal progress {temporal_progress.shape} does not match batch size {B}"
        if prev_frame_features is not None:
            assert (
                B == prev_frame_features.shape[0]
            ), f"prev frame features {prev_frame_features.shape} does not match batch size {B}"
        if noisy_target is not None:
            assert (
                B == noisy_target.shape[0]
            ), f"noisy target diffs {noisy_target.shape} does not match batch size {B}"

        # 1) Handle noisy state x_t (Sample if None for zero-shot inference)
        if noisy_target is None:
            noisy_target = torch.randn(
                (B, joint_count, 3), device=history_features.device
            )

        # 2) Handle temporal progress (Optional: use learned null token)
        # if temporal_progress is None:
        #     p = self.null_progress.expand(B, joint_count, 1)
        # else:
        #     p = temporal_progress.reshape(B, 1, 1).expand(B, joint_count, 1)

        # 3) Handle previous frame context (Optional: use learned null token)
        if prev_frame_features is None:
            prev_frame_features = self.null_prev_frame.expand(B, joint_count, 12)

        # 4) Construct concatenated conditioning vector
        # Sequence: history(32) + spatial(12) + noisy(3) = 47
        x = torch.cat(
            [history_features, prev_frame_features, noisy_target], dim=-1
        )  # [B, 22, 47]

        x = self.input_proj(x)  # [B, 22, model_dim]

        # 5) Compute sinusoidal time embedding and add as bias (like kinematic bias)
        time_embed = self._sinusoidal_time_embedding(noise_level)  # (B, time_embed_dim)
        time_bias = self.time_embed(time_embed)  # (B, model_dim)
        x = x + time_bias.unsqueeze(1)  # [B, 22, model_dim]

        # 6) Add learnable kinematic hierarchical bias
        joint_ids = torch.arange(joint_count, device=x.device)
        kinematic_bias = self.kinematic_encoder(joint_ids)  # [22, model_dim]
        x = x + kinematic_bias.unsqueeze(0)  # [B, 22, model_dim]

        # 7) Spatial attention across all 22 joints
        x = self.spatial_transformer(x)  # [B, 22, model_dim]

        # 8) Predict noise to subtract
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

    """
    Updated generate_sequence() methods for HumanMotionGenerator class
    Returns GLOBAL JOINT POSITIONS instead of feature vectors

    Add these methods to the HumanMotionGenerator class in models.py
    """

    def generate_sequence(
        self,
        text: Union[str, List[str], torch.Tensor],
        num_frames: int = 200,
        num_steps: int = 10,
        guidance_scale: float = 2.5,
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        dataset_type: str = "t2m",
    ) -> torch.Tensor:
        """
        Generate n consecutive animation frames where each frame feeds into the next iteration.
        Uses torch-based incremental feature extraction.

        Returns:
            joint_positions: (B, num_frames, 22, 3) - Global joint positions
        """

        self.eval()
        with torch.no_grad():
            if isinstance(text, torch.Tensor):
                B = text.shape[0]
            else:
                B = 1
            device = next(self.parameters()).device

            T_hist = 15

            # Initialize history
            if input_features is None:
                history = self.encoder.null_history.expand(B, T_hist, -1).clone()
            else:
                if input_features.shape[1] >= T_hist:
                    history = input_features[:, -T_hist:, :]
                else:
                    pad_size = T_hist - input_features.shape[1]
                    null_pad = self.encoder.null_history.expand(B, pad_size, -1).clone()
                    history = torch.cat([null_pad, input_features], dim=1)

            # Initialize incremental feature extractor
            config = get_dataset_config(dataset_type)
            extractor = IncrementalFeatureExtractor(
                n_raw_offsets=config["raw_offsets"],
                kinematic_chain=config["kinematic_chain"],
                face_joint_indx=config["face_joint_indx"],
                fid_r=config["fid_r"],
                fid_l=config["fid_l"],
                feet_thre=0.002,
                device=device,
            )

            joint_sequence = []

            # Extract initial global joint positions
            init_frame = history[:, -1, :]  # (B, 263)
            current_joints_global = feature_to_joints(
                init_frame, dataset_type=dataset_type
            )  # (B, 22, 3)

            # Initialize extractor with first frame
            extractor.initialize(current_joints_global)

            for frame_idx in range(num_frames):
                t_prog = torch.full((B,), frame_idx / num_frames, device=device)

                # Encode context
                context_cond = self.encoder(
                    batch_size=B,
                    text=text,
                    input_features=history,
                    total_duration=total_duration,
                )

                context_uncond = self.encoder(
                    batch_size=B,
                    text=None,
                    input_features=history,
                    total_duration=total_duration,
                )

                # Extract previous frame features
                last_frame = history[:, -1, :]
                ric_pos_21 = last_frame[:, 4:67].reshape(B, 21, 3)
                root_pos = torch.zeros((B, 1, 3), device=device, dtype=last_frame.dtype)
                prev_pos_ric = torch.cat([root_pos, ric_pos_21], dim=1)

                prev_rot6d = last_frame[:, 67:193].reshape(B, 21, 6)
                root_rot = torch.zeros((B, 1, 6), device=device, dtype=last_frame.dtype)
                root_rot[:, 0, 0] = 1.0
                root_rot[:, 0, 4] = 1.0
                prev_rot6d = torch.cat([root_rot, prev_rot6d], dim=1)

                prev_v = last_frame[:, 193:259].reshape(B, 22, 3)
                prev_frame_features = torch.cat(
                    [prev_pos_ric, prev_rot6d, prev_v], dim=-1
                )

                # Generate displacement
                x_t = torch.randn((B, 22, 3), device=device)
                dt = 1.0 / num_steps

                for step in range(num_steps):
                    t = torch.full((B,), step * dt, device=device)

                    v_cond = self.predictor(
                        history_features=context_cond,
                        noise_level=t,
                        noisy_target_diffs=x_t,
                        prev_frame_features=prev_frame_features,
                        temporal_progress=t_prog,
                    )

                    v_uncond = self.predictor(
                        history_features=context_uncond,
                        noise_level=t,
                        noisy_target_diffs=x_t,
                        prev_frame_features=prev_frame_features,
                        temporal_progress=t_prog,
                    )

                    v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
                    x_t = x_t + v_t * dt

                # Update global joint positions
                new_joints_global = current_joints_global + x_t
                current_joints_global = new_joints_global.clone()

                # Collect joint positions
                joint_sequence.append(new_joints_global.cpu())

                # EFFICIENT: Use torch-based incremental feature extraction
                # Input: (B, 22, 3) torch tensor
                # Output: (B, 263) torch tensor
                new_frame_features = extractor.process_frame(
                    new_joints_global
                )  # Already on device

                # Update history
                history = torch.cat(
                    [history[:, 1:, :], new_frame_features.unsqueeze(1)], dim=1
                )

                if (frame_idx + 1) % 50 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            # Stack all joint positions
            joint_positions = torch.stack(joint_sequence, dim=1).to(
                device
            )  # (B, num_frames, 22, 3)

            return joint_positions

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: Union[str, Path], config: Config, device: str = "cpu"
    ) -> "HumanMotionGenerator":
        """
        Load the generator from a checkpoint file.
        Prefers EMA weights if available.
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize Encoders/Predictors from Config
        encoder = MotionHistoryEncoder(
            frame_feature_dim=config.motion_dim,
            text_embedding_dim=config.text_embedding_dim,
            joint_feature_projection_dim=config.joint_feature_projection_dim,
            text_projection_dim=config.text_projection_dim,
            per_joint_out_dim=config.per_joint_out_dim,
            joint_count=config.num_joints,
            model_dim=config.model_dim,
            num_layers=config.num_encoder_layers,
            bidirectional=config.bidirectional_gru,
        )

        predictor = FlowMatchingPredictor(
            per_joint_dim=config.per_joint_out_dim,
            model_dim=config.model_dim,
            num_layers=config.num_flow_layers,
            joint_count=config.num_joints,
        )

        # Load weights (Prefer EMA)
        if "ema_mhe" in checkpoint and "ema_fmp" in checkpoint:
            print("Loading EMA weights for generation...")
            encoder.load_state_dict(checkpoint["ema_mhe"])
            predictor.load_state_dict(checkpoint["ema_fmp"])
        else:
            print("Loading standard weights (EMA not found)...")
            encoder.load_state_dict(checkpoint["motion_history_encoder"])
            predictor.load_state_dict(checkpoint["flow_predictor"])

        encoder.to(device)
        predictor.to(device)
        encoder.eval()
        predictor.eval()

        return cls(encoder, predictor)
