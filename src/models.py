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
    extract_prev_frame_features,
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
        self.global_to_joints = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, joint_count * per_joint_out_dim),
        )

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
        joint_tokens = self.global_to_joints(h_t)  # (B, 22 * D_joint)
        history_features = joint_tokens.view(
            B, self.joint_count, self.per_joint_out_dim
        )  # (B, 22, D_joint)

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

    def step(
        self,
        x_t: torch.Tensor,  # (B, motion_dim)  single frame features
        text_emb: torch.Tensor,  # (B, text_dim)
        h: Optional[torch.Tensor],  # (L, B, H) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step update for AR inference.
        Returns:
          history_features: (B, 22, per_joint_dim)  summary up to this frame
          h_next: (L, B, H)  next hidden state to carry forward
        """
        motion_in = x_t.unsqueeze(1)  # (B, 1, motion_dim)
        history_features, h_next = self._gru_block(motion_in, text_emb, h)
        return history_features, h_next

    @property
    def output_dim(self) -> int:
        # Per-joint dimension exposed to FlowMatchingPredictor
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

        # --- Project history ---
        self.input_proj_history = nn.Linear(
            per_joint_dim, model_dim
        )  # only history embeddings

        # --- Project noisy target separately ---
        self.input_proj_noisy_root = nn.Linear(9, model_dim)  # noisy root features
        self.input_proj_noisy_joint = nn.Linear(
            3, model_dim
        )  # noisy joint RIC positions

        self.fusion_proj = nn.Linear(
            3 * model_dim, model_dim
        )  # fusion of history, noisy target, and prev_frame_features

        # Prev frame projections (root + joints)
        self.input_proj_prev_root = nn.Linear(9, model_dim)  # root: height + vel + rot
        self.input_proj_prev_joint = nn.Linear(
            12, model_dim
        )  # joints: RIC + rot + velocity

        # --- Learnable bias for when prev_frame_features is None ---
        self.null_prev_bias = nn.Parameter(torch.zeros(joint_count, model_dim))

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
        prev_frame_features: Optional[
            torch.Tensor
        ] = None,  # (B, 261) - prev frame features
        temporal_progress: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for FlowMatchingPredictor.

        Args:
            history_features: (B, 22, per_joint_dim) - Context from MotionHistoryEncoder
            noise_level: (B,) - Flow time t in [0,1]
            noisy_target: (B, 72) - Current noisy state x_t (already in normalized space)
            prev_frame_features: (B, 261) - Features from previous frame (root 9D + joints 252D)
            temporal_progress: (B,) - Optional normalized frame progress

        Returns:
            pred_frame: (B, 72) - Predicted velocity field (in normalized space)
        """
        B, J, _ = history_features.shape
        device = history_features.device

        # --- Project history ---
        cond_proj = self.input_proj_history(history_features)  # (B,22,model_dim)

        # --- Handle noisy target ---
        if noisy_target is None:
            noisy_target = torch.randn(B, 72, device=device)

        noisy_root_in = noisy_target[:, :9]  # (B,9)
        noisy_joints_in = noisy_target[:, 9:].reshape(B, J - 1, 3)  # (B,21,3)

        noisy_root_proj = self.input_proj_noisy_root(noisy_root_in).unsqueeze(1)
        noisy_joint_proj = self.input_proj_noisy_joint(noisy_joints_in)

        noisy_proj = torch.cat(
            [noisy_root_proj, noisy_joint_proj], dim=1
        )  # (B,22,model_dim)

        # --- Handle prev_frame_features ---
        if prev_frame_features is None:
            # Use learned bias when not provided (backwards compatible)
            prev_proj = self.null_prev_bias.unsqueeze(0).expand(
                B, -1, -1
            )  # (B, 22, model_dim)
        else:
            # Split into root and joints
            prev_root = prev_frame_features[:, :9]  # (B, 9)
            prev_joints = prev_frame_features[:, 9:].reshape(
                B, J - 1, -1
            )  # (B, 21, 12)

            # Project separately and concatenate
            prev_root_proj = self.input_proj_prev_root(prev_root).unsqueeze(
                1
            )  # (B, 1, model_dim)
            prev_joints_proj = self.input_proj_prev_joint(
                prev_joints
            )  # (B, 21, model_dim)

            prev_proj = torch.cat(
                [prev_root_proj, prev_joints_proj], dim=1
            )  # (B, 22, model_dim)

        # --- Combine condition + noisy + prev_frame_features ---
        x = torch.cat([cond_proj, noisy_proj, prev_proj], dim=-1)  # (B,22,3*model_dim)
        x = self.fusion_proj(x)  # (B,22,model_dim)

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
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalizer = encoder.normalizer

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
        horizon: int = 16,
        input_features: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
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
            horizon: Number of recent frames to use for context encoding
            input_features: Optional initial motion history (B, N, 271) or (B, 271)
            total_duration: Optional duration tensor (not used)
            dataset_type: Dataset type for feature extraction

        Returns:
            position_history: (B, N+num_frames, 22, 3) - Global joint positions including initial history
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
                # Initialize with a zero frame for cold start
                feature_history = torch.zeros((B, 1, 271), device=device)  # (B, 1, 271)
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
                # Step B: Encode context from last horizon frames
                # normalize=True (default) - models normalize RAW features internally
                # ========================================
                # Slice to last horizon frames
                horizon_frames = min(horizon, feature_history.shape[1])
                encoder_input = feature_history[
                    :, -horizon_frames:, :
                ]  # (B, horizon, 271)

                # Prepare text embedding: (B, 1, 512) -> (B, 512)
                text_emb = text.squeeze(1) if text.dim() == 3 else text

                context_cond = self.encoder(
                    encoder_input,
                    text_emb,
                )  # (B, 22, per_joint_dim)

                # ========================================
                # Step C: Flow matching ODE loop
                # x_t starts as random noise (already in normalized space conceptually)
                # ========================================
                x_t = torch.randn((B, 72), device=device)
                dt = 1.0 / num_steps

                for step in range(num_steps):
                    t = torch.full((B,), step * dt, device=device)

                    # Extract prev_frame_features from last_frame (271D -> 261D)
                    prev_features = extract_prev_frame_features(last_frame)  # (B, 261)

                    # Only conditional prediction (no CFG)
                    v_t = self.predictor(
                        history_features=context_cond,
                        noise_level=t,
                        noisy_target=x_t,
                        prev_frame_features=prev_features,
                    )

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

        # Initialize Flow Matching Predictor
        predictor = FlowMatchingPredictor(
            per_joint_dim=config.predictor_per_joint_dim,
            model_dim=config.predictor_model_dim,
            num_layers=config.predictor_num_layers,
            joint_count=config.encoder_num_joints,
            time_embed_dim=config.predictor_time_embed_dim,
            dropout=config.predictor_dropout,
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
