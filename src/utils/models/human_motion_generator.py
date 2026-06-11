import os
import pathlib
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch


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
    predictor,
    track_features: torch.Tensor,
    current_frame_features: Optional[torch.Tensor] = None,
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
        raise ValueError(f"time_schedule_power must be positive, got {time_schedule_power}")

    flow_dim = 68
    if initial_state is None:
        x_t = torch.randn(
            (batch_size, flow_dim),
            device=device,
            dtype=dtype,
        )
    else:
        if initial_state.shape != (batch_size, flow_dim):
            raise ValueError(
                f"Expected initial_state shape ({batch_size}, {flow_dim}), got {tuple(initial_state.shape)}"
            )
        x_t = initial_state.to(device=device, dtype=dtype)

    tau = _build_inference_time_boundaries(
        num_steps,
        power=float(time_schedule_power),
        device=device,
        dtype=dtype,
    )

    use_cfg = float(guidance_scale) != 1.0 and (
        unconditional_text_embedding is not None or unconditional_track_features is not None
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
            noisy_states=noisy_features,
            timesteps=timesteps_batch,
            track_features=track_features,
            current_frame_features=current_frame_features,
            text_embedding=text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        if not use_cfg:
            return cond_velocity

        cfg_text_embedding = (
            unconditional_text_embedding if unconditional_text_embedding is not None else text_embedding
        )
        cfg_track_features = (
            unconditional_track_features if unconditional_track_features is not None else track_features
        )
        uncond_velocity = predictor(
            noisy_states=noisy_features,
            timesteps=timesteps_batch,
            track_features=cfg_track_features,
            current_frame_features=current_frame_features,
            text_embedding=cfg_text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        return uncond_velocity + float(guidance_scale) * (cond_velocity - uncond_velocity)

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
                    raise ValueError(f"Pre-encoded text must have shape (B, 1, 512); got {tuple(text.shape)}")
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
                    seed_positions = input_positions.unsqueeze(1).clone()  # (B, 1, N, 3)
                elif input_positions.ndim == 4:
                    # Sequence (B, T, N, 3)
                    seed_positions = input_positions.clone()  # (B, T, N, 3)
                else:
                    raise ValueError("input_positions must be shape (B, N, 3) or (B, T, N, 3)")

                position_history = seed_positions
                # Seeded: convert global positions to 271D feature history
                feature_history = sequence_joints_to_features(seed_positions, dataset_type=dataset_type)  # (B, T, 271)

            fk_offsets = get_fk_offsets(position_history) if use_fk else None  # (B, 22, 3)

            feature_history = (
                self.normalizer.normalize(feature_history) if self.normalizer is not None else feature_history
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
            predictor_uncond_text_emb = torch.zeros_like(text_emb) if use_cfg and guidance_drop_text else None
            frame_buffer = feature_history[:, :-1] if feature_history.shape[1] > 1 else None
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
                predictor_uncond_context = torch.zeros_like(context_cond) if use_cfg and guidance_drop_context else None

                # ========================================
                # Step C: Flow matching ODE loop
                # x_t starts as random noise in normalized reduced flow space.
                # ========================================
                x_t = integrate_flow_ode(
                    predictor=self.predictor,
                    track_features=context_cond,
                    current_frame_features=None,
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
                flow_output_raw = self.normalizer.denormalize_x68(x_t) if self.normalizer is not None else x_t
                current_frame_raw = (
                    self.normalizer.denormalize(current_frame) if self.normalizer is not None else current_frame
                )
                new_positions = x68_to_positions(
                    flow_output_raw,
                    prev_root_pos=current_positions[:, 0],
                    prev_root_rot_6d=current_frame_raw[:, 69:75],
                )
                relative_shift = new_positions - current_positions
                new_frame, _, fk_positions = generated_positions_to_x271(
                    new_positions=new_positions,
                    prev_positions=current_positions,
                    dataset_type=dataset_type,
                    normalizer=self.normalizer,
                    fk_offsets=fk_offsets,
                )  # (B, 271), (B, 3), (B, 22, 3)

                if fk_positions is not None:
                    new_positions = fk_positions  # Override with FK-corrected positions if available
                    relative_shift = new_positions - current_positions  # Recompute relative shift after FK correction

                # ========================================
                # Step F: Update tracker and history
                # ========================================
                position_history = torch.cat([position_history, new_positions.unsqueeze(1)], dim=1)  # (B, T+1, 22, 3)
                feature_history = torch.cat([feature_history, new_frame.unsqueeze(1)], dim=1)  # (B, N+1, 271)
                relative_shift_history = torch.cat(
                    [relative_shift_history, relative_shift.unsqueeze(1)], dim=1
                )  # (B, T, 22, 3)

                if (frame_idx + 1) % 50 == 0:
                    print(f"Generated {frame_idx + 1}/{num_frames} frames")

            return position_history, feature_history, relative_shift_history

    def generate_sequence_masked(
        self,
        text: Union[str, List[str], torch.Tensor],
        input_positions: Optional[torch.Tensor] = None,
        num_future_frames: int = 10,
        num_steps: int = 10,
        guidance_scale: float = 1.0,
        guidance_drop_text: bool = True,
        guidance_drop_context: bool = False,
        dataset_type: str = "t2m",
        use_fk: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple future frames using masked token prediction.

        Appends mask tokens to the known history in a single encoder forward pass,
        then predicts each masked frame sequentially (autoregressive over masked positions).

        Args:
            text: Text prompt(s) - str, List[str], or pre-encoded tensor (B, 1, 512)
            input_positions: Optional initial global positions (B, N, 22, 3) or (B, 22, 3)
            num_future_frames: Number of frames to predict
            num_steps: Number of flow matching ODE steps per frame
            guidance_scale: CFG strength
            guidance_drop_text: Zero text embedding in CFG branch
            guidance_drop_context: Zero encoder context in CFG branch
            dataset_type: Dataset type for feature extraction
            use_fk: Whether to use FK-corrected positions
        Returns:
            position_history: (B, N+num_future_frames, 22, 3)
            feature_history: (B, N+num_future_frames, 271)
            relative_shift_history: (B, N+num_future_frames, 22, 3)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(text, str):
                from utils.text_encoder import CLIPEncoder

                clip_encoder = CLIPEncoder()
                text = clip_encoder(text)
                B = 1
            elif isinstance(text, list):
                from utils.text_encoder import CLIPEncoder

                clip_encoder = CLIPEncoder()
                text: torch.Tensor = clip_encoder(text)
                B = text.shape[0]
            else:
                if text.ndim != 3 or text.shape[1] != 1:
                    raise ValueError(f"Pre-encoded text must have shape (B, 1, 512); got {tuple(text.shape)}")
                B = text.shape[0]
            device = next(self.parameters()).device
            text = text.to(device=device)

            # ---- History initialization ----
            if input_positions is None:
                joint_count = int(self.config.num_joints)
                joint_dim = int(self.config.joint_dim)
                position_history = torch.zeros((B, 1, joint_count, joint_dim), device=device)
                feature_history = sequence_joints_to_features(position_history, dataset_type=dataset_type)
            else:
                input_positions = input_positions.to(device=device)
                if input_positions.ndim == 3:
                    seed_positions = input_positions.unsqueeze(1).clone()
                elif input_positions.ndim == 4:
                    seed_positions = input_positions.clone()
                else:
                    raise ValueError("input_positions must be shape (B, N, 3) or (B, T, N, 3)")
                position_history = seed_positions
                feature_history = sequence_joints_to_features(seed_positions, dataset_type=dataset_type)

            fk_offsets = get_fk_offsets(position_history) if use_fk else None

            feature_history = (
                self.normalizer.normalize(feature_history) if self.normalizer is not None else feature_history
            )

            relative_shift_history = torch.zeros(
                (B, 1, self.config.num_joints, self.config.joint_dim),
                device=device,
            )
            if position_history.shape[1] > 1:
                relative_shift_history = torch.cat(
                    [relative_shift_history, position_history[:, 1:] - position_history[:, :-1]],
                    dim=1,
                )

            text_emb = text[:, 0, :]
            use_cfg = float(guidance_scale) != 1.0
            if use_cfg and not (guidance_drop_text or guidance_drop_context):
                raise ValueError(
                    "guidance_scale requires at least one unconditional branch input; "
                    "set guidance_drop_text and/or guidance_drop_context."
                )
            predictor_uncond_text_emb = torch.zeros_like(text_emb) if use_cfg and guidance_drop_text else None

            # ---- Build masked sequence: known frames + mask tokens ----
            masked_feature_history = feature_history
            for _ in range(num_future_frames):
                masked_feature_history = torch.cat(
                    [masked_feature_history, self.encoder.mask_token.unsqueeze(0).expand(B, -1, -1)],
                    dim=1,
                )

            mask = torch.zeros(B, masked_feature_history.shape[1], dtype=torch.bool, device=device)
            mask[:, -num_future_frames:] = True

            # ---- Single encoder forward pass ----
            encoded = self.encoder(masked_feature_history, text_emb, mask=mask)
            predictor_uncond_context = torch.zeros_like(encoded) if use_cfg and guidance_drop_context else None

            # ---- Predict each masked frame autoregressively ----
            for frame_idx in range(num_future_frames):
                hist_len = feature_history.shape[1]
                current_positions = position_history[:, -1]
                current_frame = feature_history[:, -1]

                context_cond = encoded[:, hist_len + frame_idx, :]

                x_t = integrate_flow_ode(
                    predictor=self.predictor,
                    track_features=context_cond,
                    current_frame_features=None,
                    text_embedding=text_emb,
                    num_steps=num_steps,
                    time_schedule_power=self.config.inference_t_schedule_power,
                    guidance_scale=guidance_scale,
                    unconditional_text_embedding=predictor_uncond_text_emb,
                    unconditional_track_features=predictor_uncond_context[:, hist_len + frame_idx, :]
                    if predictor_uncond_context is not None
                    else None,
                )

                flow_output_raw = self.normalizer.denormalize_x68(x_t) if self.normalizer is not None else x_t
                current_frame_raw = (
                    self.normalizer.denormalize(current_frame) if self.normalizer is not None else current_frame
                )
                new_positions = x68_to_positions(
                    flow_output_raw,
                    prev_root_pos=current_positions[:, 0],
                    prev_root_rot_6d=current_frame_raw[:, 69:75],
                )
                relative_shift = new_positions - current_positions
                new_frame, _, fk_positions = generated_positions_to_x271(
                    new_positions=new_positions,
                    prev_positions=current_positions,
                    dataset_type=dataset_type,
                    normalizer=self.normalizer,
                    fk_offsets=fk_offsets,
                )

                if fk_positions is not None:
                    new_positions = fk_positions
                    relative_shift = new_positions - current_positions

                position_history = torch.cat([position_history, new_positions.unsqueeze(1)], dim=1)
                feature_history = torch.cat([feature_history, new_frame.unsqueeze(1)], dim=1)
                relative_shift_history = torch.cat(
                    [relative_shift_history, relative_shift.unsqueeze(1)],
                    dim=1,
                )

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
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "config" in checkpoint:
            config: Config = checkpoint["config"]

        encoder = MotionHistoryEncoder(config.encoder_config).to(device)

        # Initialize Flow Matching Predictor with new config-based interface
        predictor_config = config.predictor_config
        feature_size = config.encoder_config.hidden_size

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
