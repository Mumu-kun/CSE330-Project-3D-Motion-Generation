"""Flow Matching Predictor trainer — Phase 3.

Loads a pretrained MotionHistoryEncoder from checkpoint, freezes it,
and trains a FlowMatchingPredictor using rectified flow matching in
the encoder's latent space.

Training objective: learn v_theta(z_t, t, c) such that integrating
the ODE from t=0 to t=1 transports Gaussian noise to the encoder's
latent distribution, conditioned on text c and motion history.

Key design:
  - Encoder frozen throughout (both online and EMA copies)
  - Predictor trained with separate AdamW + OneCycleLR
  - Power-schedule timestep sampling: t = u^{1/(p+1)}, p=config.t_sampling_power
  - CFG dropout: config.cfg_dropout fraction replace text with a learned null embedding
  - Rectified flow target: v* = z1 - z0, where z_t = (1-t)*z0 + t*z1
  - EMA maintained for the predictor
  - Can run standalone via FlowMatchingTrainer.run() or supply models to a combined loop

Leakage-free latent construction (requires dataset to emit separate tensors):
  - target_motion  (= batch["motion"])         : the sequence being generated
  - history_motion (= batch["history_motion"]) : frames strictly preceding the target,
                                                 zero-padded when unavailable

  z1             = EMA_encoder(target_motion,  zeros_text, return_layer_outputs=True)[:, 1:, -1, :]
  track_features = EMA_encoder(history_motion, zeros_text, return_layer_outputs=False)

  Because target_motion and history_motion are non-overlapping by construction in
  dataset.py, the encoder running on history_motion cannot see any frame that is
  part of the target being generated.  This is true regardless of whether the
  encoder is causal or bidirectional.

  Text conditioning reaches the predictor exclusively through the AdaLN text_embedding
  argument, which is also the sole location of CFG dropout.
"""

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
)

from utils.config import Config
from utils.models import (
    CheckpointMetadata,
    EMAModel,
    PretrainEngine,
    _BaseTrainer,
    estimate_time_remaining,
)
from utils.models.flow_matching_predictor import FlowMatchingPredictor
from utils.models.motion_history_encoder import MotionHistoryEncoder


# ── Windows checkpoint compat ──────────────────────────────────────────────────


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


def _torch_load_with_compat(path, map_location, weights_only):
    """Load checkpoint with Windows path compatibility."""
    with _windows_checkpoint_path_compat():
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
        finally:
            pass
    return checkpoint


# ── Latent space stats container ───────────────────────────────────────────────


class _LatentStats(nn.Module):
    """Container for latent statistics so Ignite can serialize them cleanly."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.norm_std = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.initialized = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)



# ── Learned null embedding ─────────────────────────────────────────────────────


class _NullTextEmbedding(nn.Module):
    """Learned unconditional text embedding for Classifier-Free Guidance.

    Owned by the trainer (not the predictor model) so the predictor
    architecture remains unchanged. Saved in checkpoints as a separate key
    and included in the predictor optimizer's parameter group.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return null embedding expanded to (batch_size, dim)."""
        return self.weight.expand(batch_size, -1)


# ── Trainer ────────────────────────────────────────────────────────────────────


class FlowMatchingTrainer(_BaseTrainer):
    """Phase 3: Flow Matching Predictor trainer.

    Trains FlowMatchingPredictor using rectified flow matching while keeping
    the MotionHistoryEncoder fully frozen.

    Usage (standalone)::

        trainer = FlowMatchingTrainer(
            config=config,
            pretrained_checkpoint_path="checkpoints/pretrain_best_val_xxx.pt",
        )
        trainer.run()

    Usage (combined notebook loop)::

        trainer = FlowMatchingTrainer(config=config, pretrained_checkpoint_path=ckpt)
        # Access trainer.predictor, trainer.ema_predictor, trainer.null_text_embedding,
        # trainer.normalizer, trainer.ema_encoder directly from your notebook loop.
        loss = trainer.compute_flow_loss(motion, text_emb)
    """

    def __init__(
        self,
        config: Config,
        pretrained_checkpoint_path: str | Path,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.pretrained_checkpoint_path = Path(pretrained_checkpoint_path) if pretrained_checkpoint_path is not None else None
        self.wandb_project = wandb_project

        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.use_amp)
        )

        self.session_id = datetime.now(timezone(timedelta(hours=6))).strftime("%Y%m%d_%H%M%S")
        self.phase3_session_id = self.session_id
        self.resume_id: str | None = None

        # ── Load frozen encoder from checkpoint ──
        self._load_pretrained_encoder_state()

        # ── Build predictor + EMA ──
        self.predictor = FlowMatchingPredictor(config).to(self.device)
        self.ema_predictor: EMAModel[FlowMatchingPredictor] = EMAModel(
            self.predictor, decay=float(config.ema_decay)
        ).to(self.device)

        # ── Learned null text embedding for CFG ──
        self.null_text_embedding = _NullTextEmbedding(config.text_embedding_dim).to(self.device)

        # ── Instantiate CLIP Text Sequence Encoder ──
        from utils.text_encoder import CLIPEncoder
        self.clip_encoder = CLIPEncoder().to(self.device)

        # ── Instantiate Latent statistics container ──
        self.latent_stats = _LatentStats(config.encoder_config.hidden_size).to(self.device)

        self.encoder.to(self.device)
        self.ema_encoder.to(self.device)
        self.predictor.to(self.device)

        # ── Freeze encoder completely ──
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ema_encoder.model.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.ema_encoder.model.eval()

        self._log_model_parameters(
            ("MotionHistoryEncoder (frozen)", self.encoder),
            ("FlowMatchingPredictor", self.predictor),
        )

        self.wandb_logger = None
        self._initialize()

    # ── Initialization helpers ─────────────────────────────────────────────────

    def _initialize(self) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._create_dataloaders()
        self._compute_latent_statistics()

        self.accumulation_steps = ceil(self.config.effective_batch_size / self.config.batch_size) or 1

        # Optimizer: predictor params + learned null embedding trained jointly
        self.optimizer = torch.optim.AdamW(
            list(self.predictor.parameters()) + list(self.null_text_embedding.parameters()),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.lr_scheduler = self._setup_scheduler(
            self.optimizer, num_epochs=int(self.config.get_num_epochs())
        )

        self._init_wandb(
            "phase3_flow_matching",
            extra_config={
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "predictor_params": sum(p.numel() for p in self.predictor.parameters()),
                "pretrained_checkpoint": str(self.pretrained_checkpoint_path),
                "cfg_dropout": self.config.cfg_dropout,
                "t_sampling_power": self.config.t_sampling_power,
            },
            name=self.phase3_session_id,
        )

    def _load_pretrained_encoder_state(self) -> None:
        """Load encoder (online + EMA) from a pretrain checkpoint and freeze both."""
        if self.pretrained_checkpoint_path is None:
            raise FileNotFoundError(
                "Pretrained checkpoint path is None. No pretrained checkpoint was found matching prefix 'pretrain_best_val_*.pt' in './checkpoints/pretrain'."
            )
        if not self.pretrained_checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_checkpoint_path}")

        checkpoint = _torch_load_with_compat(
            self.pretrained_checkpoint_path, self.device, weights_only=False
        )

        metadata = checkpoint.get("metadata")
        if isinstance(metadata, CheckpointMetadata):
            metadata = metadata.state_dict()
        self.pretraining_session_id = metadata.get("session_id", "unknown") if metadata else "unknown"

        config_raw = checkpoint.get("config")
        config: Config = Config()
        if isinstance(config_raw, CheckpointMetadata):
            config_raw = config_raw.state_dict()
        if isinstance(config_raw, dict):
            config.load_state_dict(config_raw)

        encoder_state = checkpoint.get("encoder")
        encoder_ema_state = checkpoint.get("encoder_ema")
        if encoder_state is None and encoder_ema_state is None:
            raise KeyError(
                f"Checkpoint {self.pretrained_checkpoint_path} does not contain "
                "encoder or encoder_ema weights"
            )

        primary_state = encoder_state if encoder_state is not None else encoder_ema_state
        ema_state = encoder_ema_state if encoder_ema_state is not None else encoder_state
        assert primary_state is not None

        self.encoder = MotionHistoryEncoder(config).to(self.device)
        self.ema_encoder = EMAModel(self.encoder, decay=float(config.ema_decay)).to(self.device)
        self.encoder.load_state_dict(primary_state)
        self.ema_encoder.load_state_dict(ema_state)

    # ── Core helpers ───────────────────────────────────────────────────────────

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps from power schedule.

        Density: p(t) ∝ t^power → inverse CDF: t = u^{1/(power+1)}.
        Concentrates samples near t→1 (harder high-noise region) where the
        flow matching loss is most informative.

        Returns:
            t: (batch_size,) float32 tensor in [0, 1].
        """
        u = torch.rand(batch_size, device=self.device)
        power = float(self.config.t_sampling_power)
        return u.pow(1.0 / (power + 1.0))

    def _apply_cfg_dropout(
        self,
        text_emb: torch.Tensor,
        batch_size: int,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Stochastically replace text with learned null embedding for CFG training.

        10% of samples use the unconditional (null) embedding so the predictor
        learns both text-conditioned and text-free trajectory distributions.
        No dropout is applied during validation.
        """
        dropout_prob = float(self.config.cfg_dropout)
        if dropout_prob <= 0.0 or not is_training:
            return text_emb
        cfg_mask = torch.rand(batch_size, device=self.device) < dropout_prob  # (B,) bool
        null_emb = self.null_text_embedding(batch_size).to(dtype=text_emb.dtype)  # (B, 512)

        # If text_emb is a sequence (B, S, 512), expand the null embedding along sequence dimension
        if text_emb.ndim == 3:
            S = text_emb.shape[1]
            null_seq_emb = null_emb.unsqueeze(1).expand(-1, S, -1)
            return torch.where(cfg_mask.unsqueeze(-1).unsqueeze(-1), null_seq_emb, text_emb)

        return torch.where(cfg_mask.unsqueeze(-1), null_emb, text_emb)

    def _compute_latent_statistics(self) -> None:
        """Compute running mean and std of the encoder's latents over the training dataset."""
        if self.latent_stats.initialized.item():
            print("Latent space statistics already loaded from checkpoint.")
            return

        print("Pre-computing latent space normalization statistics...")

        def _collect_z1(loader, num_batches):
            all_z1 = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= num_batches:
                        break
                    motion_raw = batch["motion"].to(self.device)
                    target_motion = self.normalizer.normalize(motion_raw)
                    zeros_text = torch.zeros(target_motion.shape[0], self.config.text_embedding_dim,
                                             device=self.device, dtype=target_motion.dtype)
                    raw_target = self.ema_encoder.model(target_motion, zeros_text, mask=None, return_layer_outputs=True)
                    z1 = raw_target[:, 1:, -1, :].detach()  # (B, T-1, H)
                    all_z1.append(z1.cpu())
            return torch.cat(all_z1, dim=0).flatten(0, 1)  # (N, H)

        combined_z1 = _collect_z1(self.train_loader, num_batches=20)
        latent_mean = combined_z1.mean(dim=0).to(self.device)
        latent_std = torch.clamp(combined_z1.std(dim=0).to(self.device), min=1e-5)

        # Compute per-channel std on independent held-out validation sample
        holdout_z1 = _collect_z1(self.val_loader, num_batches=5)
        normalized_holdout = (holdout_z1.to(self.device) - latent_mean) / latent_std
        norm_std = torch.clamp(normalized_holdout.std(dim=0), min=1e-5)

        self.latent_stats.mean.copy_(latent_mean)
        self.latent_stats.std.copy_(latent_std)
        if hasattr(self.latent_stats, "norm_std"):
            self.latent_stats.norm_std.copy_(norm_std)
        self.latent_stats.initialized.copy_(torch.tensor(True, device=self.device))
        print("Latent statistics computed successfully.")


    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize target latent z to have mean 0 and std 1 channel-wise."""
        mean = self.latent_stats.mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_stats.std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Denormalize latent z back to its original scale and shift."""
        mean = self.latent_stats.mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_stats.std.to(device=z.device, dtype=z.dtype)
        return z * std + mean

    # ── Latent computation (usable from external combined loops) ───────────────

    @torch.no_grad()
    def compute_target_latents(
        self,
        target_motion: torch.Tensor,
        history_motion: torch.Tensor,
        text_emb: torch.Tensor,
        history_valid_length: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frozen encoder outputs for flow matching, with no leakage.

        target_motion and history_motion are non-overlapping by construction
        (dataset.py guarantees this).  The encoder is bidirectional, so the only
        way to prevent leakage is to run it on two disjoint input sequences.

        Args:
            target_motion:  Normalised target window (B, T_target, 271).  This is the
                            sequence being generated.  Used to produce z1.
            history_motion: Normalised history window (B, T_history, 271).  Frames
                            strictly preceding the target window.  Used to produce
                            track_features.  Zero-padded at front when unavailable.
            text_emb:       Text embedding (B, D).  Used only to build a zeros tensor
                            of the right shape/dtype; NOT passed to the encoder.
            history_valid_length: Optional (B,) tensor indicating number of real history frames.

        Returns:
            z1:             Target latents (B, T_target-1, H_enc) — last encoder layer,
                            token index 1 onwards (matches decoder_trainer convention).
            track_features: History context (B, T_history, H_enc) — last encoder layer.
                            Encoder saw only history frames; no target frame information.
        """
        target_enc = self.ema_encoder.model
        target_enc.eval()

        zeros_text = torch.zeros_like(text_emb)

        # z1 from target_motion (what the predictor must learn to generate)
        raw_target = target_enc(
            target_motion, zeros_text, mask=None, return_layer_outputs=True
        )  # (B, T_target, L, H_enc)
        z1 = raw_target[:, 1:, -1, :].detach()  # (B, T_target-1, H_enc)

        # track_features from history_motion (disjoint from target — no leakage possible)
        # Text is zeroed: conditioning lives exclusively in AdaLN (text_embedding arg).
        zeros_text_hist = torch.zeros(history_motion.shape[0], text_emb.shape[1],
                                      device=history_motion.device, dtype=history_motion.dtype)
        track_features = target_enc(
            history_motion, zeros_text_hist, mask=None, return_layer_outputs=False
        ).detach()  # (B, T_history, H_enc)

        return z1, track_features

    def compute_flow_loss(
        self,
        target_motion: torch.Tensor | None = None,
        history_motion: torch.Tensor | None = None,
        text_emb: torch.Tensor | None = None,
        is_training: bool = True,
        z1: torch.Tensor | None = None,
        track_features: torch.Tensor | None = None,
        history_valid_length: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rectified flow matching loss.

        Suitable for use from external (non-Ignite) combined training loops.

        Args:
            target_motion:  Normalised target window (B, T_target, 271).
            history_motion: Normalised history window (B, T_history, 271), non-overlapping
                            with target_motion.
            text_emb:       Text embedding (B, D) or sequence (B, S, D).
            is_training:    If False, skip CFG dropout; use EMA predictor.
            z1:             Pre-computed target latents (B, T_target-1, H_enc).
                            Computed from frozen encoder if not supplied.
            track_features: Pre-computed history context (B, T_history, H_enc).
                            Computed from frozen encoder if not supplied.
            history_valid_length: Optional (B,) tensor with valid history counts.

        Returns:
            loss:  Scalar flow matching MSE loss.
            t_vec: Timestep vector (B,) for diagnostics/logging.
        """
        # Ensure text_emb has a sequence dimension (B, S, D)
        if text_emb is not None and text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(1)

        if z1 is None or track_features is None:
            if target_motion is None or history_motion is None or text_emb is None:
                raise ValueError("target_motion, history_motion, and text_emb must be provided if z1 or track_features is None.")
            # Encoder expects a 2D text embedding for shapes (B, 512).
            # We use the mean pooled text embedding to avoid breaking encoder's check.
            encoder_text_ref = text_emb.mean(dim=1)
            z1, track_features = self.compute_target_latents(target_motion, history_motion, encoder_text_ref, history_valid_length=history_valid_length)

        batch_size = z1.shape[0]

        # Apply target latent normalization channel-wise
        z1_normalized = self.normalize_latent(z1)

        z0 = torch.randn_like(z1_normalized)
        t = self._sample_timesteps(batch_size)
        t_bc = t.view(batch_size, 1, 1)

        z_t = (1.0 - t_bc) * z0 + t_bc * z1_normalized
        v_target = z1_normalized - z0

        text_with_cfg = self._apply_cfg_dropout(text_emb, batch_size, is_training=is_training)

        # Concatenate text token sequence and history context along the sequence dimension: (B, S + M, 512)
        combined_cond = torch.cat([text_with_cfg, track_features], dim=1)

        # Build boolean key-padding mask for combined_cond = [text_with_cfg, track_features]
        S_text = text_with_cfg.shape[1]
        T_hist = track_features.shape[1]
        S_cond = S_text + T_hist

        if history_valid_length is not None:
            device = track_features.device
            hist_idx = torch.arange(T_hist, device=device).unsqueeze(0)  # (1, T_hist)
            valid_start = (T_hist - history_valid_length.to(device=device)).unsqueeze(1)  # (B, 1)
            hist_mask = hist_idx >= valid_start  # (B, T_hist) bool
            text_mask = torch.ones((batch_size, S_text), dtype=torch.bool, device=device)  # (B, S_text) bool
            cond_mask = torch.cat([text_mask, hist_mask], dim=1)  # (B, S_cond) bool
            attn_mask = cond_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S_cond) bool
        else:
            attn_mask = None

        # Mean pool the text sequence to get a global vector for the predictor's global AdaLN path: (B, 512)
        text_pooled = text_with_cfg.mean(dim=1)

        active_predictor = self.predictor if is_training else self.ema_predictor.model
        v_pred, _, _ = active_predictor(
            noisy_states=z_t,
            timesteps=t,
            track_features=combined_cond,       # Pass combined condition as cross-attention target
            text_embedding=text_pooled,         # Pass pooled vector for global AdaLN text projection
            output_attentions=False,
            key_padding_mask=attn_mask,
            history_states=track_features,      # Pass pure history frames for self-attention masked_cond
        )

        # Stage 3: Inverse Latent Channel Standard Deviation Loss Reweighting (using normalized-space std)
        if hasattr(self, "latent_stats") and self.latent_stats is not None and getattr(self.latent_stats, "initialized", False):
            if hasattr(self.latent_stats, "norm_std"):
                ch_std = self.latent_stats.norm_std.view(1, 1, -1).detach() + 1e-4
            else:
                ch_std = z1_normalized.detach().flatten(0, 1).std(dim=0).view(1, 1, -1) + 1e-4
            ch_weight = 1.0 / ch_std
            ch_weight = ch_weight / ch_weight.mean()  # Normalize mean weight to 1.0
            loss = ((v_pred - v_target) ** 2 * ch_weight).mean()
        else:
            loss = F.mse_loss(v_pred, v_target, reduction="mean")
        return loss, t


    # ── Ignite step functions ──────────────────────────────────────────────────

    def _train_step(
        self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one Ignite training step."""
        self.predictor.train()
        self.null_text_embedding.train()

        target_motion, history_motion, text_emb = self._prepare_batch_phase3(batch)
        hist_valid_len = batch.get("history_valid_length", None)

        with torch.amp.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            loss, _t = self.compute_flow_loss(target_motion, history_motion, text_emb, is_training=True, history_valid_length=hist_valid_len)

        self.scaler.scale(loss / self.accumulation_steps).backward()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.predictor.parameters()) + list(self.null_text_embedding.parameters()),
                max_norm=float(self.config.gradient_clip),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.ema_predictor.update(self.predictor)
            self.lr_scheduler.step()

            engine.state.metrics["global_step"] = engine.state.iteration // self.accumulation_steps

        engine.csa_op_metrics(
            [("loss", loss.detach().item())],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,
        )

        return {
            "loss": loss.detach(),
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
        }

    def _val_step(
        self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one Ignite validation step using EMA predictor."""
        self.ema_predictor.model.eval()

        target_motion, history_motion, text_emb = self._prepare_batch_phase3(batch)
        hist_valid_len = batch.get("history_valid_length", None)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss, _t = self.compute_flow_loss(target_motion, history_motion, text_emb, is_training=False, history_valid_length=hist_valid_len)

        return {
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
            "val_loss": loss,
        }

    def _prepare_batch_phase3(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and normalize target_motion, history_motion, and text_emb from a batch.

        Returns:
            target_motion:  Normalised (B, T_target, 271)
            history_motion: Normalised (B, T_history, 271)
            text_emb:       (B, S, D) sequence embeddings (dynamically computed from captions)
                            or (B, D) pooled fallback.
        """
        motion_raw  = batch["motion"].to(self.device)          # target window, raw
        history_raw = batch["history_motion"].to(self.device)  # history window, raw

        target_motion  = self.normalizer.normalize(motion_raw)
        history_motion = self.normalizer.normalize(history_raw)

        if "captions" in batch:
            text_emb = self.clip_encoder.encode_sequence(batch["captions"])
        else:
            text_raw = batch["text_clip"].to(self.device)
            text_emb = text_raw[:, -1, :] if text_raw.ndim == 3 else text_raw  # (B, D)

        return target_motion, history_motion, text_emb

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            timer = engine.state.timer
            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()
            step_time_avg = (
                engine.state.metrics["step_time_avg"]
                if "step_time_avg" in engine.state.metrics
                else step_time
            )
            remaining_time = estimate_time_remaining(engine, step_time_avg, self.config)
            engine.set_metrics([("step_time", step_time), ("remaining_time", remaining_time)])
            metrics = engine.get_metrics(["loss", "lr"], prefix="train/")
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))
            self.wandb_logger.log(
                {
                    "train/horizon": engine.state.horizon,
                    "epoch": int(engine.state.epoch),
                    "global_step": int(engine.get_metric("global_step", 0)),
                    "step_time": step_time,
                    "remaining_time": remaining_time,
                },
                step=engine.get_metric("global_step", 0),
            )

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            engine.scale_metrics(
                ["val_loss"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )
            self.wandb_logger.log(
                engine.get_metrics(["val_loss"], prefix="val/"),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {
                    "epoch": int(trainer.state.epoch),
                    "global_step": int(trainer.get_metric("global_step", 0)),
                },
                step=int(trainer.get_metric("global_step", 0)),
            )
            val_loss = float(engine.get_metric("val_loss", float("inf")))
            best_val_loss = float(engine.get_metric("best_val_loss", float("inf")))
            if val_loss < best_val_loss:
                engine.set_metrics([("best_val_loss", val_loss)])
                self.wandb_logger.log(
                    {"val/best_val_loss": val_loss},
                    step=int(trainer.get_metric("global_step", 0)),
                )

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            output = engine.state.output if isinstance(engine.state.output, dict) else {}
            engine.csa_op_metrics(
                [("val_loss", float(output.get("val_loss", 0.0)))],
                1.0,
            )

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        super()._attach_handlers(trainer, evaluator)

        checkpoint_mapping = {
            "trainer": trainer,
            "encoder": self.encoder,
            "encoder_ema": self.ema_encoder,
            "predictor": self.predictor,
            "predictor_ema": self.ema_predictor,
            "null_text_embedding": self.null_text_embedding,
            "latent_stats": self.latent_stats,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "metadata": CheckpointMetadata(
                {
                    "pretrain_id": self.pretraining_session_id,
                    "session_id": self.session_id,
                    "resume_id": self.resume_id,
                }
            ),
            "config": CheckpointMetadata(asdict(self.config)),
        }

        def _global_step_transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"phase3_best_val_{self.phase3_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=_global_step_transform,
        )
        self.val_best_checkpoint = val_best_checkpoint

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"phase3_latest_{self.phase3_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=_global_step_transform,
        )
        self.latest_checkpoint = latest_checkpoint

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval), latest_checkpoint
        )
        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)

    # ── Public run ─────────────────────────────────────────────────────────────

    def run(self, max_epochs: int | None = None) -> None:
        """Run standalone flow matching training (uses Ignite internally)."""
        print(f"Starting Phase 3 (flow matching predictor) session: {self.session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self._attach_handlers(trainer, self.evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


# ── Convenience function ───────────────────────────────────────────────────────


def train_flow_matching(
    config: Config,
    pretrained_checkpoint_path: str | Path,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel[MotionHistoryEncoder], EMAModel[FlowMatchingPredictor], Path]:
    """Train the flow matching predictor while reusing a pretrained encoder checkpoint.

    Returns:
        ema_encoder:   Frozen EMA encoder (for use in inference or further training).
        ema_predictor: Trained EMA predictor.
        checkpoint_path: Path to the best-val checkpoint saved to disk.
    """
    trainer = FlowMatchingTrainer(
        config=config,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = trainer.val_best_checkpoint.last_checkpoint
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path()
    return trainer.ema_encoder, trainer.ema_predictor, checkpoint_path


__all__ = ["FlowMatchingTrainer", "train_flow_matching"]
