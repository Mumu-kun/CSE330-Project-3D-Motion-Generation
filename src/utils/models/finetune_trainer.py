"""
Flow Matching Fine-tuning Trainer for Motion Generation.

Standalone, self-contained trainer using PyTorch Ignite.
Handles all model building, training loop configuration, and execution.
Designed for direct use in Kaggle notebooks with minimal boilerplate.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan, global_step_from_engine
from ignite.metrics import RunningAverage

from utils.config import Config
from utils.models import MotionHistoryEncoder, FlowMatchingPredictor
from utils.motion_utils import (
    FeatureNormalizer,
    extract_prev_frame_features,
    subset_271d_to_68d,
    wrap_angle,
    sin_cos_to_yaw,
)
from utils.wandb_logger import WandbLogger as _WandbLogger

T = torch.Tensor

# Constants for flow diagnostics
LOSS_VS_T_NUM_BINS = 100


def _compute_per_sample_flow_loss(pred: T, target_flow: T) -> T:
    """Compute one mean-MSE flow loss value per flattened training sample."""
    if pred.shape != target_flow.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target_flow.shape)}")
    if pred.ndim < 2:
        raise ValueError(f"Expected at least 2D, got {tuple(pred.shape)}")
    return F.mse_loss(pred, target_flow, reduction="none").mean(dim=-1)


def _aggregate_loss_vs_t_bins(
    t_values: T,
    per_sample_flow_loss: T,
    num_bins: int = LOSS_VS_T_NUM_BINS,
) -> Dict[str, T]:
    """Aggregate per-sample flow loss into fixed bins over t in [0, 1]."""
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if t_values.ndim != 1 or per_sample_flow_loss.ndim != 1:
        raise ValueError(f"Expected 1D tensors, got {tuple(t_values.shape)} and {tuple(per_sample_flow_loss.shape)}")

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, dtype=torch.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    counts = torch.zeros(num_bins, dtype=torch.long)
    mean_losses = torch.full((num_bins,), float("nan"), dtype=torch.float64)

    t_cpu = t_values.detach().to(dtype=torch.float64, device="cpu").clamp_(0.0, 1.0)
    loss_cpu = per_sample_flow_loss.detach().to(dtype=torch.float64, device="cpu")
    bin_indices = torch.clamp((t_cpu * num_bins).to(torch.long), max=num_bins - 1)

    counts = torch.bincount(bin_indices, minlength=num_bins)
    sums = torch.bincount(bin_indices, weights=loss_cpu, minlength=num_bins)
    nonempty = counts > 0
    mean_losses[nonempty] = sums[nonempty] / counts[nonempty].to(torch.float64)

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": counts,
        "mean_flow_loss": mean_losses,
    }


class EMAModel(nn.Module):
    """Exponential Moving Average model wrapper for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def update(self, model: nn.Module) -> None:
        """Update EMA weights from source model."""
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device: torch.device) -> "EMAModel":
        self.model.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)


class FiTrainer:
    """
    Flow matching fine-tuning trainer with curriculum support.
    
    Fully self-contained: builds models, optimizer, and Ignite engine.
    Usage:
        trainer = FiTrainer(config=config, train_loader=train_loader, val_loader=val_loader)
        trainer.run()
    """

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: FeatureNormalizer | None = None,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.wandb_project = wandb_project

        # Device and AMP setup
        self.device = torch.device(config.device)
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # Models - built internally
        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config.encoder_config)
        self.predictor: FlowMatchingPredictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
        )

        # EMA models
        self.ema_encoder: EMAModel = EMAModel(self.encoder, decay=float(config.ema_decay))
        self.ema_predictor: EMAModel = EMAModel(self.predictor, decay=float(config.ema_decay))

        # Optimizer and scaler
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.scaler: GradScaler = GradScaler("cuda", enabled=self.use_amp)

        # Ignite engines
        self.trainer: Engine | None = None
        self.evaluator: Engine | None = None

        # W&B logger
        self.wandb_logger: _WandbLogger | None = None

        # State tracking
        self.global_step = 0
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")

        # Initialize all objects
        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components: models, optimizer, W&B."""
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move to device
        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.ema_encoder.to(self.device)
        self.ema_predictor.to(self.device)

        # Setup W&B
        if self.wandb_project:
            self.wandb_logger = _WandbLogger(
                project=self.wandb_project,
                config={
                    "lr": float(self.config.learning_rate),
                    "weight_decay": float(self.config.weight_decay),
                    "ema_decay": float(self.config.ema_decay),
                    "batch_size": self.train_loader.batch_size,
                    "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                    "predictor_params": sum(p.numel() for p in self.predictor.parameters()),
                    "phase": "finetune",
                },
            )

    def _sample_timesteps(self, batch_size: int, epoch: int) -> T:
        """Sample training timesteps according to config (power or uniform)."""
        u = torch.rand(batch_size, device=self.device, dtype=torch.float32)

        if self.config.t_sampling_mode == "uniform":
            return u

        if self.config.t_sampling_power <= 0 or self.config.get_num_epochs <= 1:
            return u

        return u.pow(1.0 / (float(self.config.t_sampling_power) + 1.0))

    def _compute_losses(self, batch: Dict[str, T], epoch: int) -> Tuple[T, T, T]:
        """Compute flow and consistency losses."""
        motion = batch["motion"].to(self.device)
        text = batch["text_clip"].to(self.device)

        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)

        # History and target
        history = motion[:, :-1]
        target_motion = motion[:, -1]
        current_frame = history[:, -1]

        # Text embedding with CFG dropout
        if self.config.cfg_dropout > 0.0 and torch.rand(1, device=self.device).item() <= self.config.cfg_dropout:
            text_emb = torch.zeros(
                motion.shape[0],
                self.config.encoder_config.text_embedding_dim,
                device=self.device,
                dtype=text.dtype,
            )
        else:
            text_emb = text[:, 0, :] if text.ndim == 3 else text

        # Forward pass
        track_features = self.encoder(history, text_emb, return_all=False).unsqueeze(1)
        current_frame_features = extract_prev_frame_features(
            current_frame,
            normalizer=self.normalizer,
            normalize_output=self.normalizer is not None,
        )

        x1 = subset_271d_to_68d(target_motion, prev_frame=current_frame, normalizer=self.normalizer)
        x0 = torch.randn_like(x1)
        t = self._sample_timesteps(target_motion.shape[0], epoch)
        xt = t.unsqueeze(1) * x1 + (1 - t.unsqueeze(1)) * x0

        predicted_flow, _, _ = self.predictor(
            track_features=track_features,
            noisy_features=xt,
            timesteps=t,
            text_embedding=text_emb,
            current_frame_features=current_frame_features,
            output_attentions=False,
            output_hidden_states=False,
        )

        flow_loss = F.mse_loss(predicted_flow, x1 - x0)
        per_sample_loss = _compute_per_sample_flow_loss(predicted_flow, x1 - x0)

        # Consistency loss
        consistency_loss = predicted_flow.new_zeros(())
        if self.config.use_consistency_loss:
            t_thresh_mask = t > self.config.consistency_loss_t_threshold
            if t_thresh_mask.any():
                pred_x1 = xt[t_thresh_mask] + predicted_flow[t_thresh_mask] * (1 - t.unsqueeze(1)[t_thresh_mask])

                flow_raw = self.normalizer.denormalize_flow_output(pred_x1) if self.normalizer else pred_x1
                x1_raw = self.normalizer.denormalize_flow_output(x1[t_thresh_mask]) if self.normalizer else x1[t_thresh_mask]

                # Root loss
                root_loss = F.mse_loss(flow_raw[:, :3], x1_raw[:, :3])

                # Yaw loss
                x1_dyaw = sin_cos_to_yaw(x1_raw[:, 3:5])
                pred_dyaw = sin_cos_to_yaw(flow_raw[:, 3:5])
                yaw_error = wrap_angle(pred_dyaw - x1_dyaw)
                yaw_loss = yaw_error.pow(2).mean()

                # RIC loss
                ric_loss = F.mse_loss(flow_raw[:, 5:], x1_raw[:, 5:])

                consistency_loss = ric_loss * 3 + root_loss * 20 + yaw_loss * 10

        total_loss = flow_loss + self.config.consistency_loss_weight * consistency_loss
        return total_loss, flow_loss, consistency_loss

    def _train_step(self, engine: Engine, batch: Dict[str, T]) -> Dict[str, T]:
        """Execute one training step."""
        self.encoder.train()
        self.predictor.train()

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            loss, flow_loss, consistency_loss = self._compute_losses(batch, epoch=int(engine.state.epoch))

        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            float(self.config.gradient_clip),
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA update
        self.ema_encoder.update(self.encoder)
        self.ema_predictor.update(self.predictor)

        # Update state
        self.global_step = int(engine.state.iteration)
        self.best_train_loss = min(self.best_train_loss, float(loss.detach().item()))

        return {
            "loss": loss.detach(),
            "flow_loss": torch.tensor(flow_loss.detach().item(), device=self.device),
            "consistency_loss": torch.tensor(consistency_loss.detach().item(), device=self.device),
            "lr": torch.tensor(float(self.config.learning_rate), device=self.device),
        }

    def _val_step(self, engine: Engine, batch: Dict[str, T]) -> Dict[str, T]:
        """Execute one validation step."""
        motion = batch["motion"].to(self.device)
        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)

        with torch.no_grad():
            self.ema_encoder.model.eval()
            self.ema_predictor.model.eval()

            history = motion[:, :-1]
            target_motion = motion[:, -1]
            current_frame = history[:, -1]
            text_emb = text[:, 0, :] if text.ndim == 3 else text

            track_features = self.ema_encoder.model(history, text_emb, return_all=False).unsqueeze(1)
            current_frame_features = extract_prev_frame_features(
                current_frame,
                normalizer=self.normalizer,
                normalize_output=self.normalizer is not None,
            )

            x1 = subset_271d_to_68d(target_motion, prev_frame=current_frame, normalizer=self.normalizer)
            t = torch.rand(target_motion.shape[0], device=self.device, dtype=torch.float32)
            x0 = torch.randn_like(x1)
            xt = t.unsqueeze(1) * x1 + (1 - t.unsqueeze(1)) * x0

            predicted_flow, _, _ = self.ema_predictor.model(
                track_features=track_features,
                noisy_features=xt,
                timesteps=t,
                text_embedding=text_emb,
                current_frame_features=current_frame_features,
                output_attentions=False,
                output_hidden_states=False,
            )

            loss = F.mse_loss(predicted_flow, x1 - x0)

        return {"val_loss": loss.detach()}

    def _attach_handlers(self, trainer: Engine, evaluator: Engine) -> None:
        """Attach Ignite event handlers for training orchestration."""
        # Running averages
        RunningAverage(output_transform=lambda o: o["loss"]).attach(trainer, "loss")
        RunningAverage(output_transform=lambda o: o["val_loss"]).attach(evaluator, "val_loss")

        # NaN termination
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        # Step logging
        @trainer.on(Events.ITERATION_COMPLETED(every=50))
        def _log_train_step(engine: Engine) -> None:
            if not self.wandb_logger:
                return
            output = engine.state.output
            if not isinstance(output, dict):
                return
            metrics = {
                "train/loss": float(output.get("loss", 0.0)),
                "train/loss_avg": float(engine.state.metrics.get("loss", 0.0)),
                "train/lr": float(output.get("lr", 0.0)),
                "train/loss_flow": float(output.get("flow_loss", 0.0)),
                "train/loss_consistency": float(output.get("consistency_loss", 0.0)),
            }
            self.wandb_logger.log(metrics, step=int(engine.state.iteration))

        # Latest checkpoint
        latest_checkpoint = Checkpoint(
            {
                "encoder": self.encoder,
                "predictor": self.predictor,
                "encoder_ema": self.ema_encoder,
                "predictor_ema": self.ema_predictor,
                "optimizer": self.optimizer,
                "scaler": self.scaler,
            },
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="finetune_latest",
            global_step_transform=global_step_from_engine(trainer),
        )

        # Best checkpoint
        best_checkpoint = Checkpoint(
            {
                "encoder": self.encoder,
                "predictor": self.predictor,
                "encoder_ema": self.ema_encoder,
                "predictor_ema": self.ema_predictor,
                "optimizer": self.optimizer,
                "scaler": self.scaler,
            },
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="finetune_best",
            score_function=lambda e: -float(e.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=global_step_from_engine(trainer),
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, latest_checkpoint)

        @evaluator.on(Events.COMPLETED)
        def _log_val(engine: Engine) -> None:
            val_loss = float(engine.state.metrics.get("val_loss", float("nan")))
            if self.wandb_logger:
                self.wandb_logger.log({"val/loss": val_loss}, step=int(self.global_step))

        evaluator.add_event_handler(Events.COMPLETED, best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the fine-tuning loop."""
        if self.trainer is None:
            self.trainer = Engine(self._train_step)
            self.evaluator = Engine(self._val_step)
            self._attach_handlers(self.trainer, self.evaluator)

        epochs = max_epochs or int(self.config.get_num_epochs)
        self.trainer.run(self.train_loader, max_epochs=epochs)

        if self.wandb_logger:
            self.wandb_logger.finish()


# Convenience function for notebook usage
def train_finetune(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: FeatureNormalizer | None = None,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel, EMAModel]:
    """
    Fine-tune with flow matching objective in a single call.
    
    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        normalizer: Optional feature normalizer
        wandb_project: Optional W&B project name
        max_epochs: Override number of epochs (uses config default if None)
    
    Returns:
        Tuple of (ema_encoder, ema_predictor)
    """
    trainer = FiTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    return trainer.ema_encoder, trainer.ema_predictor


__all__ = ["EMAModel", "FiTrainer", "train_finetune", "_compute_per_sample_flow_loss", "_aggregate_loss_vs_t_bins"]