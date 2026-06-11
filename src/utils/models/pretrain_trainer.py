"""
JEPA-style Pretraining Trainer for Motion History Encoder.

Standalone, self-contained trainer using PyTorch Ignite.
Handles all model building, training loop configuration, and execution.
Designed for direct use in Kaggle notebooks with minimal boilerplate.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Dict, Generic, Tuple, TypeVar, Union
from typing import Mapping as MappingABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events, State
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    TerminateOnNan,
    global_step_from_engine,
)
from ignite.handlers.tqdm_logger import ProgressBar
from torch.amp.grad_scaler import GradScaler

from utils.config import Config
from utils.dataset import create_dataloader
from utils.models.flow_matching_predictor import LatentDecoder
from utils.models.motion_history_encoder import JepaPredictor, LinearProbe, MotionHistoryEncoder
from utils.motion_utils import F as Features
from utils.motion_utils import x68_to_x271, x271_to_x68
from utils.wandb_logger import WandbLogger


class CheckpointMetadata:
    """Wrapper for non-stateful metadata to be saved with checkpoints.

    Ignite's Checkpoint handler requires all values in the to_save dict to have
    ``state_dict`` / ``load_state_dict`` methods. This wrapper lets us store
    simple metadata (like session_id) alongside model checkpoints without
    triggering infinite recursion in ignite's _tree_map (which would happen
    with bare strings since they are Sequences of single-char strings).
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def state_dict(self) -> dict[str, Any]:
        return self.data

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data = state_dict


class InterpEnum(Enum):
    """Interpolation types for ProgressScheduler."""

    NONE = "none"
    LINEAR = "linear"
    CUBIC = "cubic"


class ProgressScheduler:
    """Utility for scheduling progress through curriculum phases."""

    def __init__(self, schedule: list[tuple[float, float]]):
        progress, value = zip(*schedule)
        max_progress = max(progress)
        self.progress = [p / max_progress for p in progress]
        self.value = value

    def get_value(self, progress: float, interp: InterpEnum = InterpEnum.NONE) -> float:
        """Return progress through current curriculum phase as a float in [0, 1]."""
        if progress <= self.progress[0]:
            return self.value[0]

        for i in range(1, len(self.progress)):
            if progress <= self.progress[i]:
                if interp == InterpEnum.NONE:
                    return self.value[i - 1]
                elif interp == InterpEnum.LINEAR:
                    ratio = (progress - self.progress[i - 1]) / (self.progress[i] - self.progress[i - 1])
                    return self.value[i - 1] + ratio * (self.value[i] - self.value[i - 1])
                elif interp == InterpEnum.CUBIC:
                    ratio = (progress - self.progress[i - 1]) / (self.progress[i] - self.progress[i - 1])
                    ratio_cubic = 3 * ratio**2 - 2 * ratio**3  # Smooth cubic interpolation
                    return self.value[i - 1] + ratio_cubic * (self.value[i] - self.value[i - 1])
                else:
                    raise ValueError(f"Unsupported interpolation type: {interp}")

        return self.value[-1]


class PretrainState(State):
    """Custom Ignite State for JEPA pretraining."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = 40
        self.metrics: Dict[str, Any] = {
            "global_step": 0,
            "best_train_loss": float("inf"),
        }

    def epoch_progress(self, config: Config) -> float:
        """Return progress through current epoch as a float in [0, 1]."""
        return self.epoch / float(config.get_num_epochs()) if config.get_num_epochs() > 0 else 0.0


class PretrainEngine(Engine):
    """Custom Ignite Engine for JEPA pretraining."""

    def __init__(self, process_function: Any) -> None:
        super().__init__(process_function)
        self.state = PretrainState()

    def get_metric(self, name: str, *args, **kwargs):
        """Get a metric value by name."""
        return self.state.metrics.get(name, *args, **kwargs)

    def get_metrics(self, names: list[str], prefix: str = "") -> dict[str, Any]:
        """Get specified metrics as a dict."""
        return {f"{prefix}{name}": self.state.metrics.get(name) for name in names}

    def set_metrics(self, pairs: list[tuple[str, Any]]) -> None:
        """Set multiple metrics at once."""
        for name, value in pairs:
            self.state.metrics[name] = value

    def clear_metrics(self, names: list[str]) -> None:
        """Clear specified metrics."""
        for name in names:
            self.state.metrics.pop(name, None)

    def csa_op_metrics(self, pairs: list[tuple[str, float]], scalers: float | list[float], clear: bool = False) -> None:
        """Add a value to an existing metric (useful for running totals)."""
        if clear:
            self.clear_metrics([name for name, _ in pairs])
        if not isinstance(scalers, list):
            scalers = [scalers for _ in pairs]
        for (name, value), scaler in zip(pairs, scalers):
            self.state.metrics[name] = self.state.metrics.get(name, 0.0) + value * scaler


def random_span_mask(
    seq_len: int,
    num_spans: int = 2,
    min_span: int = 8,
    max_span: int = 20,
):
    mask = torch.zeros(seq_len, dtype=torch.bool)

    for _ in range(num_spans):
        span = torch.randint(min_span, max_span + 1, ()).item()
        start = torch.randint(0, int(max(1, seq_len - span + 1)), ()).item()
        mask[start : start + span] = True

    return mask


T = TypeVar("T", bound=nn.Module)


class EMAModel(Generic[T]):
    """
    Exponential Moving Average model wrapper.

    Maintains an EMA copy of a model for more stable evaluation.
    EMA is used for validation, sampling, and checkpointing.
    """

    def __init__(self, model: T, decay: float = 0.999):
        """
        Initialize EMA model.

        Args:
            model: The model to create EMA copy of
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.model: T = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def update(self, model: T) -> None:
        """
        Update EMA weights.

        Args:
            model: The source model to update from
        """
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device: Union[str, torch.device]) -> "EMAModel[T]":
        """Move EMA model to device."""
        self.model.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return state dict of wrapped model for checkpointing."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: MappingABC) -> None:
        """Load state dict into wrapped model."""
        self.model.load_state_dict(state_dict)


class PretrainTrainer:
    """
    JEPA-style pretraining trainer with masked frame reconstruction.

    Fully self-contained: builds models, optimizer, and Ignite engine.
    Usage:
        trainer = PretrainTrainer(config=config, train_loader=train_loader, val_loader=val_loader)
        trainer.run()
    """

    def __init__(
        self,
        config: Config,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.wandb_project = wandb_project

        # Device and AMP setup
        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

        self.scaler: GradScaler = GradScaler(self.device.type, enabled=self.use_amp)

        # Models - built internally
        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config.encoder_config)
        self.jepa_predictor: JepaPredictor = JepaPredictor(config.encoder_config)
        self.decoder: LatentDecoder = LatentDecoder(config)

        self.linear_probe: LinearProbe = LinearProbe(
            hidden_size=config.encoder_config.hidden_size,
            text_embedding_dim=config.text_embedding_dim,
        ).to(self.device)

        self._log_model_parameters(self.encoder, self.jepa_predictor, self.linear_probe, self.decoder)

        # W&B logger
        self.wandb_logger: WandbLogger | None = None

        # Initialize all objects
        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components: models, optimizer, W&B."""
        # Create checkpoint directory
        config = self.config

        utc_plus_6 = timezone(timedelta(hours=6))
        self.session_id = datetime.now(utc_plus_6).strftime("%Y%m%d_%H%M%S")
        self.pretraining_session_id = self.session_id

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader, self.normalizer = create_dataloader(config, "train", shuffle=True)
        self.val_loader, _ = create_dataloader(config, "val", shuffle=False)

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.jepa_predictor.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self.aux_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.linear_probe.parameters()) + list(self.decoder.parameters()),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )

        # EMA models
        self.ema_encoder: EMAModel = EMAModel(self.encoder, decay=float(config.ema_decay))

        # Move to device
        self.encoder.to(self.device)
        self.jepa_predictor.to(self.device)
        self.linear_probe.to(self.device)
        self.decoder.to(self.device)
        self.ema_encoder.to(self.device)

        self.accumulation_steps = ceil(config.effective_batch_size / config.batch_size) or 1

        num_epochs = int(config.get_num_epochs())
        warmup_epochs = int(config.lr_warmup_epochs)
        steps_per_epoch = max(len(self.train_loader) // self.accumulation_steps, 1)
        total_steps = num_epochs * steps_per_epoch
        pct_start = min(max(warmup_epochs / num_epochs, 0.0), 1.0) if num_epochs > 0 else 0.0

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=float(config.learning_rate),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )
        self.aux_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.aux_optimizer,
            max_lr=float(config.learning_rate),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        self.resume_id = None

        # Setup W&B
        if self.wandb_project:
            self.wandb_logger = WandbLogger(
                project=self.wandb_project,
                name=self.pretraining_session_id,
                config={
                    "lr": float(self.config.learning_rate),
                    "weight_decay": float(self.config.weight_decay),
                    "ema_decay": float(self.config.ema_decay),
                    "batch_size": self.train_loader.batch_size,
                    "effective_batch_size": self.config.effective_batch_size,
                    "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                    "jepa_params": sum(p.numel() for p in self.jepa_predictor.parameters()),
                    "phase": "pretrain",
                    "session_id": self.session_id,
                },
                resume_id=self.resume_id if self.resume_id else None,
            )

            self.resume_id = self.wandb_logger.run.id if self.wandb_logger.run else None
            print(f"Initialized W&B run with ID: {self.resume_id}")

    @staticmethod
    def _log_model_parameters(
        encoder: MotionHistoryEncoder, jepa_predictor: JepaPredictor, linear_probe: LinearProbe, decoder: LatentDecoder
    ) -> dict[str, int]:
        """Print total and category-wise parameter counts for all models."""

        def _print_model_summary(name: str, model: nn.Module) -> int:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            print(f"\n{'=' * 60}")
            print(f"Model: {name}")
            print(f"{'=' * 60}")
            print(f"Total parameters     : {total:,}")
            print(f"Trainable parameters : {trainable:,}")
            print(f"Frozen parameters    : {frozen:,}")
            print("-" * 60)
            print(f"{'Category':<30} {'Params':>12} {'%':>7}")
            print("-" * 60)
            categories = _categorize_parameters(model)
            for cat, count in categories.items():
                pct = count / total * 100 if total > 0 else 0.0
                print(f"{cat:<30} {count:>12,} {pct:>6.2f}%")
            print("-" * 60)

            return trainable

        def _categorize_parameters(model: nn.Module) -> Dict[str, int]:
            categories: Dict[str, int] = {}
            for name, param in model.named_parameters():
                cat = _assign_category(name)
                categories[cat] = categories.get(cat, 0) + param.numel()
            return categories

        def _assign_category(name: str) -> str:
            name_lower = name.lower()
            if "cross_attn" in name_lower:
                return "cross_attn"
            if "self_attn" in name_lower or "attn" in name_lower:
                return "self_attn"
            if "mlp" in name_lower:
                return "mlp"
            if "adaln" in name_lower:
                return "adaln"
            if "layer_norm" in name_lower or "norm" in name_lower:
                return "layer_norm"
            if "register" in name_lower or "mask_token" in name_lower:
                return "learnable_tokens"
            if "linear" in name_lower or "proj" in name_lower:
                return "linear_proj"
            return "other"

        p_enc = _print_model_summary("MotionHistoryEncoder", encoder)
        p_jepa = _print_model_summary("JepaPredictor", jepa_predictor)
        p_linear = _print_model_summary("LinearProbe", linear_probe)
        p_decoder = _print_model_summary("LatentDecoder", decoder)

        return {
            "encoder": p_enc,
            "jepa_predictor": p_jepa,
            "linear_probe": p_linear,
            "decoder": p_decoder,
        }

    def _train_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""

        # Forward pass
        self.encoder.train()
        self.jepa_predictor.train()
        self.linear_probe.train()
        self.decoder.train()

        # Prepare batch
        motion = batch["motion"].to(self.device)
        motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)
        joints = batch["joints"].to(self.device)

        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError(f"Expected motion (B, T, 271), got {tuple(motion.shape)}")

        losses = self._compute_loss(engine, motion, joints, text)

        loss = losses["loss"]
        probe_loss = losses["probe_loss"]
        decoder_loss = losses["decoder_loss"]

        self.scaler.scale(loss / self.accumulation_steps).backward()
        self.scaler.scale(probe_loss / self.accumulation_steps).backward()
        self.scaler.scale(decoder_loss / self.accumulation_steps).backward()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.step(self.aux_optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.aux_optimizer.zero_grad(set_to_none=True)

            self.ema_encoder.update(self.encoder)
            self.lr_scheduler.step()
            self.aux_lr_scheduler.step()

            engine.state.metrics["global_step"] = engine.state.iteration // self.accumulation_steps

        return {
            "loss": loss.detach(),
            "lr": losses["lr"],
            "probe_loss": probe_loss.detach(),
            "decoder_loss": decoder_loss.detach(),
        }

    def _val_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one validation step."""
        if self.config.val_use_ema:
            self.ema_encoder.model.eval()
        else:
            self.encoder.eval()
        self.jepa_predictor.eval()
        self.linear_probe.eval()
        self.decoder.eval()

        motion = batch["motion"].to(self.device)
        motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)
        joints = batch["joints"].to(self.device)

        with torch.no_grad():
            losses = self._compute_loss(engine, motion, joints, text)

        return {
            "lr": losses["lr"],
            "val_loss": losses["loss"],
            "val_probe_loss": losses["probe_loss"],
            "val_decoder_loss": losses["decoder_loss"],
        }

    def _compute_loss(
        self, engine: PretrainEngine, motion: torch.Tensor, joints: torch.Tensor, text: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = motion.shape

        # Build mask
        mask_bool = (
            random_span_mask(seq_len, num_spans=2, min_span=2, max_span=4).unsqueeze(0).expand(batch_size, -1)
        ).to(self.device)  # (B, seq_len)

        with torch.amp.autocast(  # type: ignore
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            text_emb = text[:, -1, :] if text.ndim == 3 else text
            masked_context = self.encoder.forward(
                motion, torch.zeros_like(text_emb), mask=mask_bool, return_layer_outputs=True
            )  # (B, seq_len, L, H) where L = num_hidden_layers

            with torch.no_grad():
                target_encoder = self.ema_encoder.model
                target_encoder.eval()
                target_context = target_encoder(
                    motion, torch.zeros_like(text_emb), mask=None, return_layer_outputs=True
                ).detach()  # (B, seq_len, L, H)

            _, _, L, H = masked_context.shape

            predicted = self.jepa_predictor(masked_context)  # (B, seq_len, L, H)

            #
            # Compute Loss
            #
            _loss = F.smooth_l1_loss(predicted, target_context, reduction="none").mean(dim=(2, 3))  # (B, seq_len)
            # Extract masked tokens
            mask_loss = (_loss * mask_bool.float()).sum() / mask_bool.sum().clamp(min=1)
            context_loss = self._context_loss(engine, _loss, mask_bool)
            loss = mask_loss + context_loss * self.config.jepa_ctx_weight

            #
            # Linear probe loss (cosine similarity between predicted context and text embedding)
            #
            probe_out = self.linear_probe(target_context[:, :, -1, :])  # Use last layer's output for probing
            probe_out = F.normalize(probe_out, dim=-1)
            normalized_text = F.normalize(text_emb, dim=-1)
            probe_loss = 1 - F.cosine_similarity(probe_out, normalized_text, dim=-1).mean()

            #
            # Decoder loss
            #
            latent = target_context[:, 1:, -1, :]  # (B, seq_len-1, H_enc)
            decoded = self.decoder(latent)  # (B, seq_len-1, 68)

            decoder_losses = self._decoder_loss(engine, motion, joints, decoded)
            decoder_loss = decoder_losses["decoder_loss"]

        engine.csa_op_metrics(
            [
                ("loss", loss.detach().item()),
                ("mask_loss", mask_loss.detach().item()),
                ("context_loss", context_loss.detach().item()),
                ("probe_loss", probe_loss.detach().item()),
            ],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,  # Clear on first step of accumulation
        )

        return {
            "loss": loss,
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
            "probe_loss": probe_loss,
            "decoder_loss": decoder_loss,
        }

    def _context_loss(self, engine: PretrainEngine, _loss: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask_bool.shape
        all_mask_indices = [
            mask_bool[b].nonzero(as_tuple=False).squeeze(1)  # (num_masked_b,)
            for b in range(batch_size)
        ]

        pos = torch.arange(seq_len, device=self.device)  # (seq_len,)
        weights = torch.zeros(batch_size, seq_len, device=self.device)

        # Compute distance of each position to nearest masked position
        pos_exp = pos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1)
        mask_idx_exp = torch.stack(all_mask_indices).unsqueeze(1)  # (B, 1, num_masked)
        distances = (pos_exp - mask_idx_exp).abs()  # (B, seq_len, num_masked)
        min_distances = distances.min(dim=2).values  # (B, seq_len)
        weights = 1.0 / torch.sqrt(min_distances + 1e-8)  # Higher weight = closer to masked
        weights[mask_bool] = 0.0  # Masked positions don't contribute to context loss

        unmasked_bool = ~mask_bool  # (B, seq_len)
        context_loss = (_loss * unmasked_bool.float() * weights.detach()).sum() / (unmasked_bool.sum().float() + 1e-8)

        return context_loss

    def _decoder_loss(self, engine: PretrainEngine, motion, joints, decoded):
        prev_positions = joints[:, :-1, :].flatten(0, 1)  # (B * (seq_len-1), 22, 3)
        prev_frames = motion[:, :-1, :].flatten(0, 1)  # (B * (seq_len-1), 271)
        y_frames = motion[:, 1:, :].flatten(0, 1)  # (B * (seq_len-1), 271)
        decoded = decoded.flatten(0, 1)  # (B * (seq_len-1), 68)

        y_68d = x271_to_x68(y_frames, self.normalizer, prev_frames)  # (B * (seq_len-1), 68)
        y_vel = y_frames[..., Features.VEL]  # (B * (seq_len-1), 66)
        y_feet = y_frames[..., Features.CONTACTS]  # (B * (seq_len-1), 4)

        dec_271d = x68_to_x271(decoded, self.normalizer, prev_positions, prev_frames)  # (B * (seq_len-1), 271)
        dec_vel = dec_271d[..., Features.VEL]  # (B * (seq_len-1), 66)
        dec_feet = dec_271d[..., Features.CONTACTS]  # (B * (seq_len-1), 4)
        dec_foot_vel = dec_271d[..., Features.joint_mask(["feet"], Features.VEL)]

        loss_68d = F.mse_loss(decoded, y_68d, reduction="mean")
        loss_vel = F.smooth_l1_loss(dec_vel, y_vel, reduction="mean")

        mask_feet_4d = (
            y_feet.bool() & ~dec_feet.bool()
        )  # (B * (seq_len-1), 4) - 4d foot contact false negatives - ground truth contact - prediction does not
        mask_feet_miss = mask_feet_4d.any(dim=-1)  # (B * (seq_len-1),) - boolean mask for any foot contact miss
        loss_feet = (
            dec_foot_vel[mask_feet_miss].square().mean()
            if mask_feet_miss.any()
            else torch.tensor(0.0, device=self.device)
        )

        decoder_loss = loss_68d + loss_vel * 0.5 + loss_feet * 0.5

        feet_miss_rate = (
            mask_feet_4d.sum().float() / y_feet.bool().sum().float()
            if y_feet.bool().sum() > 0
            else torch.tensor(0.0, device=self.device)
        )

        engine.csa_op_metrics(
            [
                ("decoder_loss", decoder_loss.detach().item()),
                ("loss_68d", loss_68d.detach().item()),
                ("loss_vel", loss_vel.detach().item()),
                ("loss_feet", loss_feet.detach().item()),
                ("feet_miss_rate", feet_miss_rate.detach().item()),
            ],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,  # Clear on first step of accumulation
        )

        return {
            "decoder_loss": decoder_loss,
            "loss_68d": loss_68d.detach(),
            "loss_vel": loss_vel.detach(),
            "loss_feet": loss_feet.detach(),
            "feet_miss_rate": feet_miss_rate.detach(),
        }

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach Ignite event handlers for training orchestration."""

        # NaN termination
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        # Progress bar for console display
        pbar = ProgressBar(
            mininterval=10.0,
        )
        pbar.attach(trainer, ["loss"])

        trainer.state.horizon = self.config.horizon

        # Step logging
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _log_train_step(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            output = engine.state.output
            if not isinstance(output, dict):
                return
            metrics = engine.get_metrics(
                [
                    "loss",
                    "mask_loss",
                    "context_loss",
                    "probe_loss",
                    "decoder_loss",
                    "decoder_loss_68d",
                    "decoder_loss_vel",
                    "decoder_loss_feet",
                    "feet_miss_rate",
                    "lr",
                ],
                prefix="train/",
            )
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))
            self.wandb_logger.log(
                {
                    "train/horizon": engine.state.horizon,
                    "epoch": int(engine.state.epoch),
                    "global_step": int(engine.get_metric("global_step", 0)),
                },
                step=engine.get_metric("global_step", 0),
            )

        @trainer.on(Events.GET_BATCH_STARTED)
        def _set_horizon(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(engine.state.horizon)  # type: ignore[attr-defined]

        @evaluator.on(Events.GET_BATCH_STARTED)
        def _set_horizon_eval(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(trainer.state.horizon)  # type: ignore[attr-defined]

        checkpoint_mapping = {
            "encoder": self.encoder,
            "jepa_predictor": self.jepa_predictor,
            "encoder_ema": self.ema_encoder,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "aux_lr_scheduler": self.aux_lr_scheduler,
            "metadata": CheckpointMetadata({"session_id": self.pretraining_session_id, "resume_id": self.resume_id}),
        }
        # Best checkpoint handler
        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="pretrain_best_val",
            score_function=lambda engine: -float(engine.state.metrics["loss"]),
            score_name="val_loss",
            global_step_transform=global_step_from_engine(trainer),
        )

        eval_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="pretrain_best_eval",
            score_function=lambda engine: -float(engine.state.metrics["decoder_loss"]),
            score_name="eval_loss",
            global_step_transform=global_step_from_engine(trainer),
        )

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="pretrain_latest",
            filename_pattern="{filename_prefix}.pt",
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval),
            latest_checkpoint,
        )

        @trainer.on(Events.EPOCH_COMPLETED(every=self.config.val_interval))
        def _run_validation(engine: PretrainEngine) -> None:
            evaluator.run(self.val_loader)

        val_batches = self.config.val_batches
        if val_batches > 0:

            @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
            def _limit_val_batches(engine: PretrainEngine) -> None:
                if engine.state.iteration // self.accumulation_steps >= val_batches:
                    engine.terminate()

        if self.config.save_best_val:

            @evaluator.on(Events.COMPLETED)
            def _log_best_validation(engine: PretrainEngine) -> None:
                val_loss = float(engine.get_metric("loss", float("inf")))
                best_val_loss = float(engine.state.metrics.get("best_val_loss", float("inf")))
                if val_loss < best_val_loss:
                    engine.set_metrics([("best_val_loss", val_loss)])

                eval_loss = float(engine.get_metric("decoder_loss", float("inf")))
                best_eval_loss = float(engine.state.metrics.get("best_eval_loss", float("inf")))
                if eval_loss < best_eval_loss:
                    engine.set_metrics([("best_eval_loss", eval_loss)])

                if self.wandb_logger:
                    self.wandb_logger.log(
                        engine.get_metrics(
                            [
                                "loss",
                                "mask_loss",
                                "context_loss",
                                "probe_loss",
                                "decoder_loss",
                                "decoder_loss_68d",
                                "decoder_loss_vel",
                                "decoder_loss_feet",
                                "feet_miss_rate",
                            ],
                            prefix="val/",
                        ),
                        step=int(trainer.get_metric("global_step", 0)),
                    )

                    self.wandb_logger.log(
                        {
                            "epoch": int(trainer.state.epoch),
                            "global_step": int(trainer.get_metric("global_step", 0)),
                        },
                        step=int(trainer.get_metric("global_step", 0)),
                    )

                    if val_loss < best_val_loss:
                        self.wandb_logger.log(
                            {"val/best_loss": val_loss}, step=int(trainer.get_metric("global_step", 0))
                        )

                    if eval_loss < best_eval_loss:
                        self.wandb_logger.log(
                            {"val/best_eval_loss": eval_loss}, step=int(trainer.get_metric("global_step", 0))
                        )

            evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)
            evaluator.add_event_handler(Events.COMPLETED, eval_best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the pretraining loop."""
        print(f"Starting pretraining session: {self.pretraining_session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch))
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch))
        self._attach_handlers(trainer, self.evaluator)

        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)

        if self.wandb_logger:
            self.wandb_logger.finish()


# Convenience function for notebook usage
def train_pretrain(
    config: Config,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel, None, Path]:
    """
    Train encoder with JEPA objective in a single call.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        normalizer: Feature normalizer
        wandb_project: Optional W&B project name
        max_epochs: Override number of epochs (uses config default if None)

    Returns:
        Tuple of (ema_encoder, ema_jepa_predictor, checkpoint_path)
    """
    trainer = PretrainTrainer(
        config=config,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = config.checkpoint_dir / "pretrain_latest.pt"
    return trainer.ema_encoder, None, checkpoint_path


__all__ = ["EMAModel", "PretrainTrainer", "train_pretrain"]
