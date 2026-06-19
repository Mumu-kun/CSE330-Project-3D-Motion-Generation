"""
JEPA-style Pretraining Trainer for Motion History Encoder.

Standalone, self-contained trainer using PyTorch Ignite.
Handles all model building, training loop configuration, and execution.
Designed for direct use in Kaggle notebooks with minimal boilerplate.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    TerminateOnNan,
    Timer,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import RunningAverage
from torch.amp.grad_scaler import GradScaler

from utils.config import Config
from utils.dataset import create_dataloader
from utils.models import CheckpointMetadata, EMAModel, PretrainEngine, estimate_time_remaining
from utils.models.flow_matching_predictor import LatentDecoder
from utils.models.motion_history_encoder import JepaPredictor, LinearProbe, MotionHistoryEncoder
from utils.motion_utils import Features, x68_to_x271, x271_to_x68
from utils.wandb_logger import WandbLogger


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

        return self.value[-1]


def random_span_mask(
    seq_len: int,
    num_spans: int = 2,
    min_span: int = 8,
    max_span: int = 20,
    engine: PretrainEngine | None = None,
    device: torch.device | str | None = None,
):
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    span_lengths = []

    free_intervals = [(0, seq_len)]

    for _ in range(num_spans):
        valid = []
        for idx, (left, right) in enumerate(free_intervals):
            length = right - left
            if length >= min_span:
                valid.append((idx, left, right, length))

        if not valid:
            break

        idx, left, right, length = valid[0]
        span = int(torch.randint(min_span, min(max_span, length) + 1, ()).item())
        start = int(torch.randint(left, right - span + 1, ()).item())
        end = start + span

        mask[start:end] = True
        span_lengths.append(span)

        new_intervals = []
        for j, (free_left, free_right) in enumerate(free_intervals):
            if j != idx:
                new_intervals.append((free_left, free_right))
                continue
            if free_left < start:
                new_intervals.append((free_left, start))
            if end < free_right:
                new_intervals.append((end, free_right))
        free_intervals = new_intervals

    total_mask_len = sum(span_lengths)
    min_mask_len = min(span_lengths) if span_lengths else 0
    max_mask_len = max(span_lengths) if span_lengths else 0

    metrics = {
        "total_mask_len": total_mask_len,
        "min_mask_len": min_mask_len,
        "max_mask_len": max_mask_len,
        "num_spans": len(span_lengths),
    }

    if engine is not None:
        engine.set_metrics(list(metrics.items()))

    return mask


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
        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config)
        self.ema_encoder: EMAModel = EMAModel(self.encoder, decay=float(config.ema_decay))
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
                    "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
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
        self.scaler.step(self.aux_optimizer)
        self.aux_optimizer.zero_grad(set_to_none=True)
        self.aux_lr_scheduler.step()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.ema_encoder.update(self.encoder)
            self.lr_scheduler.step()

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

        num_spans = round(engine.state.get_schedule_value("num_spans", default=2, interp="linear"))
        min_span = self.config.pre_conf.mask_min_span
        max_span = self.config.pre_conf.mask_max_span

        # Build mask
        mask_bool = (
            random_span_mask(seq_len, num_spans, min_span, max_span, engine).unsqueeze(0).expand(batch_size, -1)
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

        engine.set_metrics(
            [
                ("decoder_loss", decoder_loss.detach().item()),
                ("loss_68d", loss_68d.detach().item()),
                ("loss_vel", loss_vel.detach().item()),
                ("loss_feet", loss_feet.detach().item()),
                ("feet_miss_rate", feet_miss_rate.detach().item()),
            ],
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

        timer = Timer(average=False)

        timer.attach(trainer, start=Events.STARTED, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED)

        step_time_avg = RunningAverage(output_transform=lambda _: trainer.get_metric("step_time", 0.0))

        # Step logging
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _log_train_step(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return

            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()

            _step_time_avg = cast(float, step_time_avg.compute())
            remaining_time = estimate_time_remaining(engine, _step_time_avg, self.config)

            engine.set_metrics(
                [
                    ("step_time", step_time),
                    ("remaining_time", remaining_time),
                ]
            )

            metrics = engine.get_metrics(
                [
                    "loss",
                    "mask_loss",
                    "context_loss",
                    "probe_loss",
                    "decoder_loss",
                    "loss_68d",
                    "loss_vel",
                    "loss_feet",
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
                    "step_time": step_time,
                    "remaining_time": remaining_time,
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
            "trainer": trainer,
            "encoder": self.encoder,
            "linear_probe": self.linear_probe,
            "jepa_predictor": self.jepa_predictor,
            "encoder_ema": self.ema_encoder,
            "decoder": self.decoder,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "aux_lr_scheduler": self.aux_lr_scheduler,
            "metadata": CheckpointMetadata({"session_id": self.pretraining_session_id, "resume_id": self.resume_id}),
            "config": CheckpointMetadata(asdict(self.config)),
        }

        def global_step_transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        # Best checkpoint handler
        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_best_val_{self.pretraining_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["loss"]),
            score_name="val_loss",
            global_step_transform=global_step_transform,
        )

        eval_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_best_eval_{self.pretraining_session_id}",
            score_function=lambda engine: -float(engine.state.metrics["decoder_loss"]),
            score_name="eval_loss",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=global_step_transform,
        )

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_latest_{self.pretraining_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=global_step_transform,
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval),
            latest_checkpoint,
        )

        @trainer.on(Events.EPOCH_COMPLETED(every=self.config.val_interval))
        def _run_validation(engine: PretrainEngine) -> None:
            evaluator.run(self.val_loader)

        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def track_batch_loss(engine: PretrainEngine) -> None:
            engine.csa_op_metrics(
                [
                    ("val_loss", engine.get_metric("loss", 0.0)),
                    ("val_decoder_loss", engine.get_metric("decoder_loss", 0.0)),
                    ("val_feet_miss_rate", engine.get_metric("feet_miss_rate", 0.0)),
                ],
                1.0,
            )

        val_batches = self.config.val_batches
        if val_batches > 0:

            @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
            def _limit_val_batches(engine: PretrainEngine) -> None:
                if engine.state.iteration // self.accumulation_steps >= val_batches:
                    engine.terminate()

        @evaluator.on(Events.COMPLETED)
        def _log_best_validation(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return

            batch_count = engine.state.iteration // self.accumulation_steps

            engine.scale_metrics(
                ["val_loss", "val_decoder_loss", "val_feet_miss_rate"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )

            self.wandb_logger.log(
                engine.get_metrics(
                    [
                        "val_loss",
                        "val_decoder_loss",
                        "val_feet_miss_rate",
                        "loss",
                        "mask_loss",
                        "context_loss",
                        "probe_loss",
                        "decoder_loss",
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

            val_loss = float(engine.get_metric("val_loss", float("inf")))
            best_val_loss = float(engine.get_metric("best_val_loss", float("inf")))
            if val_loss < best_val_loss:
                engine.set_metrics([("best_val_loss", val_loss)])
                self.wandb_logger.log({"val/best_loss": val_loss}, step=int(trainer.get_metric("global_step", 0)))

            eval_loss = float(engine.get_metric("val_decoder_loss", float("inf")))
            best_eval_loss = float(engine.get_metric("best_eval_loss", float("inf")))
            if eval_loss < best_eval_loss:
                engine.set_metrics([("best_eval_loss", eval_loss)])
                self.wandb_logger.log({"val/best_eval_loss": eval_loss}, step=int(trainer.get_metric("global_step", 0)))

        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, eval_best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the pretraining loop."""
        print(f"Starting pretraining session: {self.pretraining_session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
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
