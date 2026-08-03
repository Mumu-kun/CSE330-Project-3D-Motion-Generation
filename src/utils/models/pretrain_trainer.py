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
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
)

from utils.config import Config
from utils.models import CheckpointMetadata, EMAModel, PretrainEngine, _BaseTrainer, estimate_time_remaining
from utils.models.flow_matching_predictor import LatentDecoder
from utils.models.motion_history_encoder import JepaPredictor, LinearProbe, MotionHistoryEncoder


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

        weights = torch.tensor([v[3] for v in valid], dtype=torch.float)
        choice_idx = int(torch.multinomial(weights, 1).item())

        idx, left, right, length = valid[choice_idx]
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


class PretrainTrainer(_BaseTrainer):
    """
    JEPA-style pretraining trainer with masked frame reconstruction.

    Fully self-contained:, and Ignite engine.
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

        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.use_amp)
        )

        self.encoder: MotionHistoryEncoder = MotionHistoryEncoder(config)
        self.ema_encoder: EMAModel = EMAModel(self.encoder, decay=float(config.ema_decay))
        self.jepa_predictor: JepaPredictor = JepaPredictor(config.encoder_config)
        self.decoder: LatentDecoder = LatentDecoder(config)

        self.linear_probe: LinearProbe = LinearProbe(
            hidden_size=config.encoder_config.hidden_size,
            text_embedding_dim=config.text_embedding_dim,
        ).to(self.device)

        self._log_model_parameters(
            ("MotionHistoryEncoder", self.encoder),
            ("JepaPredictor", self.jepa_predictor),
            ("LinearProbe", self.linear_probe),
            ("LatentDecoder", self.decoder),
        )

        self.wandb_logger = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize all components: models, optimizer, W&B."""
        config = self.config

        utc_plus_6 = timezone(timedelta(hours=6))
        self.session_id = datetime.now(utc_plus_6).strftime("%Y%m%d_%H%M%S")
        self.pretraining_session_id = self.session_id

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._create_dataloaders()

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

        self.encoder.to(self.device)
        self.jepa_predictor.to(self.device)
        self.linear_probe.to(self.device)
        self.decoder.to(self.device)
        self.ema_encoder.to(self.device)

        self.accumulation_steps = ceil(config.effective_batch_size / config.batch_size) or 1

        num_epochs = int(config.get_num_epochs())
        self.lr_scheduler = self._setup_scheduler(self.optimizer, num_epochs)
        self.aux_lr_scheduler = self._setup_scheduler(self.aux_optimizer, num_epochs)

        self.resume_id = None

        self._init_wandb(
            "pretrain",
            extra_config={
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "jepa_params": sum(p.numel() for p in self.jepa_predictor.parameters()),
                "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
            },
            name=self.pretraining_session_id,
        )
        if self.wandb_logger:
            print(f"Initialized W&B run with ID: {self.resume_id}")

    def _train_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""
        self.encoder.train()
        self.jepa_predictor.train()
        self.linear_probe.train()
        self.decoder.train()

        motion, text, joints = self._prepare_batch(batch)

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
            self.aux_optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)

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

        motion, text, joints = self._prepare_batch(batch)

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

        mask_bool = (
            random_span_mask(seq_len, num_spans, min_span, max_span, engine).unsqueeze(0).expand(batch_size, -1)
        ).to(self.device)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            text_emb = text[:, -1, :] if text.ndim == 3 else text
            masked_context = self.encoder.forward(
                motion, torch.zeros_like(text_emb), mask=mask_bool, return_layer_outputs=True
            )

            with torch.no_grad():
                target_encoder = self.ema_encoder.model
                target_encoder.eval()
                target_context = target_encoder(
                    motion, torch.zeros_like(text_emb), mask=None, return_layer_outputs=True
                ).detach()

            _, _, L, H = masked_context.shape

            predicted = self.jepa_predictor(masked_context)

            _loss = F.smooth_l1_loss(predicted, target_context, reduction="none").mean(dim=(2, 3))
            mask_loss = (_loss * mask_bool.float()).sum() / mask_bool.sum().clamp(min=1)
            context_loss = self._context_loss(engine, _loss, mask_bool)
            loss = mask_loss + context_loss * self.config.jepa_ctx_weight

            probe_out = self.linear_probe(target_context[:, :, -1, :])
            probe_out = F.normalize(probe_out, dim=-1)
            normalized_text = F.normalize(text_emb, dim=-1)
            probe_loss = 1 - F.cosine_similarity(probe_out, normalized_text, dim=-1).mean()

            latent = target_context[:, 1:, -1, :]
            decoded = self.decoder(latent)

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
            engine.state.iteration % self.accumulation_steps == 1,
        )

        return {
            "loss": loss,
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
            "probe_loss": probe_loss,
            "decoder_loss": decoder_loss,
        }

    def _context_loss(self, engine: PretrainEngine, _loss: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask_bool.shape
        all_mask_indices = [mask_bool[b].nonzero(as_tuple=False).squeeze(1) for b in range(batch_size)]

        pos = torch.arange(seq_len, device=self.device)
        weights = torch.zeros(batch_size, seq_len, device=self.device)

        pos_exp = pos.unsqueeze(0).unsqueeze(2)
        mask_idx_exp = torch.stack(all_mask_indices).unsqueeze(1)
        distances = (pos_exp - mask_idx_exp).abs()
        min_distances = distances.min(dim=2).values
        weights = 1.0 / torch.sqrt(min_distances + 1e-8)
        weights[mask_bool] = 0.0

        unmasked_bool = ~mask_bool
        context_loss = (_loss * unmasked_bool.float() * weights.detach()).sum() / (unmasked_bool.sum().float() + 1e-8)

        return context_loss

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return
            timer = engine.state.timer
            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()
            step_time_avg = (
                engine.state.metrics["step_time_avg"] if "step_time_avg" in engine.state.metrics else step_time
            )
            remaining_time = estimate_time_remaining(engine, step_time_avg, self.config)
            engine.set_metrics([("step_time", step_time), ("remaining_time", remaining_time)])
            metrics = engine.get_metrics(
                [
                    "loss",
                    "mask_loss",
                    "context_loss",
                    "probe_loss",
                    "decoder_loss",
                    "loss_root",
                    "loss_yaw",
                    "loss_ric",
                    "loss_vel",
                    "loss_joint",
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

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            engine.scale_metrics(
                ["val_loss", "val_decoder_loss", "val_probe_loss"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )
            self.wandb_logger.log(
                engine.get_metrics(
                    [
                        "val_loss",
                        "val_decoder_loss",
                        "val_probe_loss",
                    ],
                    prefix="val/",
                ),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {"epoch": int(trainer.state.epoch), "global_step": int(trainer.get_metric("global_step", 0))},
                step=int(trainer.get_metric("global_step", 0)),
            )
            val_loss = float(engine.get_metric("val_loss", float("inf")))
            best_val_loss = float(engine.get_metric("best_val_loss", float("inf")))
            if val_loss < best_val_loss:
                engine.set_metrics([("best_val_loss", val_loss)])
                self.wandb_logger.log({"val/best_val_loss": val_loss}, step=int(trainer.get_metric("global_step", 0)))
            eval_loss = float(engine.get_metric("val_decoder_loss", float("inf")))
            best_eval_loss = float(engine.get_metric("best_eval_loss", float("inf")))
            if eval_loss < best_eval_loss:
                engine.set_metrics([("best_eval_loss", eval_loss)])
                self.wandb_logger.log({"val/best_eval_loss": eval_loss}, step=int(trainer.get_metric("global_step", 0)))

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            # Ignite stores process_function return dict in state.output, not state.metrics
            output = engine.state.output if isinstance(engine.state.output, dict) else {}
            engine.csa_op_metrics(
                [
                    ("val_loss", float(output.get("val_loss", 0.0))),
                    ("val_decoder_loss", float(output.get("val_decoder_loss", 0.0))),
                    ("val_probe_loss", float(output.get("val_probe_loss", 0.0))),
                ],
                1.0,
            )

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        super()._attach_handlers(trainer, evaluator)

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

        global_step_transform = self._create_global_step_transform(trainer)

        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_best_val_{self.pretraining_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=global_step_transform,
        )

        eval_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"pretrain_best_eval_{self.pretraining_session_id}",
            score_function=lambda engine: -float(engine.state.metrics["val_decoder_loss"]),
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

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval), latest_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, eval_best_checkpoint)

    @staticmethod
    def _create_global_step_transform(trainer: PretrainEngine):
        def _transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        return _transform

    def run(self, max_epochs: int | None = None) -> None:
        print(f"Starting pretraining session: {self.pretraining_session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self._attach_handlers(trainer, self.evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


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
