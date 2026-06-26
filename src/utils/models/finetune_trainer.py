"""Decoder-only finetuning trainer for motion latent reconstruction.

Loads a pretrained MotionHistoryEncoder and its EMA copy from checkpoint,
freezes them, and trains a fresh LatentDecoder on the same reconstruction loss
used during pretraining.
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
from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
)
from torch.amp.grad_scaler import GradScaler

from utils.config import Config
from utils.models import CheckpointMetadata, EMAModel, PretrainEngine, _BaseTrainer, estimate_time_remaining
from utils.models.flow_matching_predictor import LatentDecoder
from utils.models.motion_history_encoder import MotionHistoryEncoder


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
    """Load checkpoint with Windows path and module path compatibility."""
    with _windows_checkpoint_path_compat():
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
        finally:
            pass
    return checkpoint


class FinetuneTrainer(_BaseTrainer):
    """Decoder-only finetuning trainer."""

    def __init__(
        self,
        config: Config,
        pretrained_checkpoint_path: str | Path,
        wandb_project: str | None = None,
    ) -> None:
        self.config = config
        self.pretrained_checkpoint_path = Path(pretrained_checkpoint_path)
        self.wandb_project = wandb_project

        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        self.session_id = datetime.now(timezone(timedelta(hours=6))).strftime("%Y%m%d_%H%M%S")
        self.finetuning_session_id = self.session_id
        self.resume_id: str | None = None

        self._load_pretrained_encoder_state()

        self.decoder = LatentDecoder(config).to(self.device)
        self.ema_decoder: EMAModel[LatentDecoder] = EMAModel(self.decoder, decay=float(config.ema_decay)).to(
            self.device
        )

        self.encoder.to(self.device)
        self.ema_encoder.to(self.device)
        self.decoder.to(self.device)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ema_encoder.model.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.ema_encoder.model.eval()

        self._log_model_parameters(
            ("MotionHistoryEncoder", self.encoder),
            ("LatentDecoder", self.decoder),
        )

        self.wandb_logger = None

        self._initialize()

    def _initialize(self) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._create_dataloaders()

        self.accumulation_steps = ceil(self.config.effective_batch_size / self.config.batch_size) or 1

        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.lr_scheduler = self._setup_scheduler(self.optimizer, num_epochs=int(self.config.get_num_epochs()))

        self._init_wandb(
            "finetune_decoder",
            extra_config={
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
                "pretrained_checkpoint": str(self.pretrained_checkpoint_path),
            },
            name=self.finetuning_session_id,
        )

    def _load_pretrained_encoder_state(self) -> None:
        if not self.pretrained_checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_checkpoint_path}")

        checkpoint = _torch_load_with_compat(self.pretrained_checkpoint_path, self.device, weights_only=False)

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
                f"Checkpoint {self.pretrained_checkpoint_path} does not contain encoder or encoder_ema weights"
            )

        primary_state = encoder_state if encoder_state is not None else encoder_ema_state
        ema_state = encoder_ema_state if encoder_ema_state is not None else encoder_state
        assert primary_state is not None

        self.encoder = MotionHistoryEncoder(config).to(self.device)
        self.ema_encoder = EMAModel(self.encoder, decay=float(config.ema_decay)).to(self.device)

        self.encoder.load_state_dict(primary_state)
        self.ema_encoder.load_state_dict(ema_state)

    def _train_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""
        self._decoder = self.decoder
        self._decoder.train()

        motion, text, joints = self._prepare_batch(batch)

        losses = self._compute_loss(engine, motion, joints, text)

        loss = losses["loss"]

        self.scaler.scale(loss / self.accumulation_steps).backward()

        if engine.state.iteration % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.ema_decoder.update(self.decoder)
            self.lr_scheduler.step()

            engine.state.metrics["global_step"] = engine.state.iteration // self.accumulation_steps

        return {
            "loss": loss.detach(),
            "lr": losses["lr"],
        }

    def _val_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one validation step."""
        self._encoder = self.ema_encoder.model
        self._encoder.eval()
        self._decoder = self.ema_decoder.model
        self._decoder.eval()

        motion, text, joints = self._prepare_batch(batch)

        with torch.no_grad():
            losses = self._compute_loss(engine, motion, joints, text)

        return {
            "lr": losses["lr"],
            "val_loss": losses["loss"],
        }

    def _compute_loss(
        self, engine: PretrainEngine, motion: torch.Tensor, joints: torch.Tensor, text: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = motion.shape

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            text_emb = text[:, -1, :] if text.ndim == 3 else text

            with torch.no_grad():
                target_encoder = self.ema_encoder.model
                target_encoder.eval()
                target_context = target_encoder(
                    motion, torch.zeros_like(text_emb), mask=None, return_layer_outputs=True
                ).detach()

            latent = target_context[:, 1:, -1, :]
            decoded = self._decoder(latent)

            decoder_losses = self._decoder_loss(engine, motion, joints, decoded)
            loss = decoder_losses["decoder_loss"]
            decoder_loss = decoder_losses["decoder_loss"]

        engine.csa_op_metrics(
            [("loss", loss.detach().item())],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,
        )

        return {
            "loss": loss,
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
        }

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
                    "decoder_loss",
                    "loss_root_y",
                    "loss_root_xz",
                    "loss_yaw",
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
                ["val_loss", "val_decoder_loss"],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )
            self.wandb_logger.log(
                engine.get_metrics(
                    ["val_loss", "val_decoder_loss", "decoder_loss"],
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
                self.wandb_logger.log({"val/best_loss": val_loss}, step=int(trainer.get_metric("global_step", 0)))
            eval_loss = float(engine.get_metric("val_decoder_loss", float("inf")))
            best_eval_loss = float(engine.get_metric("best_eval_loss", float("inf")))
            if eval_loss < best_eval_loss:
                engine.set_metrics([("best_eval_loss", eval_loss)])
                self.wandb_logger.log({"val/best_eval_loss": eval_loss}, step=int(trainer.get_metric("global_step", 0)))

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            engine.csa_op_metrics(
                [
                    ("val_loss", engine.get_metric("val_loss", 0.0)),
                    ("val_decoder_loss", engine.get_metric("decoder_loss", 0.0)),
                ],
                1.0,
            )

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        super()._attach_handlers(trainer, evaluator)

        checkpoint_mapping = {
            "trainer": trainer,
            "encoder": self.encoder,
            "encoder_ema": self.ema_encoder,
            "decoder": self.decoder,
            "decoder_ema": self.ema_decoder,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
            "lr_scheduler": self.lr_scheduler,
            "metadata": CheckpointMetadata(
                {"pretrain_id": self.pretraining_session_id, "session_id": self.session_id, "resume_id": self.resume_id}
            ),
            "config": CheckpointMetadata(asdict(self.config)),
        }

        def _global_step_transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"finetune_best_val_{self.finetuning_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["loss"]),
            score_name="val_loss",
            global_step_transform=_global_step_transform,
        )
        self.val_best_checkpoint = val_best_checkpoint

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"finetune_latest_{self.finetuning_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=_global_step_transform,
        )
        self.latest_checkpoint = latest_checkpoint

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.config.checkpoint_interval), latest_checkpoint)
        evaluator.add_event_handler(Events.COMPLETED, val_best_checkpoint)

    def run(self, max_epochs: int | None = None) -> None:
        print(f"Starting finetuning session: {self.session_id}")
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        self.evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self._attach_handlers(trainer, self.evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


def train_finetune(
    config: Config,
    pretrained_checkpoint_path: str | Path,
    wandb_project: str | None = None,
    max_epochs: int | None = None,
) -> Tuple[EMAModel[MotionHistoryEncoder], EMAModel[LatentDecoder], Path]:
    """Train the decoder while reusing a pretrained encoder checkpoint."""

    trainer = FinetuneTrainer(
        config=config,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        wandb_project=wandb_project,
    )
    trainer.run(max_epochs=max_epochs)
    checkpoint_path = trainer.val_best_checkpoint.last_checkpoint
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path()
    return trainer.ema_encoder, trainer.ema_decoder, checkpoint_path


__all__ = ["FinetuneTrainer", "train_finetune"]
