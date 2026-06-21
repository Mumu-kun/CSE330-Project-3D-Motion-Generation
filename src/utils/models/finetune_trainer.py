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
from utils.models.motion_history_encoder import MotionHistoryEncoder
from utils.motion_utils import Features, positions_to_x271, x68_to_positions, x271_to_x68
from utils.wandb_logger import WandbLogger


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


class FinetuneTrainer:
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

        # Move to device
        self.encoder.to(self.device)
        self.ema_encoder.to(self.device)
        self.decoder.to(self.device)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ema_encoder.model.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.ema_encoder.model.eval()

        self._log_model_parameters()

        self.wandb_logger: WandbLogger | None = None

        self._initialize()

    def _initialize(self) -> None:
        config = self.config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader, self.normalizer = create_dataloader(self.config, "train", shuffle=True)
        self.val_loader, _ = create_dataloader(self.config, "val", shuffle=False)

        self.accumulation_steps = ceil(config.effective_batch_size / config.batch_size) or 1

        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self._setup_scheduler(num_epochs=int(config.get_num_epochs()))

        if self.wandb_project:
            self.wandb_logger = WandbLogger(
                project=self.wandb_project,
                name=self.finetuning_session_id,
                config={
                    "lr": float(self.config.learning_rate),
                    "weight_decay": float(self.config.weight_decay),
                    "ema_decay": float(self.config.ema_decay),
                    "batch_size": getattr(self.train_loader, "batch_size", self.config.batch_size),
                    "effective_batch_size": self.config.effective_batch_size,
                    "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                    "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
                    "phase": "finetune_decoder",
                    "session_id": self.session_id,
                    "pretrained_checkpoint": str(self.pretrained_checkpoint_path),
                },
                resume_id=self.resume_id,
            )
            self.resume_id = self.wandb_logger.run.id if self.wandb_logger.run else None

    def _load_pretrained_encoder_state(self) -> None:
        if not self.pretrained_checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pretrained_checkpoint_path}")

        checkpoint = _torch_load_with_compat(self.pretrained_checkpoint_path, self.device, weights_only=False)

        # Handle both wrapped and unwrapped metadata/config
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, CheckpointMetadata):
            metadata = metadata.state_dict()
        self.pretraining_session_id = metadata.get("session_id", "unknown") if metadata else "unknown"

        config_raw = checkpoint.get("config")
        config: Config = Config()
        if isinstance(config_raw, CheckpointMetadata):
            config_raw = config_raw.state_dict()
        if isinstance(config_raw, dict):
            # Reconstruct config from dict - Config.load_state_dict handles nested dataclasses
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

    def _log_model_parameters(self) -> None:
        def _summary(name: str, module: nn.Module) -> None:
            total = sum(param.numel() for param in module.parameters())
            trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
            frozen = total - trainable
            print(f"\n{'=' * 60}")
            print(f"Model: {name}")
            print(f"{'=' * 60}")
            print(f"Total parameters     : {total:,}")
            print(f"Trainable parameters : {trainable:,}")
            print(f"Frozen parameters    : {frozen:,}")

        _summary("MotionHistoryEncoder", self.encoder)
        _summary("LatentDecoder", self.decoder)

    def _set_dataset_horizon(self) -> None:
        horizon = int(self.config.horizon)
        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "set_horizon"):
            self.train_loader.dataset.set_horizon(horizon)  # type: ignore[attr-defined]
        if hasattr(self.val_loader, "dataset") and hasattr(self.val_loader.dataset, "set_horizon"):
            self.val_loader.dataset.set_horizon(horizon)  # type: ignore[attr-defined]

    def _setup_scheduler(self, num_epochs: int) -> None:
        assert self.optimizer is not None
        steps_per_epoch = max(ceil(len(self.train_loader) / self.accumulation_steps), 1)
        total_steps = max(num_epochs * steps_per_epoch, 1)
        warmup_epochs = int(self.config.lr_warmup_epochs)
        pct_start = min(max(warmup_epochs / num_epochs, 0.0), 1.0) if num_epochs > 0 else 0.0
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=float(self.config.learning_rate),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    @staticmethod
    def _prepare_text_embedding(text: torch.Tensor) -> torch.Tensor:
        return text[:, -1, :] if text.ndim == 3 else text

    def _train_step(self, engine: PretrainEngine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute one pretraining step."""
        self._decoder = self.decoder
        # Forward pass
        self._decoder.train()

        # Prepare batch
        motion = batch["motion"].to(self.device)
        motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)
        joints = batch["joints"].to(self.device)

        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError(f"Expected motion (B, T, 271), got {tuple(motion.shape)}")

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

        motion = batch["motion"].to(self.device)
        motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)
        joints = batch["joints"].to(self.device)

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

        with torch.amp.autocast(  # type: ignore
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
                ).detach()  # (B, seq_len, L, H)

            latent = target_context[:, 1:, -1, :]  # (B, seq_len-1, H_enc)
            decoded = self._decoder(latent)  # (B, seq_len-1, 68)

            decoder_losses = self._decoder_loss(engine, motion, joints, decoded)
            loss = decoder_losses["decoder_loss"]
            decoder_loss = decoder_losses["decoder_loss"]

        engine.csa_op_metrics(
            [
                ("loss", loss.detach().item()),
            ],
            1.0 / self.accumulation_steps,
            engine.state.iteration % self.accumulation_steps == 1,  # Clear on first step of accumulation
        )

        return {
            "loss": loss,
            "lr": torch.tensor(self.lr_scheduler.get_last_lr()[0], device=self.device),
        }

    def _decoder_loss(self, engine: PretrainEngine, motion, joints, decoded):
        prev_positions = joints[:, :-1, :].flatten(0, 1)  # (B * (seq_len-1), 22, 3)
        prev_frames = motion[:, :-1, :].flatten(0, 1)  # (B * (seq_len-1), 271)
        y_frames = motion[:, 1:, :].flatten(0, 1)  # (B * (seq_len-1), 271)

        decoded = decoded.flatten(0, 1)  # (B * (seq_len-1), 68)
        decoded = self.normalizer.denormalize_x68(decoded)  # Denormalize for loss computation

        y_68d = x271_to_x68(y_frames, self.normalizer, prev_positions)  # (B * (seq_len-1), 68)
        y_68d = self.normalizer.denormalize_x68(y_68d)  # Denormalize for loss computation
        y_vel = y_frames[..., Features.VEL]  # (B * (seq_len-1), 66)
        # y_feet = y_frames[..., Features.CONTACTS]  # (B * (seq_len-1), 4)

        dec_positions = x68_to_positions(decoded, self.normalizer, prev_positions)  # (B * (seq_len-1), 22, 3)
        dec_271d, _ = positions_to_x271(dec_positions, prev_positions, self.normalizer)  # (B * (seq_len-1), 271)
        dec_vel = dec_271d[..., Features.VEL]  # (B * (seq_len-1), 66)
        # dec_feet = dec_271d[..., Features.CONTACTS]  # (B * (seq_len-1), 4)
        # dec_foot_vel = dec_271d[..., Features.joint_mask(["feet"], Features.VEL)]

        loss_root = F.mse_loss(decoded[:, :3], y_68d[:, :3], reduction="mean")
        loss_yaw = F.mse_loss(decoded[:, 3:5], y_68d[:, 3:5], reduction="mean")
        loss_ric = F.mse_loss(decoded[:, 5:], y_68d[:, 5:], reduction="mean")
        loss_vel = F.smooth_l1_loss(dec_vel, y_vel, reduction="mean")
        loss_joint = F.mse_loss(dec_positions, joints[:, 1:, :].flatten(0, 1), reduction="mean")

        # mask_feet_4d = (
        #     y_feet.bool() & ~dec_feet.bool()
        # )  # (B * (seq_len-1), 4) - 4d foot contact false negatives - ground truth contact - prediction does not
        # mask_feet_miss = mask_feet_4d.any(dim=-1)  # (B * (seq_len-1),) - boolean mask for any foot contact miss
        # feet_miss_rate = (
        #     mask_feet_4d.sum().float() / y_feet.bool().sum().float()
        #     if y_feet.bool().sum() > 0
        #     else torch.tensor(0.0, device=self.device)
        # )
        # loss_feet = (
        #     dec_foot_vel[mask_feet_miss].square().mean()
        #     if mask_feet_miss.any()
        #     else torch.tensor(0.0, device=self.device)
        # )

        decoder_loss = 0.2 * loss_root + 1.0 * loss_yaw + 1.0 * loss_ric + 0.5 * loss_vel + 1.0 * loss_joint

        engine.set_metrics(
            [
                ("decoder_loss", decoder_loss.detach().item()),
                ("loss_root", loss_root.detach().item()),
                ("loss_yaw", loss_yaw.detach().item()),
                ("loss_ric", loss_ric.detach().item()),
                ("loss_vel", loss_vel.detach().item()),
                ("loss_joint", loss_joint.detach().item()),
                # ("loss_feet", loss_feet.detach().item()),
                # ("feet_miss_rate", feet_miss_rate.detach().item()),
            ],
        )

        return {
            "decoder_loss": decoder_loss,
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

        step_time_avg = RunningAverage(output_transform=lambda _: trainer.get_metric("step_time", 0.0)).attach(
            trainer, "step_time_avg"
        )

        # Step logging
        @trainer.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _log_train_step(engine: PretrainEngine) -> None:
            if not self.wandb_logger:
                return

            step_time = timer.value() if timer.value() is not None else 0.0
            timer.reset()

            _step_time_avg = (
                trainer.state.metrics["step_time_avg"] if "step_time_avg" in trainer.state.metrics else step_time
            )
            remaining_time = estimate_time_remaining(engine, _step_time_avg, self.config)

            engine.set_metrics(
                [
                    ("step_time", step_time),
                    ("remaining_time", remaining_time),
                ]
            )

            metrics = engine.get_metrics(
                [
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

        @trainer.on(Events.GET_BATCH_STARTED)
        def _set_horizon(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(engine.state.horizon)  # type: ignore[attr-defined]

        @evaluator.on(Events.GET_BATCH_STARTED)
        def _set_horizon_eval(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(trainer.state.horizon)  # type: ignore[attr-defined]

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

        def global_step_transform(engine: PretrainEngine, _) -> int:
            return trainer.get_metric("global_step", 0)

        # Best checkpoint handler
        val_best_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"finetune_best_val_{self.finetuning_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            score_function=lambda engine: -float(engine.state.metrics["loss"]),
            score_name="val_loss",
            global_step_transform=global_step_transform,
        )
        self.val_best_checkpoint = val_best_checkpoint

        # eval_best_checkpoint = Checkpoint(
        #     checkpoint_mapping,
        #     DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
        #     n_saved=1,
        #    filename_prefix=f"finetune_best_eval_{self.finetuning_session_id}",
        #     score_function=lambda engine: -float(engine.state.metrics["decoder_loss"]),
        #     score_name="eval_loss",
        #     filename_pattern="{filename_prefix}_{global_step}.pt",
        #     global_step_transform=global_step_transform,
        # )

        latest_checkpoint = Checkpoint(
            checkpoint_mapping,
            DiskSaver(self.config.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix=f"finetune_latest_{self.finetuning_session_id}",
            filename_pattern="{filename_prefix}_{global_step}.pt",
            global_step_transform=global_step_transform,
        )

        self.latest_checkpoint = latest_checkpoint

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
                    ("val_loss", engine.get_metric("val_loss", 0.0)),
                    ("val_decoder_loss", engine.get_metric("decoder_loss", 0.0)),
                    # ("val_feet_miss_rate", engine.get_metric("feet_miss_rate", 0.0)),
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
                [
                    "val_loss",
                    "val_decoder_loss",
                ],
                1.0 / batch_count if batch_count > 0 else 1.0,
            )

            self.wandb_logger.log(
                engine.get_metrics(
                    [
                        "val_loss",
                        "val_decoder_loss",
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

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the finetuning loop."""
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
