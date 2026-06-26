"""Shared mixin and training utilities for motion-history trainer classes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Generic, OrderedDict, TypeVar, Union
from typing import Mapping as MappingABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events, State
from ignite.handlers import (
    TerminateOnNan,
    Timer,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import RunningAverage
from scipy import interpolate as Interp

from utils.config import Config
from utils.dataset import create_dataloader
from utils.motion_utils import x68_to_positions, x271_to_x68
from utils.wandb_logger import WandbLogger

T = TypeVar("T", bound=nn.Module)


class PretrainState(State):
    """Custom Ignite State for JEPA pretraining."""

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.horizon = 40
        self.metrics: Dict[str, Any] = {
            "global_step": 0,
            "best_train_loss": float("inf"),
        }
        self.schedules = config.pre_conf.schedules
        self.num_epochs = config.get_num_epochs()
        self.timer: Any = None

    def epoch_progress(self) -> float:
        """Return progress through current epoch as a float in [0, 1]."""
        return self.epoch / self.num_epochs if self.num_epochs > 0 else 0.0

    def get_schedule_value(self, name: str, default: float, interp: str = "previous") -> float:
        """Get current value of a scheduled parameter based on epoch progress."""
        schedule = self.schedules.get(name)

        if not schedule:
            return default

        t = self.epoch_progress()
        t_list, v_list = zip(*schedule)
        t_list = np.array(t_list)
        v_list = np.array(v_list)
        t_list = t_list / np.max(t_list)

        f = Interp.interp1d(t_list, v_list, kind=interp, assume_sorted=True)

        return float(f(t))

    def state_dict(self) -> dict:
        """Return a dictionary containing the state of the trainer."""
        super_dict: OrderedDict = super().state_dict()
        super_dict.update(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "metrics": self.metrics,
                "schedules": self.schedules,
                "num_epochs": self.num_epochs,
                "horizon": self.horizon,
            }
        )
        return super_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the trainer from a dictionary."""
        super().load_state_dict(state_dict)
        self.epoch = state_dict.get("epoch", 0)
        self.iteration = state_dict.get("iteration", 0)
        self.metrics = state_dict.get("metrics", {})
        self.schedules = state_dict.get("schedules", {})
        self.num_epochs = state_dict.get("num_epochs", 0)
        self.horizon = state_dict.get("horizon", 40)


class PretrainEngine(Engine):
    """Custom Ignite Engine for JEPA pretraining."""

    def __init__(self, process_function: Any, config: Config) -> None:
        super().__init__(process_function)
        self.state = PretrainState(config=config)
        self.config = config

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

    def scale_metrics(self, names: list[str], scaler: float) -> None:
        """Scale specified metrics by a factor."""
        for name in names:
            self.state.metrics[name] *= scaler

    def csa_op_metrics(self, pairs: list[tuple[str, float]], scalers: float | list[float], clear: bool = False) -> None:
        """Add a value to an existing metric (useful for running totals)."""
        if clear:
            self.clear_metrics([name for name, _ in pairs])
        if not isinstance(scalers, list):
            scalers = [scalers for _ in pairs]
        for (name, value), scaler in zip(pairs, scalers):
            self.state.metrics[name] = self.state.metrics.get(name, 0.0) + value * scaler


def estimate_time_remaining(engine: PretrainEngine, step_time: float, config: Config) -> float:
    """Estimate remaining training time based on current progress and elapsed time."""
    num_epochs = config.get_num_epochs()
    epoch = engine.state.epoch
    global_step = engine.get_metric("global_step")
    steps_per_epoch = global_step / epoch
    total_steps = num_epochs * steps_per_epoch
    remaining_steps = total_steps - global_step

    remaining_time = remaining_steps * step_time
    return remaining_time


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


def _find_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> Path | None:
    """Find the latest checkpoint file with given prefix."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


class _BaseTrainer:
    """Shared mixin for PretrainTrainer and FinetuneTrainer.

    Rust-style composition: no __init__; concrete trainers keep their own
    construction and call mixin helpers via self.<method>().
    """

    # -- dataloader + scheduler + wandb helpers --

    def _create_dataloaders(self) -> None:
        self.train_loader, self.normalizer = create_dataloader(self.config, "train", shuffle=True)
        self.val_loader, _ = create_dataloader(self.config, "val", shuffle=False)

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer, num_epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        accumulation_steps = (
            self.accumulation_steps
            if hasattr(self, "accumulation_steps") and self.accumulation_steps is not None
            else 1
        )
        steps_per_epoch = max(len(self.train_loader) // accumulation_steps, 1)
        total_steps = max(num_epochs * steps_per_epoch, 1)
        warmup_epochs = int(self.config.lr_warmup_epochs)
        pct_start = min(max(warmup_epochs / num_epochs, 0.0), 1.0) if num_epochs > 0 else 0.0
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.config.learning_rate),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    def _init_wandb(self, phase: str, extra_config: dict[str, Any] | None = None, name: str | None = None) -> None:
        if not getattr(self, "wandb_project", None):
            return
        cfg: dict[str, Any] = {
            "lr": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "ema_decay": float(self.config.ema_decay),
            "batch_size": getattr(self.train_loader, "batch_size", self.config.batch_size),
            "effective_batch_size": self.config.effective_batch_size,
            "phase": phase,
        }
        if extra_config:
            cfg.update(extra_config)
        self.wandb_logger = WandbLogger(
            project=self.wandb_project,
            name=name,
            config=cfg,
            resume_id=self.resume_id if self.resume_id else None,
        )
        self.resume_id = self.wandb_logger.run.id if self.wandb_logger.run else None

    # -- batch prep --

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion = batch["motion"].to(self.device)
        motion = self.normalizer.normalize(motion)
        text = batch["text_clip"].to(self.device)
        joints = batch["joints"].to(self.device)
        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError(f"Expected motion (B, T, 271), got {tuple(motion.shape)}")
        return motion, text, joints

    # -- decoder loss (shared, fixes latent bug in PretrainTrainer) --

    def _decoder_loss(
        self, engine: PretrainEngine, motion: torch.Tensor, joints: torch.Tensor, decoded: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prev_positions = joints[:, :-1, :].flatten(0, 1)
        prev_frames = motion[:, :-1, :].flatten(0, 1)
        y_frames = motion[:, 1:, :].flatten(0, 1)

        decoded_flat = decoded.flatten(0, 1)
        decoded_denorm = self.normalizer.denormalize_x68(decoded_flat)

        y_68d = x271_to_x68(y_frames, self.normalizer, prev_positions=prev_positions, prev_x271=prev_frames)
        y_68d = self.normalizer.denormalize_x68(y_68d)

        dec_positions = x68_to_positions(
            decoded_denorm,
            self.normalizer,
            prev_x271=prev_frames,
            prev_positions=prev_positions,
        )

        loss_root_y = F.mse_loss(decoded_denorm[:, :1], y_68d[:, :1], reduction="mean")
        loss_root_xz = F.mse_loss(decoded_denorm[:, 1:3], y_68d[:, 1:3], reduction="mean")
        loss_yaw = F.mse_loss(decoded_denorm[:, 3:5], y_68d[:, 3:5], reduction="mean")
        loss_vel = F.smooth_l1_loss(decoded_denorm[:, 5:], y_68d[:, 5:], reduction="mean")
        loss_joint = F.mse_loss(dec_positions, joints[:, 1:, :].flatten(0, 1), reduction="mean")

        decoder_loss = 1 * loss_root_y + 0.2 * loss_root_xz + 1.0 * loss_yaw + 0.5 * loss_vel

        engine.set_metrics(
            [
                ("decoder_loss", decoder_loss.detach().item()),
                ("loss_root_y", loss_root_y.detach().item()),
                ("loss_root_xz", loss_root_xz.detach().item()),
                ("loss_yaw", loss_yaw.detach().item()),
                ("loss_vel", loss_vel.detach().item()),
                ("loss_joint", loss_joint.detach().item()),
            ],
        )

        return {"decoder_loss": decoder_loss}

    # -- model parameter logging --

    @staticmethod
    def _log_model_parameters(*models: tuple[str, nn.Module]) -> None:
        for name, model in models:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            print(f"\n{'=' * 60}")
            print(f"Model: {name}")
            print(f"{'=' * 60}")
            print(f"Total parameters     : {total:,}")
            print(f"Trainable parameters : {trainable:,}")
            print(f"Frozen parameters    : {frozen:,}")
            if total > 0:
                print("-" * 60)
                print(f"{'Category':<30} {'Params':>12} {'%':>7}")
                print("-" * 60)
                categories: dict[str, int] = {}
                for pname, param in model.named_parameters():
                    cat = _assign_category(pname)
                    categories[cat] = categories.get(cat, 0) + param.numel()
                for cat, count in categories.items():
                    pct = count / total * 100
                    print(f"{cat:<30} {count:>12,} {pct:>6.2f}%")
            print("-" * 60)

    # -- handler hooks (override in subclasses) --

    def _log_train_step(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach per-iteration training log handler."""

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
            metrics = engine.get_metrics(["loss", "lr"], prefix="train/")
            self.wandb_logger.log(metrics, step=engine.get_metric("global_step", 0))

    def _log_best_validation(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach COMPLETED handler that logs validation metrics."""

        @evaluator.on(Events.COMPLETED)
        def _handler(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            self.wandb_logger.log(
                engine.get_metrics(["val_loss", "loss"], prefix="val/"),
                step=int(trainer.get_metric("global_step", 0)),
            )
            self.wandb_logger.log(
                {"epoch": int(trainer.state.epoch), "global_step": int(trainer.get_metric("global_step", 0))},
                step=int(trainer.get_metric("global_step", 0)),
            )

    def _track_batch_loss(self, evaluator: PretrainEngine) -> None:
        """Attach per-iteration running-val-loss handler."""

        @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
        def _handler(engine: PretrainEngine) -> None:
            engine.csa_op_metrics([("val_loss", engine.get_metric("loss", 0.0))], 1.0)

    # -- common handler scaffolding --

    def _limit_val_batches(self, evaluator: PretrainEngine) -> None:
        val_batches = self.config.val_batches
        if val_batches > 0:

            @evaluator.on(Events.ITERATION_COMPLETED(every=self.accumulation_steps))
            def _handler(engine: PretrainEngine) -> None:
                if engine.state.iteration // self.accumulation_steps >= val_batches:
                    engine.terminate()

    def _attach_handlers(self, trainer: PretrainEngine, evaluator: PretrainEngine) -> None:
        """Attach Ignite event handlers for training orchestration. Subclasses override hook methods to customize."""
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        ProgressBar(mininterval=10.0).attach(trainer, ["loss"])
        trainer.state.horizon = self.config.horizon

        timer = Timer(average=False)
        timer.attach(trainer, start=Events.STARTED, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED)
        RunningAverage(output_transform=lambda _: trainer.get_metric("step_time", 0.0)).attach(trainer, "step_time_avg")

        trainer.state.timer = timer

        @trainer.on(Events.GET_BATCH_STARTED)
        def _set_horizon(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(engine.state.horizon)

        @evaluator.on(Events.GET_BATCH_STARTED)
        def _set_horizon_eval(engine: PretrainEngine) -> None:
            engine.state.dataloader.dataset.set_horizon(trainer.state.horizon)

        self._limit_val_batches(evaluator)

        @trainer.on(Events.EPOCH_COMPLETED(every=self.config.val_interval))
        def _run_validation(engine: PretrainEngine) -> None:
            evaluator.run(self.val_loader)

        self._log_train_step(trainer, evaluator)
        self._track_batch_loss(evaluator)
        self._log_best_validation(trainer, evaluator)

        @trainer.on(Events.EPOCH_STARTED)
        def _reset_train_loss(engine: PretrainEngine) -> None:
            engine.state.metrics["loss"] = 0.0

        @trainer.on(Events.EPOCH_COMPLETED)
        def _track_best_train_loss(engine: PretrainEngine) -> None:
            if self.wandb_logger is None:
                return
            batch_count = engine.state.iteration // self.accumulation_steps
            if batch_count > 0:
                engine.scale_metrics(["loss"], 1.0 / batch_count)
            epoch_loss = float(engine.get_metric("loss", float("inf")))
            best_train_loss = float(engine.get_metric("best_train_loss", float("inf")))
            if epoch_loss < best_train_loss:
                engine.set_metrics([("best_train_loss", epoch_loss)])
                self.wandb_logger.log(
                    {"train/best_train_loss": epoch_loss},
                    step=int(trainer.get_metric("global_step", 0)),
                )

    # -- public run --

    def run(self, max_epochs: int | None = None) -> None:
        """Execute the training loop."""
        trainer = PretrainEngine(lambda engine, batch: self._train_step(engine, batch), self.config)
        evaluator = PretrainEngine(lambda engine, batch: self._val_step(engine, batch), self.config)
        self.evaluator = evaluator
        self._attach_handlers(trainer, evaluator)
        epochs = max_epochs or int(self.config.get_num_epochs())
        trainer.run(self.train_loader, max_epochs=epochs)
        if self.wandb_logger:
            self.wandb_logger.finish()


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
