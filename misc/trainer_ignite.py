"""Ignite-driven trainer starter for the motion generation models.

Ignite owns the loop, validation cadence, checkpointing, and timing hooks.
The runtime state lives on `engine.state` so step, summary, and checkpoint
handlers all read from the same place.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, Union, TypeVar, Generic, Protocol
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    TerminateOnNan,
    global_step_from_engine,
)
from ignite.metrics import RunningAverage
from torch.amp.grad_scaler import GradScaler

from config import Config
from models import FlowMatchingPredictor, MotionHistoryEncoder
from utils.models.motion_history_encoder import JepaPredictor
from utils.motion_utils import (
    FeatureNormalizer,
    x271_to_x68,
    x271_to_positions,
)
from utils.wandb_logger import WandbLogger

T = TypeVar("T", bound=nn.Module)
BatchDict = dict[str, Any]


class TrainingPhase(Enum):
    """Training phase enumeration for two-step training."""
    PRETRAINING = "pretrain"
    FINETUNING = "finetune"


class TrainingStrategy(Protocol):
    """Protocol for training phase-specific strategies."""
    
    learning_rate: float
    weight_decay: float
    num_epochs: int
    ema_decay: float
    
    def configure_trainer(self, trainer: IgniteMotionTrainer) -> None: ...
    def get_checkpoint_prefix(self) -> str: ...


_ENGINE_STATE_DEFAULTS: dict[str, Any] = {
    "global_step": 0,
    "epoch": 0,
    "current_horizon": 0,
    "last_train_loss": float("nan"),
    "last_val_loss": float("nan"),
    "best_train_loss": float("inf"),
    "best_val_loss": float("inf"),
    "latest_metrics": {},
}


def _engine_state_snapshot(engine_state: Any) -> dict[str, Any]:
    return {
        "global_step": int(getattr(engine_state, "global_step", 0)),
        "epoch": int(getattr(engine_state, "epoch", 0)),
        "current_horizon": int(getattr(engine_state, "current_horizon", 0)),
        "last_train_loss": float(
            getattr(engine_state, "last_train_loss", float("nan"))
        ),
        "last_val_loss": float(getattr(engine_state, "last_val_loss", float("nan"))),
        "best_train_loss": float(
            getattr(engine_state, "best_train_loss", float("inf"))
        ),
        "best_val_loss": float(getattr(engine_state, "best_val_loss", float("inf"))),
        "latest_metrics": dict(getattr(engine_state, "latest_metrics", {})),
    }


def _apply_engine_state(engine_state: Any, state: MappingABC) -> None:
    engine_state.global_step = int(state.get("global_step", 0))
    engine_state.epoch = int(state.get("epoch", 0))
    engine_state.current_horizon = int(state.get("current_horizon", 0))
    engine_state.last_train_loss = float(state.get("last_train_loss", float("nan")))
    engine_state.last_val_loss = float(state.get("last_val_loss", float("nan")))
    engine_state.best_train_loss = float(state.get("best_train_loss", float("inf")))
    engine_state.best_val_loss = float(state.get("best_val_loss", float("inf")))
    engine_state.latest_metrics = dict(state.get("latest_metrics", {}))


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


class EngineStateCheckpointProxy:
    """Tiny checkpoint wrapper around Ignite engine.state."""

    def __init__(self, engine_state: Any):
        self.engine_state = engine_state

    def state_dict(self) -> dict[str, Any]:
        return _engine_state_snapshot(self.engine_state)

    def load_state_dict(self, state: MappingABC) -> None:
        _apply_engine_state(self.engine_state, state)


class IgniteMotionTrainer:
    """Minimal Ignite trainer that keeps orchestration in Ignite handlers."""

    def __init__(
        self,
        *,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
        config: Config,
        dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        wandb_logger: Optional[WandbLogger] = None,
        checkpoint_dir: Optional[Path | str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        log_to_console: bool = True,
        log_to_wandb: bool = False,
        log_every_n_steps: int = 50,
        validate_every_n_epochs: int = 1,
        save_every_n_epochs: int = 1,
        use_ema_for_validation: bool = True,
        jepa_predictor: Optional[JepaPredictor] = None,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.jepa_predictor = jepa_predictor
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.normalizer = normalizer
        self.wandb_logger = wandb_logger
        self.checkpoint_dir = Path(checkpoint_dir or config.checkpoint_dir)
        self.log_to_console = bool(log_to_console)
        self.log_to_wandb = bool(log_to_wandb and wandb_logger is not None)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.validate_every_n_epochs = max(1, int(validate_every_n_epochs))
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))
        self.use_ema_for_validation = bool(use_ema_for_validation)

        self.device = torch.device(config.device)
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = (
            scaler if scaler is not None else GradScaler("cuda", enabled=self.use_amp)
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                self._collect_trainable_params(),
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
            )
        )

        self.encoder.to(self.device)
        self.predictor.to(self.device)
        if self.jepa_predictor is not None:
            self.jepa_predictor.to(self.device)
        self.ema_encoder = EMAModel(self.encoder, decay=float(config.ema_decay)).to(
            self.device
        )
        self.ema_predictor = EMAModel(self.predictor, decay=float(config.ema_decay)).to(
            self.device
        )
        if self.jepa_predictor is not None:
            self.ema_jepa: Optional[EMAModel[JepaPredictor]] = EMAModel(
                self.jepa_predictor, decay=float(config.ema_decay)
            ).to(self.device)
        else:
            self.ema_jepa = None

        # Central runtime state lives on `trainer.state` so logging and
        # checkpointing stay on the same source of truth.
        self.state: Any = None
        self._state_proxy: Optional[EngineStateCheckpointProxy] = None
        self.trainer: Optional[Engine] = None
        self.evaluator: Optional[Engine] = None

    def _get_positions_from_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """Extract joint positions from motion features using x271_to_positions."""
        return x271_to_positions(motion, self.normalizer)

    def _initialize_engine_state(self, engine_state: Any) -> None:
        """Initialize engine state with default values for training."""
        engine_state.global_step = _ENGINE_STATE_DEFAULTS["global_step"]
        engine_state.epoch = _ENGINE_STATE_DEFAULTS["epoch"]
        engine_state.current_horizon = int(self.config.horizon)
        engine_state.last_train_loss = _ENGINE_STATE_DEFAULTS["last_train_loss"]
        engine_state.last_val_loss = _ENGINE_STATE_DEFAULTS["last_val_loss"]
        engine_state.best_train_loss = _ENGINE_STATE_DEFAULTS["best_train_loss"]
        engine_state.best_val_loss = _ENGINE_STATE_DEFAULTS["best_val_loss"]
        engine_state.latest_metrics = _ENGINE_STATE_DEFAULTS["latest_metrics"]

    def _prepare_batch(self, batch: Any) -> BatchDict:
        if not isinstance(batch, MappingABC):
            raise TypeError("Expected batch to be a mapping with motion/text tensors.")

        prepared = dict(batch)
        prepared["motion"] = prepared["motion"].to(self.device)
        prepared["text_clip"] = prepared["text_clip"].to(self.device)
        if self.normalizer is not None:
            prepared["motion"] = self.normalizer.normalize(prepared["motion"])
        return prepared

    def _text_embedding(self, batch: BatchDict) -> torch.Tensor:
        text = batch["text_clip"]
        return text[:, 0, :] if text.ndim == 3 else text

    def _validation_models(self) -> tuple[MotionHistoryEncoder, FlowMatchingPredictor]:
        if self.use_ema_for_validation:
            return self.ema_encoder.model, self.ema_predictor.model
        return self.encoder, self.predictor

    def _compute_loss(
        self,
        batch: BatchDict,
        *,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
    ) -> torch.Tensor:
        motion = batch["motion"]
        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError("Expected motion shape (B, T, 271) with T >= 2.")

        history = motion[:, :-1]
        target_frame = motion[:, -1]
        current_frame = history[:, -1]
        text_embedding = self._text_embedding(batch)

        track_features = encoder(history, text_embedding, return_all=False)
        target_flow = x271_to_x68(
            target_frame,
            prev_positions=self._get_positions_from_motion(current_frame),
            normalizer=self.normalizer,
        )
        noise = torch.randn_like(target_flow)
        timesteps = torch.rand(
            target_flow.shape[0],
            device=self.device,
            dtype=target_flow.dtype,
        )
        noisy_flow = (
            timesteps.unsqueeze(1) * target_flow
            + (1.0 - timesteps.unsqueeze(1)) * noise
        )

        predicted_flow, _, _ = predictor(
            noisy_features=noisy_flow,
            timesteps=timesteps,
            text_embedding=text_embedding,
            track_features=track_features,
            current_frame_features=None,
            output_attentions=False,
            output_hidden_states=False,
        )
        return F.mse_loss(predicted_flow, target_flow - noise)

    def _compute_jepa_loss(
        self,
        batch: BatchDict,
        *,
        encoder: MotionHistoryEncoder,
        jepa_predictor: JepaPredictor,
        target_encoder: Optional[MotionHistoryEncoder] = None,
        mask_ratio: float = 0.25,
        context_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        motion = batch["motion"]
        if motion.ndim != 3 or motion.shape[1] < 2:
            raise ValueError("Expected motion shape (B, T, 271) with T >= 2.")

        text_embedding = self._text_embedding(batch)

        batch_size, seq_len, _ = motion.shape
        num_masked = max(1, int(seq_len * mask_ratio))

        mask_indices_list = []
        for b in range(batch_size):
            perm = torch.randperm(seq_len, device=self.device)
            mask_indices_list.append(perm[:num_masked])
        mask_indices = torch.stack(mask_indices_list, dim=0)

        mask_bool = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            mask_bool[b, mask_indices[b]] = True
        unmasked_bool = ~mask_bool

        if target_encoder is None:
            target_encoder = self.ema_encoder.model

        masked_context_all = encoder(
            motion, text_embedding, mask=mask_bool, return_all=True
        )

        with torch.no_grad():
            target_encoder.eval()
            target_context_all = target_encoder(
                motion, text_embedding, mask=None, return_all=True
            ).detach()

        b_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, num_masked)
        masked_tokens = masked_context_all[b_idx, mask_indices, :].reshape(batch_size * num_masked, -1)
        target_masked = target_context_all[b_idx, mask_indices, :].reshape(batch_size * num_masked, -1)

        predicted = jepa_predictor(masked_tokens)
        mask_loss = F.smooth_l1_loss(predicted, target_masked)

        token_diff = F.smooth_l1_loss(
            masked_context_all, target_context_all, reduction="none"
        ).mean(dim=-1)

        pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
        mask_idx_exp = mask_indices.unsqueeze(1)
        distances = torch.abs(pos.unsqueeze(0) - mask_idx_exp)
        min_distances = distances.min(dim=2).values

        weights = 1.0 / torch.sqrt(min_distances + 1.0)

        context_loss = (token_diff * unmasked_bool.float() * weights).sum()
        context_loss = context_loss / (unmasked_bool.sum().float() + 1e-8)

        return mask_loss + context_loss_weight * context_loss

    def _collect_trainable_params(self):
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.jepa_predictor is not None:
            params += list(self.jepa_predictor.parameters())
        return params

    def _current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    # Ignite owns the loop; these handlers only compute loss, update the
    # shared engine.state, and route logs to console and/or wandb.
    def _train_step(self, engine: Engine, batch: Any) -> dict[str, torch.Tensor]:
        prepared_batch = self._prepare_batch(batch)
        self.encoder.train()
        self.predictor.train()

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            loss = self._compute_loss(
                prepared_batch,
                encoder=self.encoder,
                predictor=self.predictor,
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema_encoder.update(self.encoder)
        self.ema_predictor.update(self.predictor)

        loss_value = float(loss.detach().item())
        engine.state.global_step = int(engine.state.iteration)
        engine.state.last_train_loss = loss_value
        engine.state.best_train_loss = min(
            float(engine.state.best_train_loss), loss_value
        )
        engine.state.latest_metrics = {"loss": loss_value, "lr": self._current_lr()}
        return {
            "loss": loss.detach(),
            "lr": torch.tensor(self._current_lr(), device=self.device),
        }

    def _val_step(self, engine: Engine, batch: Any) -> dict[str, torch.Tensor]:
        prepared_batch = self._prepare_batch(batch)
        self.encoder.eval()
        self.predictor.eval()

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                encoder, predictor = self._validation_models()
                loss = self._compute_loss(
                    prepared_batch,
                    encoder=encoder,
                    predictor=predictor,
                )

        return {"val_loss": loss.detach()}

    def _pretrain_step(self, engine: Engine, batch: Any) -> dict[str, torch.Tensor]:
        assert self.jepa_predictor is not None, "jepa_predictor must be set for pretraining"
        prepared_batch = self._prepare_batch(batch)
        self.encoder.train()
        self.jepa_predictor.train()

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            loss = self._compute_jepa_loss(
                prepared_batch,
                encoder=self.encoder,
                jepa_predictor=self.jepa_predictor,
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema_encoder.update(self.encoder)

        loss_value = float(loss.detach().item())
        engine.state.global_step = int(engine.state.iteration)
        engine.state.last_train_loss = loss_value
        engine.state.best_train_loss = min(
            float(engine.state.best_train_loss), loss_value
        )
        engine.state.latest_metrics = {"loss": loss_value, "lr": self._current_lr()}
        return {
            "loss": loss.detach(),
            "lr": torch.tensor(self._current_lr(), device=self.device),
        }

    def _log(
        self, scope: str, metrics: dict[str, float], *, step: Optional[int] = None
    ) -> None:
        payload = {f"{scope}/{key}": float(value) for key, value in metrics.items()}
        if self.state is not None:
            self.state.latest_metrics = payload

        if self.log_to_console:
            rendered = ", ".join(f"{key}={value:.6f}" for key, value in payload.items())
            current_step = int(getattr(self.state, "global_step", 0))
            print(f"[{scope}] step={current_step} {rendered}")

        if self.log_to_wandb and self.wandb_logger is not None:
            self.wandb_logger.log(
                payload,
                step=(
                    step
                    if step is not None
                    else int(getattr(self.state, "global_step", 0))
                ),
            )

    def _checkpoint_objects(self) -> dict[str, Any]:
        if self._state_proxy is None:
            raise RuntimeError("Trainer state proxy is not initialized.")

        objects: dict[str, Any] = {
            "encoder": self.encoder,
            "predictor": self.predictor,
            "trainer_state": self._state_proxy,
            "encoder_ema": self.ema_encoder,
            "predictor_ema": self.ema_predictor,
        }
        if self.jepa_predictor is not None:
            objects["jepa_predictor"] = self.jepa_predictor
        if self.ema_jepa is not None:
            objects["jepa_ema"] = self.ema_jepa
        if self.optimizer is not None:
            objects["optimizer"] = self.optimizer
        if self.scaler is not None:
            objects["scaler"] = self.scaler
        return objects

    def _setup_running_averages(self, trainer: Engine, evaluator: Engine) -> None:
        """Attach running average metrics to trainer and evaluator."""
        RunningAverage(output_transform=lambda output: output["loss"]).attach(
            trainer, "loss"
        )
        RunningAverage(output_transform=lambda output: output["val_loss"]).attach(
            evaluator, "val_loss"
        )

    def _setup_nan_termination(self, trainer: Engine) -> None:
        """Attach handler to terminate training on NaN values."""
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def _setup_step_logging(self, trainer: Engine) -> None:
        """Attach handler for step-level training logging."""

        @trainer.on(Events.ITERATION_COMPLETED(every=self.log_every_n_steps))
        def _log_train_step(engine: Engine) -> None:
            output = engine.state.output
            if not isinstance(output, MappingABC):
                return

            loss_value = float(output.get("loss", 0.0))
            metrics = {
                "loss": loss_value,
                "loss_avg": float(engine.state.metrics.get("loss", loss_value)),
                "lr": float(output.get("lr", 0.0)),
            }
            self._log("train/step", metrics, step=int(engine.state.iteration))

    def _setup_epoch_logging(self, trainer: Engine) -> None:
        """Attach handlers for epoch-level logging and state sync."""

        @trainer.on(Events.EPOCH_STARTED)
        def _sync_epoch_state(engine: Engine) -> None:
            self.state = engine.state
            self.state.epoch = int(engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def _log_train_summary(engine: Engine) -> None:
            train_loss = float(
                engine.state.metrics.get(
                    "loss", getattr(self.state, "last_train_loss", float("nan"))
                )
            )
            self.state.last_train_loss = train_loss
            self.state.best_train_loss = min(
                float(getattr(self.state, "best_train_loss", float("inf"))), train_loss
            )
            metrics = {
                "loss": train_loss,
                "best_loss": float(
                    getattr(self.state, "best_train_loss", float("inf"))
                ),
            }
            self._log("train/summary", metrics, step=int(engine.state.iteration))

    def _setup_validation_scheduling(self, trainer: Engine, evaluator: Engine) -> None:
        """Attach handler to run validation on epoch completion."""

        @trainer.on(Events.EPOCH_COMPLETED(every=self.validate_every_n_epochs))
        def _run_validation(_: Engine) -> None:
            evaluator.run(self.val_dataloader)

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load model weights, EMA, optimizer, and state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory with checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = next(checkpoint_path.glob("*.pt"), checkpoint_path / "checkpoint.pt")
        
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(state_dict["encoder"])
        self.predictor.load_state_dict(state_dict["predictor"])
        
        if "encoder_ema" in state_dict:
            self.ema_encoder.load_state_dict(state_dict["encoder_ema"])
        if "predictor_ema" in state_dict:
            self.ema_predictor.load_state_dict(state_dict["predictor_ema"])
        
        if "jepa_predictor" in state_dict and self.jepa_predictor is not None:
            self.jepa_predictor.load_state_dict(state_dict["jepa_predictor"])
        if "jepa_ema" in state_dict and self.ema_jepa is not None:
            self.ema_jepa.load_state_dict(state_dict["jepa_ema"])
        
        if "optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scaler" in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict["scaler"])
        
        if "trainer_state" in state_dict and self._state_proxy is not None:
            self._state_proxy.load_state_dict(state_dict["trainer_state"])

    def save_checkpoint(self, prefix: str = "latest") -> Path:
        """Save model weights, EMA, optimizer, and state to checkpoint.
        
        Args:
            prefix: Filename prefix for checkpoint (e.g., "latest", "best_val", "pretrain").
            
        Returns:
            Path to saved checkpoint file.
        """
        if self._state_proxy is None:
            raise RuntimeError("Trainer state proxy is not initialized.")
        
        checkpoint_path = self.checkpoint_dir / f"{prefix}.pt"
        state_dict: dict[str, Any] = {
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "encoder_ema": self.ema_encoder.state_dict(),
            "predictor_ema": self.ema_predictor.state_dict(),
            "trainer_state": self._state_proxy.state_dict(),
        }
        if self.jepa_predictor is not None:
            state_dict["jepa_predictor"] = self.jepa_predictor.state_dict()
        if self.ema_jepa is not None:
            state_dict["jepa_ema"] = self.ema_jepa.state_dict()
        if self.optimizer is not None:
            state_dict["optimizer"] = self.optimizer.state_dict()
        if self.scaler is not None:
            state_dict["scaler"] = self.scaler.state_dict()
        
        torch.save(state_dict, checkpoint_path)
        return checkpoint_path

    def _setup_checkpointing(self, trainer: Engine, evaluator: Engine) -> None:
        """Attach checkpoint handlers for latest and best model saving."""
        latest_checkpoint = Checkpoint(
            self._checkpoint_objects(),
            DiskSaver(self.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="latest",
            global_step_transform=global_step_from_engine(trainer),
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.save_every_n_epochs),
            latest_checkpoint,
        )

        best_checkpoint = Checkpoint(
            self._checkpoint_objects(),
            DiskSaver(self.checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=1,
            filename_prefix="best_val",
            score_function=lambda engine: -float(engine.state.metrics["val_loss"]),
            score_name="val_loss",
            global_step_transform=global_step_from_engine(trainer),
        )

        @evaluator.on(Events.COMPLETED)
        def _log_validation(engine: Engine) -> None:
            val_loss = float(
                engine.state.metrics.get(
                    "val_loss", getattr(self.state, "last_val_loss", float("nan"))
                )
            )
            self.state.last_val_loss = val_loss
            self.state.best_val_loss = min(
                float(getattr(self.state, "best_val_loss", float("inf"))), val_loss
            )
            metrics = {
                "val_loss": val_loss,
                "best_val_loss": float(
                    getattr(self.state, "best_val_loss", float("inf"))
                ),
            }
            self._log("val/summary", metrics, step=int(trainer.state.iteration))

        evaluator.add_event_handler(Events.COMPLETED, best_checkpoint)

    def _attach_handlers(self, trainer: Engine, evaluator: Engine) -> None:
        """Attach all Ignite handlers for training orchestration."""
        self._setup_running_averages(trainer, evaluator)
        self._setup_nan_termination(trainer)
        self._setup_step_logging(trainer)
        self._setup_epoch_logging(trainer)
        self._setup_validation_scheduling(trainer, evaluator)
        self._setup_checkpointing(trainer, evaluator)

    def build_train_engine(
        self,
        train_step: Optional[Any] = None,
    ) -> Engine:
        step_fn = train_step if train_step is not None else self._train_step
        trainer = Engine(step_fn)
        self._initialize_engine_state(trainer.state)
        self.state = trainer.state
        self._state_proxy = EngineStateCheckpointProxy(self.state)
        self.trainer = trainer
        return trainer

    def build_validation_engine(self) -> Engine:
        evaluator = Engine(self._val_step)
        self.evaluator = evaluator
        return evaluator

    def run(
        self,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        max_epochs: Optional[int] = None,
    ) -> Engine:
        train_loader = (
            train_dataloader if train_dataloader is not None else self.dataloader
        )
        if train_loader is None:
            raise ValueError("A training dataloader is required.")

        if val_dataloader is not None:
            self.val_dataloader = val_dataloader
        if self.val_dataloader is None:
            raise ValueError("A validation dataloader is required for Ignite startup.")

        trainer = self.build_train_engine()
        evaluator = self.build_validation_engine()
        self._attach_handlers(trainer, evaluator)

        trainer.run(train_loader, max_epochs=max_epochs or int(self.config.get_num_epochs))
        return trainer


class PretrainStrategy:
    """Strategy for JEPA-style pretraining with masked frame reconstruction."""

    def __init__(
        self,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 100,
        ema_decay: float = 0.999,
        mask_ratio: float = 0.25,
        context_loss_weight: float = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio
        self.context_loss_weight = context_loss_weight

    def configure_trainer(self, trainer: IgniteMotionTrainer) -> None:
        """Apply pretraining configuration to trainer."""
        trainer.ema_encoder.decay = self.ema_decay
        if trainer.ema_jepa is not None:
            trainer.ema_jepa.decay = self.ema_decay

    def get_checkpoint_prefix(self) -> str:
        return "pretrain_latest"


class FinetuneStrategy:
    """Strategy for finetuning phase with advanced techniques."""
    
    def __init__(
        self,
        *,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        ema_decay: float = 0.999,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.ema_decay = ema_decay
    
    def configure_trainer(self, trainer: IgniteMotionTrainer) -> None:
        """Apply finetuning configuration to trainer."""
        trainer.ema_encoder.decay = self.ema_decay
        trainer.ema_predictor.decay = self.ema_decay
    
    def get_checkpoint_prefix(self) -> str:
        return "finetune_latest"


class TwoStepMotionTrainer:
    """Orchestrates two-step training: pretraining followed by finetuning."""
    
    def __init__(
        self,
        *,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
        config: Config,
        pretrain_strategy: PretrainStrategy,
        finetune_strategy: FinetuneStrategy,
        dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        wandb_logger: Optional[WandbLogger] = None,
        checkpoint_dir: Optional[Path | str] = None,
        log_to_console: bool = True,
        log_to_wandb: bool = False,
        log_every_n_steps: int = 50,
        validate_every_n_epochs: int = 1,
        save_every_n_epochs: int = 1,
        use_ema_for_validation: bool = True,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.config = config
        self.pretrain_strategy = pretrain_strategy
        self.finetune_strategy = finetune_strategy
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.normalizer = normalizer
        self.wandb_logger = wandb_logger
        self.checkpoint_dir = Path(checkpoint_dir or config.checkpoint_dir)
        self.log_to_console = bool(log_to_console)
        self.log_to_wandb = bool(log_to_wandb and wandb_logger is not None)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.validate_every_n_epochs = max(1, int(validate_every_n_epochs))
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))
        self.use_ema_for_validation = bool(use_ema_for_validation)
        self._current_phase: Optional[TrainingPhase] = None
        
        self._trainer: Optional[IgniteMotionTrainer] = None
    
    @property
    def trainer(self) -> IgniteMotionTrainer:
        """Lazily initialize and return the underlying IgniteMotionTrainer."""
        if self._trainer is None:
            self._trainer = self._get_fresh_trainer()
        return self._trainer
    
    @property
    def pretrain_checkpoint_path(self) -> Path:
        """Return path where pretrain checkpoint will be saved."""
        return self.checkpoint_dir / f"{self.pretrain_strategy.get_checkpoint_prefix()}.pt"
    
    def run_pretrain(
        self,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        max_epochs: Optional[int] = None,
    ) -> Engine:
        """Execute pretraining phase with JEPA objective.

        Args:
            train_dataloader: Training data loader (uses self.dataloader if None).
            val_dataloader: Validation data loader (uses self.val_dataloader if None).
            max_epochs: Override number of epochs (uses strategy default if None).

        Returns:
            The Ignite Engine after pretraining completes.
        """
        self._current_phase = TrainingPhase.PRETRAINING

        jepa_predictor = JepaPredictor(self.config.encoder_config)
        jepa_predictor.to(self.trainer.device if self.trainer else torch.device(self.config.device))

        inner_trainer = IgniteMotionTrainer(
            encoder=self.encoder,
            predictor=self.predictor,
            config=self.config,
            dataloader=self.dataloader,
            val_dataloader=self.val_dataloader,
            normalizer=self.normalizer,
            wandb_logger=self.wandb_logger,
            checkpoint_dir=self.checkpoint_dir,
            log_to_console=self.log_to_console,
            log_to_wandb=self.log_to_wandb,
            log_every_n_steps=self.log_every_n_steps,
            validate_every_n_epochs=self.validate_every_n_epochs,
            save_every_n_epochs=self.save_every_n_epochs,
            use_ema_for_validation=self.use_ema_for_validation,
            jepa_predictor=jepa_predictor,
        )
        self._trainer = inner_trainer

        self.pretrain_strategy.configure_trainer(inner_trainer)

        inner_trainer.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(jepa_predictor.parameters()),
            lr=self.pretrain_strategy.learning_rate,
            weight_decay=self.pretrain_strategy.weight_decay,
        )

        def _jepa_train_step(engine: Engine, batch: Any) -> dict[str, torch.Tensor]:
            prepared_batch = inner_trainer._prepare_batch(batch)
            inner_trainer.encoder.train()
            assert inner_trainer.jepa_predictor is not None
            inner_trainer.jepa_predictor.train()

            inner_trainer.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=inner_trainer.device.type,
                dtype=inner_trainer.amp_dtype,
                enabled=inner_trainer.use_amp,
            ):
                loss = inner_trainer._compute_jepa_loss(
                    prepared_batch,
                    encoder=inner_trainer.encoder,
                    jepa_predictor=inner_trainer.jepa_predictor,
                    target_encoder=inner_trainer.ema_encoder.model,
                    mask_ratio=self.pretrain_strategy.mask_ratio,
                    context_loss_weight=self.pretrain_strategy.context_loss_weight,
                )

            inner_trainer.scaler.scale(loss).backward()
            inner_trainer.scaler.step(inner_trainer.optimizer)
            inner_trainer.scaler.update()
            inner_trainer.ema_encoder.update(inner_trainer.encoder)
            assert inner_trainer.ema_jepa is not None
            inner_trainer.ema_jepa.update(inner_trainer.jepa_predictor)

            loss_value = float(loss.detach().item())
            engine.state.global_step = int(engine.state.iteration)
            engine.state.last_train_loss = loss_value
            engine.state.best_train_loss = min(
                float(engine.state.best_train_loss), loss_value
            )
            engine.state.latest_metrics = {
                "loss": loss_value,
                "jepa_loss": loss_value,
                "lr": inner_trainer._current_lr(),
            }
            return {
                "loss": loss.detach(),
                "lr": torch.tensor(inner_trainer._current_lr(), device=inner_trainer.device),
            }

        trainer_engine = inner_trainer.build_train_engine(train_step=_jepa_train_step)
        evaluator_engine = inner_trainer.build_validation_engine()
        inner_trainer._attach_handlers(trainer_engine, evaluator_engine)

        epochs = max_epochs or self.pretrain_strategy.num_epochs
        trainer_engine.run(
            train_dataloader if train_dataloader is not None else self.dataloader,
            max_epochs=epochs,
        )

        inner_trainer.save_checkpoint(self.pretrain_strategy.get_checkpoint_prefix())
        return trainer_engine

    def _get_fresh_trainer(self) -> IgniteMotionTrainer:
        """Create a fresh IgniteMotionTrainer with initialized engines."""
        trainer = IgniteMotionTrainer(
            encoder=self.encoder,
            predictor=self.predictor,
            config=self.config,
            dataloader=self.dataloader,
            val_dataloader=self.val_dataloader,
            normalizer=self.normalizer,
            wandb_logger=self.wandb_logger,
            checkpoint_dir=self.checkpoint_dir,
            log_to_console=self.log_to_console,
            log_to_wandb=self.log_to_wandb,
            log_every_n_steps=self.log_every_n_steps,
            validate_every_n_epochs=self.validate_every_n_epochs,
            save_every_n_epochs=self.save_every_n_epochs,
            use_ema_for_validation=self.use_ema_for_validation,
        )
        trainer.build_train_engine()
        trainer.build_validation_engine()
        return trainer
    
    def run_finetune(
        self,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        max_epochs: Optional[int] = None,
        pretrain_checkpoint_path: Optional[Path | str] = None,
    ) -> Engine:
        """Execute finetuning phase, loading from pretrain checkpoint.
        
        Args:
            train_dataloader: Training data loader (uses self.dataloader if None).
            val_dataloader: Validation data loader (uses self.val_dataloader if None).
            max_epochs: Override number of epochs (uses strategy default if None).
            pretrain_checkpoint_path: Path to pretrain checkpoint (uses default if None).
            
        Returns:
            The Ignite Engine after finetuning completes.
        """
        self._current_phase = TrainingPhase.FINETUNING
        
        inner_trainer = self._get_fresh_trainer()
        self._trainer = inner_trainer
        
        checkpoint_path = pretrain_checkpoint_path or self.pretrain_checkpoint_path
        inner_trainer.load_checkpoint(checkpoint_path)
        
        self.finetune_strategy.configure_trainer(inner_trainer)
        
        inner_trainer.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.finetune_strategy.learning_rate,
            weight_decay=self.finetune_strategy.weight_decay,
        )
        
        epochs = max_epochs or self.finetune_strategy.num_epochs
        
        inner_trainer.run(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=epochs,
        )
        
        return inner_trainer.trainer


def build_ignite_trainer(
    *,
    encoder: MotionHistoryEncoder,
    predictor: FlowMatchingPredictor,
    config: Config,
    dataloader: Optional[Any] = None,
    val_dataloader: Optional[Any] = None,
    normalizer: Optional[FeatureNormalizer] = None,
    wandb_logger: Optional[WandbLogger] = None,
    checkpoint_dir: Optional[Path | str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    log_to_console: bool = True,
    log_to_wandb: bool = False,
    log_every_n_steps: int = 50,
    validate_every_n_epochs: int = 1,
    save_every_n_epochs: int = 1,
    use_ema_for_validation: bool = True,
) -> IgniteMotionTrainer:
    return IgniteMotionTrainer(
        encoder=encoder,
        predictor=predictor,
        config=config,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        normalizer=normalizer,
        wandb_logger=wandb_logger,
        checkpoint_dir=checkpoint_dir,
        optimizer=optimizer,
        scaler=scaler,
        log_to_console=log_to_console,
        log_to_wandb=log_to_wandb,
        log_every_n_steps=log_every_n_steps,
        validate_every_n_epochs=validate_every_n_epochs,
        save_every_n_epochs=save_every_n_epochs,
        use_ema_for_validation=use_ema_for_validation,
    )


__all__ = [
    "BatchDict",
    "EMAModel",
    "EngineStateCheckpointProxy",
    "IgniteMotionTrainer",
    "TrainingPhase",
    "TrainingStrategy",
    "PretrainStrategy",
    "FinetuneStrategy",
    "TwoStepMotionTrainer",
    "build_ignite_trainer",
]
