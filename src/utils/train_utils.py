"""
Training utilities for Motion History Encoder and Flow Matching Predictor.

Redesigned training mechanism:

- Single-step flow matching:
  - Predict ONLY the next frame given a variable-length motion history.
- Progressive AR horizon curriculum:
  - Curriculum controls max history length (in frames).
  - Per-batch, sample history length H ∈ [1, curr_horizon].
- Standard flow matching objective:
  - Velocity prediction in a reduced 72D feature space.
- Light CFG support:
  - Optional low-probability dropout of text conditioning.
- EMA for validation and checkpointing.
- Optional AR-style validation that approximates autoregressive rollout.
"""

import copy
import os
import time
from typing import Optional, Tuple, Dict, Any, Union, TypeVar, Generic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from config import Config
from models import FlowMatchingPredictor, MotionHistoryEncoder
from utils.dataset import Text2MotionDataset
from utils.text_encoder import CLIPEncoder
from utils.wandb_logger import WandbLogger
from utils.motion_utils import (
    FeatureNormalizer,
    RootPositionTracker,  # kept in case you use it elsewhere
    flow_output_to_271d,  # kept in case you use it elsewhere
    generated_positions_to_271d,
    extract_prev_frame_features,
)

# Type variable for generic EMA model typing
T = TypeVar("T", bound=nn.Module)

# =============================================================================
# EMA Model Wrapper
# =============================================================================


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

    def to(self, device: str) -> "EMAModel[T]":
        """Move EMA model to device."""
        self.model.to(device)
        return self


class Trainer:
    """Class-based trainer that holds global training dependencies and config."""

    def __init__(
        self,
        encoder: MotionHistoryEncoder,
        predictor: FlowMatchingPredictor,
        dataloader: DataLoader,
        config: "Config",
        clip_encoder: Optional[CLIPEncoder] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        resume_from: Optional[str] = None,
        normalizer: Optional["FeatureNormalizer"] = None,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.dataloader = dataloader
        self.config = config
        self.clip_encoder = clip_encoder
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.resume_from = resume_from
        self.normalizer = normalizer
        self.val_dataloader = val_dataloader

    @staticmethod
    def extract_clean_target(frame: torch.Tensor) -> torch.Tensor:
        """Extract per-joint 3D track targets from a 271D frame."""
        return frame[..., 3:69].reshape(*frame.shape[:-1], 22, 3)

    @staticmethod
    def compute_global_relative_shifts(tracks: torch.Tensor) -> torch.Tensor:
        """Compute root-relative shifts in track space."""
        root_track = tracks[:, :1, :]
        return tracks - root_track

    @classmethod
    def _build_models_from_config(
        cls,
        config: Config,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> Tuple[MotionHistoryEncoder, FlowMatchingPredictor]:
        """Create encoder and predictor from config for classmethod-based training."""
        encoder = MotionHistoryEncoder(
            frame_feature_dim=config.encoder_motion_dim,
            text_embedding_dim=config.encoder_text_dim,
            text_proj_dim=config.encoder_text_proj_dim,
            model_dim=config.encoder_hidden_dim,
            per_joint_out_dim=config.encoder_per_joint_dim,
            num_layers=config.encoder_num_layers,
            joint_count=config.encoder_num_joints,
            text_scale=config.encoder_text_scale,
            dropout=config.encoder_dropout,
            normalizer=normalizer,
        )
        predictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
            out_channels=config.predictor_config.track_dimensionality,
            use_relative_shift=True,
        )
        return encoder, predictor

    @classmethod
    def train(
        cls,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        clip_encoder: Optional[CLIPEncoder] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        resume_from: Optional[str] = None,
        encoder_override: Optional[MotionHistoryEncoder] = None,
        predictor_override: Optional[FlowMatchingPredictor] = None,
    ) -> Tuple[EMAModel[MotionHistoryEncoder], EMAModel[FlowMatchingPredictor]]:
        """Canonical training entrypoint parameterized by config and dataloaders."""
        encoder, predictor = (
            encoder_override,
            predictor_override,
        )
        if encoder is None or predictor is None:
            encoder, predictor = cls._build_models_from_config(config, normalizer)

        trainer = cls(
            encoder=encoder,
            predictor=predictor,
            dataloader=train_dataloader,
            config=config,
            clip_encoder=clip_encoder,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            resume_from=resume_from,
            normalizer=normalizer if normalizer is not None else encoder.normalizer,
            val_dataloader=val_dataloader,
        )
        return trainer._run_training()

    def setup_training_environment(
        self,
    ) -> Tuple[
        Any,
        Any,
        Optional[WandbLogger],
        EMAModel[MotionHistoryEncoder],
        EMAModel[FlowMatchingPredictor],
        Optimizer,
        GradScaler,
        int,
        Dict[str, Any],
        str,
        bool,
    ]:
        device = self.config.device
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        ema_decay = self.config.ema_decay
        horizon = self.config.horizon
        curriculum = self.config.curriculum
        cfg_dropout = self.config.cfg_dropout
        num_epochs = self.config.num_epochs

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.encoder.to(device)
        self.predictor.to(device)

        wandb_logger = None
        if self.wandb_project:
            wandb_config = {
                "lr": lr,
                "weight_decay": weight_decay,
                "ema_decay": ema_decay,
                "num_epochs": num_epochs,
                "horizon": horizon,
                "cfg_dropout": cfg_dropout,
                "batch_size": self.dataloader.batch_size,
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "predictor_params": sum(p.numel() for p in self.predictor.parameters()),
                "curriculum": curriculum,
            }
            wandb_logger = WandbLogger(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
            )

        encoder_ema = EMAModel(self.encoder, decay=ema_decay).to(device)
        predictor_ema = EMAModel(self.predictor, decay=ema_decay).to(device)
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # type: ignore[arg-type]

        device_str = str(device)
        use_amp = device_str.startswith("cuda")
        scaler = GradScaler("cuda", enabled=use_amp)

        training_state: Dict[str, Any] = {
            "global_step": 0,
            "best_loss": float("inf"),
            "best_epoch": -1,
            "best_val_loss": float("inf"),
            "best_val_epoch": -1,
        }
        start_epoch = 0

        if self.resume_from is not None and os.path.exists(self.resume_from):
            print(f"Resuming from checkpoint: {self.resume_from}")
            checkpoint = torch.load(
                self.resume_from, map_location=device, weights_only=False
            )
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.predictor.load_state_dict(checkpoint["predictor"])
            encoder_ema.model.load_state_dict(checkpoint["encoder_ema"])
            predictor_ema.model.load_state_dict(checkpoint["predictor_ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            training_state["global_step"] = checkpoint.get("global_step", 0)
            training_state["best_loss"] = checkpoint.get("best_loss", float("inf"))
            training_state["best_epoch"] = checkpoint.get("best_epoch", -1)
            training_state["best_val_loss"] = checkpoint.get(
                "best_val_loss", float("inf")
            )
            training_state["best_val_epoch"] = checkpoint.get("best_val_epoch", -1)
            if "current_horizon" in checkpoint:
                training_state["current_horizon"] = checkpoint["current_horizon"]
            print(
                f"Resumed from epoch {start_epoch}, "
                f"step {training_state['global_step']}"
            )

        if curriculum is not None and len(curriculum) > 0:
            print(f"Training for {num_epochs} epochs with curriculum: {curriculum}")
        else:
            print(f"Training for {num_epochs} epochs with fixed horizon={horizon}")
        print(f"Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(
            f"Predictor params: {sum(p.numel() for p in self.predictor.parameters()):,}"
        )

        self.encoder.train()
        self.predictor.train()

        return (
            device,
            str(self.config.checkpoint_dir),
            wandb_logger,
            encoder_ema,
            predictor_ema,
            optimizer,
            scaler,
            start_epoch,
            training_state,
            device_str,
            use_amp,
        )

    def setup_curriculum_state(
        self,
        checkpoint_state: Optional[dict] = None,
    ) -> dict:
        curriculum = self.config.curriculum
        use_curriculum = curriculum is not None and len(curriculum) > 0

        if checkpoint_state and "current_horizon" in checkpoint_state:
            current_horizon = checkpoint_state["current_horizon"]
            max_horizon = checkpoint_state["max_horizon"]
        elif use_curriculum and curriculum:
            current_horizon = curriculum[0]["horizon"]
            max_horizon = curriculum[-1]["horizon"]
        else:
            current_horizon = self.config.horizon
            max_horizon = self.config.horizon

        return {
            "use_curriculum": use_curriculum,
            "current_horizon": current_horizon,
            "max_horizon": max_horizon,
        }

    def save_training_checkpoint(
        self,
        save_dir: str,
        filename: str,
        encoder_ema: EMAModel[MotionHistoryEncoder],
        predictor_ema: EMAModel[FlowMatchingPredictor],
        optimizer: Optimizer,
        scaler: GradScaler,
        epoch: int,
        global_step: int,
        loss: float,
        curriculum_state: dict,
        training_state: dict,
    ) -> None:
        path = os.path.join(save_dir, filename)
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "encoder_ema": encoder_ema.model.state_dict(),
            "predictor_ema": predictor_ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "horizon": curriculum_state["max_horizon"],
            "current_horizon": curriculum_state["current_horizon"],
            "use_curriculum": curriculum_state["use_curriculum"],
            "best_loss": training_state["best_loss"],
            "best_epoch": training_state["best_epoch"],
            "best_val_loss": training_state["best_val_loss"],
            "best_val_epoch": training_state["best_val_epoch"],
        }
        checkpoint["config"] = self.config
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def unpack_batch(
        self,
        batch: dict,
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        motion_raw = batch["motion"].to(device)
        joints = batch["joints"].to(device)
        B, T, _ = motion_raw.shape
        lengths = batch.get(
            "lengths", torch.full((B,), T, device=device, dtype=torch.long)
        )

        motion = (
            self.normalizer.normalize(motion_raw)
            if self.normalizer is not None
            else motion_raw
        )

        if joints.shape[0] != B or joints.shape[1] != T:
            raise ValueError(
                f"Batch shape mismatch between motion {motion.shape} and joints {joints.shape}."
            )

        if "text_clip" in batch:
            text = batch["text_clip"].to(device)
        elif "captions" in batch and self.clip_encoder is not None:
            captions = batch["captions"]
            with torch.no_grad():
                text = self.clip_encoder(captions)
        elif "captions" in batch:
            raise ValueError(
                "Raw captions provided but no clip_encoder. "
                "Pass clip_encoder to Trainer() or provide pre-encoded 'text_clip'."
            )
        else:
            raise ValueError(
                "No text input found. Batch must contain 'text_clip' or 'captions'."
            )

        if text.ndim != 3 or text.shape[0] != B or text.shape[1] != 1:
            raise ValueError(
                f"Invalid text embedding shape {tuple(text.shape)}. Expected (B, 1, D) with B={B}."
            )

        return motion, joints, text, lengths, B, T

    def sample_next_frame_window(
        self,
        motion: torch.Tensor,
        joints: torch.Tensor,
        curr_horizon: int,
    ):
        _, T, _ = motion.shape
        assert curr_horizon <= T - 1, f"curr_horizon {curr_horizon} > T-1 {T-1}"
        if joints.shape[:2] != motion.shape[:2]:
            raise ValueError(
                f"sample_next_frame_window got motion shape {motion.shape} and joints shape {joints.shape}."
            )
        relative_shifts = joints[:, 1:] - joints[:, :-1]
        return motion[:, 1:], joints[:, 1:], relative_shifts, curr_horizon

    def get_rollout_probability(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.config.rollout_prob_end
        progress = float(epoch) / float(total_epochs - 1)
        progress = max(0.0, min(1.0, progress))
        return self.config.rollout_prob_start + (
            (self.config.rollout_prob_end - self.config.rollout_prob_start) * progress
        )

    def get_rollout_mask(
        self,
        batch_size: int,
        rollout_prob: float,
        device: Union[str, torch.device],
        stochastic: bool,
    ) -> torch.Tensor:
        p = float(max(0.0, min(1.0, rollout_prob)))
        if p == 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if p == 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if stochastic:
            return torch.rand(batch_size, device=device) < p
        threshold = max(1, int(round(p * batch_size)))
        return torch.arange(batch_size, device=device) < threshold

    def incremental_flow_loss(
        self,
        motion: torch.Tensor,
        joints: torch.Tensor,
        relative_shifts: torch.Tensor,
        text_for_encoder: torch.Tensor,
        epoch: int,
        total_epochs: int,
        device: Union[str, torch.device],
        stochastic_rollout: bool,
        encoder: Optional[MotionHistoryEncoder] = None,
        predictor: Optional[FlowMatchingPredictor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        enc = encoder if encoder is not None else self.encoder
        pred_model = predictor if predictor is not None else self.predictor

        if not hasattr(enc, "gru_step"):
            raise AttributeError(
                "Encoder must expose a gru_step(x_t, text_emb, h) method."
            )

        B, _, _ = motion.shape
        _, j_len, _, _ = joints.shape
        if j_len <= 1:
            raise ValueError("No prediction steps found in target window.")
        if motion.shape[1] != j_len:
            raise ValueError(
                f"motion length {motion.shape[1]} != joints length {j_len}."
            )

        pred_steps = j_len - 1
        hist = motion[:, :-1]
        hist_joints = joints[:, :-1]
        prev_relative_shifts = relative_shifts[:, :-1]
        target_motion = motion[:, 1:]
        target_joints = joints[:, 1:]
        target_relative_shifts = relative_shifts[:, 1:]

        context, h_state = enc.gru_step(hist[:, 0], text_for_encoder, h=None)
        if context is None:
            raise RuntimeError("Encoder context was not initialized from history.")

        flow_losses: list[torch.Tensor] = []
        consistency_losses: list[torch.Tensor] = []
        rollout_prob = self.get_rollout_probability(
            epoch=epoch, total_epochs=total_epochs
        )
        ode_steps = max(1, int(self.config.rollout_integration_steps))
        dt = 1.0 / float(ode_steps)
        current_positions = hist_joints[:, 0].detach().clone()

        for step_idx in range(pred_steps):
            x1 = target_relative_shifts[:, step_idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            t_ = t.view(B, 1, 1)
            xt = t_ * x1 + (1 - t_) * x0
            relative_shifts_step = prev_relative_shifts[:, step_idx]

            pred, _, _ = pred_model.forward(
                track_features=context,
                noised_tracks=xt,
                timesteps=t,
                relative_shifts=relative_shifts_step,
                text_embedding=text_for_encoder,
                output_attentions=False,
                output_hidden_states=False,
            )
            flow_losses.append(F.mse_loss(pred, x1 - x0))

            rollout_mask = self.get_rollout_mask(
                B, rollout_prob, device, stochastic_rollout
            )
            next_input = target_motion[:, step_idx].clone()
            if rollout_mask.any():
                predictor_was_training = pred_model.training
                pred_model.eval()
                x_t_roll = torch.randn_like(x1)
                pred_roll_endpoint: Optional[torch.Tensor] = None
                try:
                    with torch.no_grad():
                        for ode_step in range(ode_steps):
                            tau = torch.full((B,), float(ode_step) * dt, device=device)
                            pred_roll = pred_model.forward(
                                track_features=context,
                                noised_tracks=x_t_roll,
                                timesteps=tau,
                                relative_shifts=x_t_roll,
                                text_embedding=text_for_encoder,
                                output_attentions=False,
                                output_hidden_states=False,
                            )[0]
                            x_t_roll = x_t_roll + pred_roll * dt

                    if self.config.use_consistency_loss:
                        rolled_count = int(rollout_mask.sum().item())
                        tau_endpoint = torch.full(
                            (rolled_count,), float(ode_steps - 1) * dt, device=device
                        )
                        pred_roll_endpoint = pred_model.forward(
                            track_features=context[rollout_mask],
                            noised_tracks=x_t_roll[rollout_mask],
                            timesteps=tau_endpoint,
                            relative_shifts=x_t_roll[rollout_mask],
                            text_embedding=text_for_encoder[rollout_mask],
                            output_attentions=False,
                            output_hidden_states=False,
                        )[0]
                finally:
                    if predictor_was_training:
                        pred_model.train()

                pred_positions_roll = current_positions + x_t_roll
                rollout_frame, _ = generated_positions_to_271d(
                    new_positions=pred_positions_roll,
                    prev_positions=current_positions,
                    dataset_type="t2m",
                    use_fk_for_ric=False,
                    normalizer=self.normalizer,
                )
                next_input[rollout_mask] = rollout_frame[rollout_mask]

                if self.config.use_consistency_loss and pred_roll_endpoint is not None:
                    pred_positions_endpoint = (
                        current_positions[rollout_mask] + pred_roll_endpoint
                    )
                    consistency_losses.append(
                        F.mse_loss(
                            pred_positions_endpoint,
                            target_joints[:, step_idx][rollout_mask],
                        )
                    )

                next_positions = target_joints[:, step_idx].clone()
                next_positions[rollout_mask] = pred_positions_roll[rollout_mask]
                current_positions = next_positions.detach()
            else:
                current_positions = target_joints[:, step_idx].detach()

            context, h_state = enc.gru_step(
                next_input.detach(), text_for_encoder, h_state
            )

        flow_loss = torch.stack(flow_losses).mean()
        consistency_loss = (
            torch.stack(consistency_losses).mean()
            if consistency_losses
            else flow_loss.new_zeros(())
        )
        total_loss = flow_loss + (0.2 * consistency_loss)
        return total_loss, flow_loss, consistency_loss, pred_steps, rollout_prob

    def apply_cfg_dropout(
        self,
        text: torch.Tensor,
        device: Union[str, torch.device],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if self.config.cfg_dropout <= 0.0:
            return text
        if torch.rand(1).item() > self.config.cfg_dropout:
            return text
        return None

    def prepare_text_for_encoder(
        self,
        text_input: Optional[torch.Tensor],
        device: Union[str, torch.device],
        batch_size: int,
    ) -> torch.Tensor:
        if text_input is None:
            return torch.zeros(batch_size, self.config.encoder_text_dim, device=device)
        if text_input.ndim != 3:
            raise ValueError(
                f"Invalid text input rank {text_input.ndim}. Expected rank 3 with shape (B, 1, {self.config.encoder_text_dim})."
            )
        if text_input.shape[0] != batch_size:
            raise ValueError(
                f"Invalid text batch size {text_input.shape[0]}. Expected {batch_size}."
            )
        if (
            text_input.shape[1] != 1
            or text_input.shape[2] != self.config.encoder_text_dim
        ):
            raise ValueError(
                f"Invalid text input shape {tuple(text_input.shape)}. Expected (B, 1, {self.config.encoder_text_dim})."
            )
        return text_input[:, 0, :]

    def log_batch_metrics(
        self,
        wandb_logger: Optional[WandbLogger],
        loss: torch.Tensor,
        lr: float,
        epoch: int,
        grad_norm: torch.Tensor,
        batch_time: float,
        B: int,
        current_horizon: int,
        effective_horizon: int,
        num_pred_frames: int,
        loss_components: dict,
        global_step: int,
    ) -> None:
        if wandb_logger is None:
            return
        log_dict = {}
        for key, value in loss_components.items():
            log_dict[f"train/loss_{key}"] = (
                value.item() if hasattr(value, "item") else value
            )
        log_dict.update(
            {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/epoch": epoch,
                "train/grad_norm": (
                    grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                ),
                "train/batch_time": batch_time,
                "train/samples_per_sec": (B / batch_time if batch_time > 0 else 0),
                "train/current_horizon": current_horizon,
                "train/effective_horizon": effective_horizon,
                "train/num_pred_frames": num_pred_frames,
            }
        )
        wandb_logger.log(log_dict, step=global_step)

    def log_epoch_metrics(
        self,
        wandb_logger: Optional[WandbLogger],
        avg_epoch_loss: float,
        epoch: int,
        current_horizon: int,
        val_metrics: dict,
        global_step: int,
    ) -> None:
        if wandb_logger is None:
            return
        log_dict = {
            "epoch/avg_loss": avg_epoch_loss,
            "epoch/num": epoch,
            "epoch/current_horizon": current_horizon,
        }
        if val_metrics and "val_loss" in val_metrics:
            log_dict["epoch/val_loss"] = val_metrics["val_loss"]
        wandb_logger.log(log_dict, step=global_step)

    def handle_checkpointing(
        self,
        save_dir: str,
        encoder_ema: EMAModel[MotionHistoryEncoder],
        predictor_ema: EMAModel[FlowMatchingPredictor],
        optimizer: Optimizer,
        scaler: GradScaler,
        epoch: int,
        global_step: int,
        avg_epoch_loss: float,
        curriculum_state: dict,
        training_state: dict,
        val_metrics: dict,
    ) -> dict:
        self.save_training_checkpoint(
            save_dir=save_dir,
            filename="latest.pt",
            encoder_ema=encoder_ema,
            predictor_ema=predictor_ema,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            loss=avg_epoch_loss,
            curriculum_state=curriculum_state,
            training_state=training_state,
        )

        if avg_epoch_loss < training_state["best_loss"]:
            tqdm.write(
                f"New best model! (Loss: {training_state['best_loss']:.6f} -> {avg_epoch_loss:.6f})"
            )
            training_state["best_loss"] = avg_epoch_loss
            training_state["best_epoch"] = epoch
            self.save_training_checkpoint(
                save_dir=save_dir,
                filename="best.pt",
                encoder_ema=encoder_ema,
                predictor_ema=predictor_ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                loss=avg_epoch_loss,
                curriculum_state=curriculum_state,
                training_state=training_state,
            )

        if self.config.save_best_val and val_metrics and "val_loss" in val_metrics:
            val_loss = val_metrics["val_loss"]
            if val_loss < training_state["best_val_loss"]:
                tqdm.write(
                    "New best validation model! "
                    f"(Val Loss: {training_state['best_val_loss']:.6f} -> {val_loss:.6f})"
                )
                training_state["best_val_loss"] = val_loss
                training_state["best_val_epoch"] = epoch
                self.save_training_checkpoint(
                    save_dir=save_dir,
                    filename="best_val.pt",
                    encoder_ema=encoder_ema,
                    predictor_ema=predictor_ema,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    loss=avg_epoch_loss,
                    curriculum_state=curriculum_state,
                    training_state=training_state,
                )

        return training_state

    def validate(
        self,
        encoder: Optional[MotionHistoryEncoder] = None,
        predictor: Optional[FlowMatchingPredictor] = None,
        epoch: int = 0,
        total_epochs: Optional[int] = None,
        horizon: Optional[int] = None,
        num_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> dict:
        if self.val_dataloader is None:
            return {}

        if total_epochs is None:
            total_epochs = self.config.num_epochs
        if horizon is None:
            horizon = self.config.horizon
        if num_batches is None:
            num_batches = self.config.val_batches
        device_value: Union[str, torch.device] = (
            self.config.device if device is None else device
        )
        enc = encoder if encoder is not None else self.encoder
        pred = predictor if predictor is not None else self.predictor
        enc.eval()
        pred.eval()

        total_total_loss = 0.0
        total_flow_loss = 0.0
        total_consistency_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if num_batches > 0 and i >= num_batches:
                    break

                motion, joints, text, _lengths, B, _T = self.unpack_batch(
                    batch=batch, device=device_value
                )
                motion, joints, relative_shifts, _ = self.sample_next_frame_window(
                    motion=motion,
                    joints=joints,
                    curr_horizon=horizon,
                )
                text_for_encoder = self.prepare_text_for_encoder(text, device_value, B)

                total_loss, flow_loss, consistency_loss, _, _ = (
                    self.incremental_flow_loss(
                        motion=motion,
                        joints=joints,
                        relative_shifts=relative_shifts,
                        text_for_encoder=text_for_encoder,
                        epoch=epoch,
                        total_epochs=total_epochs,
                        device=device_value,
                        stochastic_rollout=False,
                        encoder=enc,
                        predictor=pred,
                    )
                )

                total_total_loss += total_loss.item() * B
                total_flow_loss += flow_loss.item() * B
                total_consistency_loss += consistency_loss.item() * B
                total_samples += B

        enc.train()
        pred.train()
        return {
            "val_loss": total_total_loss / max(1, total_samples),
            "val_flow_loss": total_flow_loss / max(1, total_samples),
            "val_consistency_loss": total_consistency_loss / max(1, total_samples),
        }

    def _run_training(
        self,
    ) -> Tuple[EMAModel[MotionHistoryEncoder], EMAModel[FlowMatchingPredictor]]:
        """Run training with class-owned config/state and helper methods."""
        (
            device,
            checkpoint_dir,
            wandb_logger,
            encoder_ema,
            predictor_ema,
            optimizer,
            scaler,
            start_epoch,
            training_state,
            device_str,
            use_amp,
        ) = self.setup_training_environment()

        curriculum_state = self.setup_curriculum_state(
            checkpoint_state=(
                training_state if "current_horizon" in training_state else None
            ),
        )

        num_epochs = self.config.num_epochs
        lr = self.config.learning_rate
        max_grad_norm = self.config.gradient_clip
        val_interval = self.config.val_interval
        val_batches = self.config.val_batches
        val_use_ema = self.config.val_use_ema

        amp_dtype = torch.float32
        if use_amp:
            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        epoch = start_epoch - 1
        try:
            training_state = training_state
            for epoch in tqdm(
                range(start_epoch, num_epochs), desc="Training", unit="epoch"
            ):
                prev_horizon = curriculum_state["current_horizon"]
                if curriculum_state["use_curriculum"] and self.config.curriculum:
                    for level in reversed(self.config.curriculum):
                        if epoch <= level["epochs"]:
                            curriculum_state["current_horizon"] = level["horizon"]
                        else:
                            break
                    if curriculum_state["current_horizon"] != prev_horizon:
                        tqdm.write(
                            "Curriculum update: "
                            f"horizon {prev_horizon} -> {curriculum_state['current_horizon']}"
                        )

                epoch_loss = 0.0
                num_batches = 0
                pred_horizon = 1

                self.dataloader.dataset.set_horizon(1 + curriculum_state["current_horizon"] + pred_horizon)  # type: ignore

                pbar = tqdm(
                    self.dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch"
                )
                batch_start_time = time.time()

                for batch in pbar:
                    motion, joints, text, lengths, B, T = self.unpack_batch(
                        batch=batch,
                        device=device,
                    )

                    motion, joints, relative_shifts, effective_horizon = (
                        self.sample_next_frame_window(
                            motion=motion,
                            joints=joints,
                            curr_horizon=curriculum_state["current_horizon"],
                        )
                    )

                    text_input = self.apply_cfg_dropout(text, device, B)
                    text_for_encoder = self.prepare_text_for_encoder(
                        text_input, device, B
                    )

                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_str, dtype=amp_dtype, enabled=use_amp):  # type: ignore
                        (
                            loss,
                            flow_loss,
                            consistency_loss,
                            pred_horizon,
                            rollout_prob,
                        ) = self.incremental_flow_loss(
                            motion=motion,
                            joints=joints,
                            relative_shifts=relative_shifts,
                            text_for_encoder=text_for_encoder,
                            epoch=epoch,
                            total_epochs=num_epochs,
                            device=device,
                            stochastic_rollout=True,
                        )

                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.predictor.parameters()),
                        max_grad_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    encoder_ema.update(self.encoder)
                    predictor_ema.update(self.predictor)

                    batch_time = time.time() - batch_start_time
                    batch_start_time = time.time()

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "L_flow": f"{flow_loss.item():.4f}",
                            "L_cons": f"{consistency_loss.item():.4f}",
                            "p_roll": f"{rollout_prob:.2f}",
                            "lr": f"{lr:.2e}",
                        }
                    )

                    self.log_batch_metrics(
                        wandb_logger=wandb_logger,
                        loss=loss,
                        lr=lr,
                        epoch=epoch,
                        grad_norm=grad_norm,
                        batch_time=batch_time,
                        B=B,
                        current_horizon=curriculum_state["current_horizon"],
                        effective_horizon=effective_horizon,
                        num_pred_frames=pred_horizon,
                        loss_components={
                            "L_total": loss.item(),
                            "L_flow": flow_loss.item(),
                            "L_consistency": consistency_loss.item(),
                            "rollout_prob": rollout_prob,
                        },
                        global_step=training_state["global_step"],
                    )

                    if training_state["global_step"] % 100 == 0:
                        tqdm.write(
                            f"[Epoch {epoch}] [Step {training_state['global_step']}] "
                            f"loss={loss.item():.6f} lr={lr:.2e}"
                        )

                    epoch_loss += loss.item()
                    num_batches += 1
                    training_state["global_step"] += 1

                pbar.close()
                avg_epoch_loss = epoch_loss / max(1, num_batches)
                tqdm.write(f"==> End of Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}")

                val_metrics: dict = {}
                if self.val_dataloader is not None and (epoch + 1) % val_interval == 0:
                    tqdm.write("Running validation...")

                    if val_use_ema:
                        val_encoder: MotionHistoryEncoder = encoder_ema.model
                        val_predictor: FlowMatchingPredictor = predictor_ema.model
                    else:
                        val_encoder = self.encoder
                        val_predictor = self.predictor

                    val_metrics = self.validate(
                        encoder=val_encoder,
                        predictor=val_predictor,
                        horizon=curriculum_state["current_horizon"],
                        num_batches=val_batches,
                        device=device,
                        epoch=epoch,
                        total_epochs=num_epochs,
                    )

                    val_loss = val_metrics["val_loss"]
                    tqdm.write(f"Validation loss: {val_loss:.6f}")

                    if wandb_logger is not None:
                        wandb_logger.log(
                            {
                                "val/loss": val_loss,
                                "val/epoch": epoch,
                            },
                            step=training_state["global_step"],
                        )

                self.log_epoch_metrics(
                    wandb_logger=wandb_logger,
                    avg_epoch_loss=avg_epoch_loss,
                    epoch=epoch,
                    current_horizon=curriculum_state["current_horizon"],
                    val_metrics=val_metrics,
                    global_step=training_state["global_step"],
                )

                training_state = self.handle_checkpointing(
                    save_dir=checkpoint_dir,
                    encoder_ema=encoder_ema,
                    predictor_ema=predictor_ema,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=training_state["global_step"],
                    avg_epoch_loss=avg_epoch_loss,
                    curriculum_state=curriculum_state,
                    training_state=training_state,
                    val_metrics=val_metrics,
                )

        except KeyboardInterrupt:
            tqdm.write("Training interrupted. Saving emergency checkpoint...")
            self.save_training_checkpoint(
                save_dir=checkpoint_dir,
                filename="latest_interrupted.pt",
                encoder_ema=encoder_ema,
                predictor_ema=predictor_ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=training_state["global_step"],
                loss=0.0,
                curriculum_state=curriculum_state,
                training_state=training_state,
            )
            tqdm.write("Done.")

        if wandb_logger is not None:
            summary = {
                "best_loss": training_state["best_loss"],
                "best_epoch": training_state["best_epoch"],
                "best_val_loss": training_state["best_val_loss"],
                "best_val_epoch": training_state["best_val_epoch"],
            }
            if curriculum_state["use_curriculum"]:
                summary["final_horizon"] = curriculum_state["current_horizon"]
                summary["max_horizon"] = curriculum_state["max_horizon"]
            wandb_logger.log_summary(summary)
            wandb_logger.finish()

        return encoder_ema, predictor_ema
