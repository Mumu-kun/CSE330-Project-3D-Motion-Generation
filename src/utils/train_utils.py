"""
Training utilities for Motion History Encoder and Flow Matching Predictor.

Redesigned training mechanism:

- Single-step flow matching:
  - Predict ONLY the next frame given a variable-length motion history.
- Progressive AR horizon curriculum:
  - Curriculum controls max history length (in frames).
  - Per-batch, sample history length H ∈ [1, curr_horizon].
- Standard flow matching objective:
  - Velocity prediction in a reduced 68D feature space.
- Light CFG support:
  - Optional low-probability dropout of text conditioning.
- EMA for validation and checkpointing.
- Optional AR-style validation that approximates autoregressive rollout.
"""

import copy
import csv
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, TypeVar, Generic

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from config import Config
from models import FlowMatchingPredictor, MotionHistoryEncoder, integrate_flow_ode
from utils.text_encoder import CLIPEncoder
from utils.wandb_logger import WandbLogger
from utils.motion_utils import (
    FeatureNormalizer,
    flow_output_to_positions,
    generated_positions_to_271d,
    extract_prev_frame_features,
    subset_271d_to_72d,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Type variable for generic EMA model typing
T = TypeVar("T", bound=nn.Module)

LOSS_VS_T_NUM_BINS = 100
LOSS_VS_T_SCATTER_MAX_POINTS = 5000

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

    def to(self, device: Union[str, torch.device]) -> "EMAModel[T]":
        """Move EMA model to device."""
        self.model.to(device)
        return self


class TimingStats:
    """Aggregates timing statistics across training steps."""

    def __init__(self):
        self.timings: Dict[str, list[float]] = {}
        self.step_count = 0

    def record(self, key: str, elapsed_ms: float) -> None:
        """Record a timing measurement in milliseconds."""
        if key not in self.timings:
            self.timings[key] = []
        self.timings[key].append(elapsed_ms)

    def get_averages(self) -> Dict[str, float]:
        """Get average timing for each key in milliseconds."""
        return {
            key: sum(times) / len(times) for key, times in self.timings.items() if times
        }

    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()
        self.step_count = 0

    def __str__(self) -> str:
        """Pretty print timing summary."""
        averages = self.get_averages()
        lines = ["=== Timing Summary ==="]
        for key in sorted(averages.keys()):
            lines.append(f"  {key}: {averages[key]:.2f}ms")
        total = sum(averages.values())
        lines.append(f"  Total: {total:.2f}ms")
        return "\n".join(lines)


@contextmanager
def timer(stats: Optional[TimingStats], key: str):
    """Context manager for timing operations with optional recording."""
    if stats is None:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats.record(key, elapsed_ms)


def compute_per_sample_flow_loss(
    pred: torch.Tensor,
    target_flow: torch.Tensor,
) -> torch.Tensor:
    """Compute one mean-MSE flow loss value per flattened training sample."""
    if pred.shape != target_flow.shape:
        raise ValueError(
            "Per-sample flow loss requires matching shapes, got "
            f"{tuple(pred.shape)} and {tuple(target_flow.shape)}"
        )
    if pred.ndim < 2:
        raise ValueError(
            "Per-sample flow loss expects at least 2 dimensions, got "
            f"{tuple(pred.shape)}"
        )
    return F.mse_loss(pred, target_flow, reduction="none").mean(dim=-1)


def aggregate_loss_vs_t_bins(
    t_values: torch.Tensor,
    per_sample_flow_loss: torch.Tensor,
    num_bins: int = LOSS_VS_T_NUM_BINS,
) -> Dict[str, torch.Tensor]:
    """Aggregate per-sample flow loss into fixed bins over t in [0, 1]."""
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if t_values.ndim != 1 or per_sample_flow_loss.ndim != 1:
        raise ValueError(
            "Loss-vs-t aggregation expects 1D tensors, got "
            f"{tuple(t_values.shape)} and {tuple(per_sample_flow_loss.shape)}"
        )
    if t_values.shape != per_sample_flow_loss.shape:
        raise ValueError(
            "Loss-vs-t aggregation expects matching tensor lengths, got "
            f"{tuple(t_values.shape)} and {tuple(per_sample_flow_loss.shape)}"
        )

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, dtype=torch.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    counts = torch.zeros(num_bins, dtype=torch.long)
    mean_losses = torch.full((num_bins,), float("nan"), dtype=torch.float64)

    if t_values.numel() == 0:
        return {
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "counts": counts,
            "mean_flow_loss": mean_losses,
        }

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


def _append_csv_rows(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[Dict[str, Any]],
) -> None:
    """Append rows to a CSV file, creating the header on first write."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _plot_loss_vs_t_curve(
    plot_path: Path,
    epoch: int,
    t_values: torch.Tensor,
    per_sample_flow_loss: torch.Tensor,
    aggregated: Dict[str, torch.Tensor],
) -> None:
    """Render a high-resolution loss-vs-t plot for one epoch."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    t_cpu = t_values.detach().to(dtype=torch.float64, device="cpu")
    loss_cpu = per_sample_flow_loss.detach().to(dtype=torch.float64, device="cpu")
    centers = aggregated["bin_centers"].cpu().numpy()
    mean_losses = aggregated["mean_flow_loss"].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    if t_cpu.numel() > 0 and t_cpu.numel() <= LOSS_VS_T_SCATTER_MAX_POINTS:
        ax.scatter(
            t_cpu.numpy(),
            loss_cpu.numpy(),
            s=8,
            alpha=0.12,
            color="tab:gray",
            linewidths=0,
        )
    ax.plot(
        centers,
        mean_losses,
        color="tab:blue",
        linewidth=2,
        marker="o",
        markersize=2,
    )
    ax.set_title(f"Flow Loss vs Noise Level t (Epoch {epoch})")
    ax.set_xlabel("Noise level t")
    ax.set_ylabel("Per-sample flow loss")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def write_loss_vs_t_epoch_artifacts(
    output_dir: Union[str, Path],
    epoch: int,
    global_steps: torch.Tensor,
    t_values: torch.Tensor,
    per_sample_flow_loss: torch.Tensor,
    num_bins: int = LOSS_VS_T_NUM_BINS,
) -> Dict[str, Path]:
    """Write raw CSV, binned CSV, and PNG diagnostics for one epoch."""
    if global_steps.ndim != 1 or t_values.ndim != 1 or per_sample_flow_loss.ndim != 1:
        raise ValueError(
            "Loss-vs-t artifact writing expects 1D tensors for steps, t, and loss"
        )
    if global_steps.shape != t_values.shape or t_values.shape != per_sample_flow_loss.shape:
        raise ValueError(
            "Loss-vs-t artifact writing expects matching tensor lengths, got "
            f"{tuple(global_steps.shape)}, {tuple(t_values.shape)}, "
            f"and {tuple(per_sample_flow_loss.shape)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_cpu = global_steps.detach().to(dtype=torch.long, device="cpu")
    t_cpu = t_values.detach().to(dtype=torch.float64, device="cpu")
    loss_cpu = per_sample_flow_loss.detach().to(dtype=torch.float64, device="cpu")
    aggregated = aggregate_loss_vs_t_bins(t_cpu, loss_cpu, num_bins=num_bins)

    raw_csv_path = output_dir / "loss_vs_t_raw.csv"
    raw_rows = [
        {
            "epoch": epoch,
            "global_step": int(step),
            "t": float(t_value),
            "per_sample_flow_loss": float(loss_value),
        }
        for step, t_value, loss_value in zip(
            steps_cpu.tolist(),
            t_cpu.tolist(),
            loss_cpu.tolist(),
        )
    ]
    if raw_rows:
        _append_csv_rows(
            raw_csv_path,
            ["epoch", "global_step", "t", "per_sample_flow_loss"],
            raw_rows,
        )

    binned_csv_path = output_dir / "loss_vs_t_binned.csv"
    edges = aggregated["bin_edges"].tolist()
    centers = aggregated["bin_centers"].tolist()
    counts = aggregated["counts"].tolist()
    mean_losses = aggregated["mean_flow_loss"].tolist()
    binned_rows = [
        {
            "epoch": epoch,
            "bin_idx": bin_idx,
            "t_left": float(edges[bin_idx]),
            "t_right": float(edges[bin_idx + 1]),
            "t_center": float(centers[bin_idx]),
            "mean_flow_loss": float(mean_losses[bin_idx]),
            "count": int(counts[bin_idx]),
        }
        for bin_idx in range(num_bins)
    ]
    _append_csv_rows(
        binned_csv_path,
        [
            "epoch",
            "bin_idx",
            "t_left",
            "t_right",
            "t_center",
            "mean_flow_loss",
            "count",
        ],
        binned_rows,
    )

    epoch_plot_path = output_dir / f"loss_vs_t_epoch_{epoch:03d}.png"
    latest_plot_path = output_dir / "loss_vs_t_latest.png"
    _plot_loss_vs_t_curve(epoch_plot_path, epoch, t_cpu, loss_cpu, aggregated)
    _plot_loss_vs_t_curve(latest_plot_path, epoch, t_cpu, loss_cpu, aggregated)

    return {
        "raw_csv": raw_csv_path,
        "binned_csv": binned_csv_path,
        "epoch_plot": epoch_plot_path,
        "latest_plot": latest_plot_path,
    }


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
        self.timing_stats: Optional[TimingStats] = (
            TimingStats() if config.enable_profiling else None
        )
        self.device: Union[str, torch.device] = config.device
        self.checkpoint_dir = str(config.checkpoint_dir)
        self.wandb_logger: Optional[WandbLogger] = None
        self.encoder_ema: Optional[EMAModel[MotionHistoryEncoder]] = None
        self.predictor_ema: Optional[EMAModel[FlowMatchingPredictor]] = None
        self.optimizer: Optional[Optimizer] = None
        self.scaler: Optional[GradScaler] = None
        self.training_state: Dict[str, Any] = {}
        self.curriculum_state: Dict[str, Any] = {}
        self._latest_flow_diagnostics: Optional[Dict[str, torch.Tensor]] = None
        self._checkpoint_flow_diagnostics: Optional[Dict[str, torch.Tensor]] = None
        self.start_epoch = 0
        self.device_str = str(config.device)
        self.use_amp = False
        self.amp_dtype = torch.float32

    def _compute_post_warmup_progress(self, epoch: int) -> float | None:
        """Return rollout-schedule progress after warmup, or None during warmup."""
        if self.config.num_epochs <= 1:
            return 1.0

        progress = float(epoch) / float(self.config.num_epochs - 1)
        progress = max(0.0, min(1.0, progress))
        warmup_fraction = min(
            max(float(self.config.rollout_warmup_fraction), 0.0),
            1.0 - 1e-6,
        )

        if warmup_fraction <= 0.0:
            return progress
        if progress <= warmup_fraction:
            return None
        return max(
            0.0,
            min(1.0, (progress - warmup_fraction) / (1.0 - warmup_fraction)),
        )

    def _compute_rollout_probability(self, epoch: int) -> float:
        """Compute rollout probability with an epoch-level warmup window."""
        if self.config.num_epochs <= 1:
            return float(self.config.rollout_prob_end)

        schedule_progress = self._compute_post_warmup_progress(epoch)
        if schedule_progress is None:
            return 0.0

        return float(self.config.rollout_prob_start) + (
            (
                float(self.config.rollout_prob_end)
                - float(self.config.rollout_prob_start)
            )
            * schedule_progress
        )

    def _compute_rollout_block_length(self, epoch: int) -> int:
        """Compute the scheduled contiguous rollout block length for this epoch."""
        start = max(1, int(self.config.rollout_block_len_start))
        end = max(start, int(self.config.rollout_block_len_end))

        if self.config.num_epochs <= 1:
            return end

        schedule_progress = self._compute_post_warmup_progress(epoch)
        if schedule_progress is None:
            return start

        scheduled_length = start + (end - start) * schedule_progress
        rounded_length = int(math.floor(scheduled_length + 0.5))
        return max(start, min(end, rounded_length))

    def _compute_t_sampling_power(self, epoch: int) -> float:
        """Compute the effective high-t power-law exponent for this epoch."""
        if self.config.t_sampling_mode != "power":
            return 0.0

        target_power = max(0.0, float(self.config.t_sampling_power))
        if target_power == 0.0 or self.config.num_epochs <= 1:
            return target_power

        warmup_fraction = min(
            max(float(self.config.t_sampling_power_warmup_fraction), 0.0),
            1.0,
        )
        if warmup_fraction <= 0.0:
            return target_power

        progress = float(epoch) / float(self.config.num_epochs - 1)
        progress = max(0.0, min(1.0, progress))
        if progress >= warmup_fraction:
            return target_power

        return target_power * (progress / warmup_fraction)

    def _sample_training_timesteps(
        self,
        batch_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        epoch: int,
    ) -> torch.Tensor:
        """Sample training timesteps according to the configured PDF."""
        u = torch.rand(batch_size, device=device, dtype=dtype)
        if self.config.t_sampling_mode == "uniform":
            return u

        power = self._compute_t_sampling_power(epoch)
        if power <= 0.0:
            return u

        # Inverse CDF for p(t) = (k + 1) * t^k on [0, 1].
        return u.pow(1.0 / (power + 1.0))

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

    def setup_training_environment(self) -> None:
        device = self.device
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        ema_decay = self.config.ema_decay
        horizon = self.config.horizon
        curriculum = self.config.curriculum
        cfg_dropout = self.config.cfg_dropout
        num_epochs = self.config.num_epochs

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.encoder.to(device)
        self.predictor.to(device)

        self.wandb_logger = None
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
            self.wandb_logger = WandbLogger(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
            )

        self.encoder_ema = EMAModel(self.encoder, decay=ema_decay).to(device)
        self.predictor_ema = EMAModel(self.predictor, decay=ema_decay).to(device)
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = torch.optim.AdamW(  # type: ignore[arg-type]
            params, lr=lr, weight_decay=weight_decay
        )

        self.device_str = str(device)
        self.use_amp = self.device_str.startswith("cuda")
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float32
        if self.use_amp:
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        self.training_state = {
            "global_step": 0,
            "best_loss": float("inf"),
            "best_epoch": -1,
            "best_val_loss": float("inf"),
            "best_val_epoch": -1,
        }
        self.start_epoch = 0

        if self.resume_from is not None and os.path.exists(self.resume_from):
            print(f"Resuming from checkpoint: {self.resume_from}")
            checkpoint = torch.load(
                self.resume_from, map_location=device, weights_only=False
            )
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.predictor.load_state_dict(checkpoint["predictor"])
            if self.encoder_ema is not None:
                self.encoder_ema.model.load_state_dict(checkpoint["encoder_ema"])
            if self.predictor_ema is not None:
                self.predictor_ema.model.load_state_dict(checkpoint["predictor_ema"])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scaler is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.training_state["global_step"] = checkpoint.get("global_step", 0)
            self.training_state["best_loss"] = checkpoint.get("best_loss", float("inf"))
            self.training_state["best_epoch"] = checkpoint.get("best_epoch", -1)
            self.training_state["best_val_loss"] = checkpoint.get(
                "best_val_loss", float("inf")
            )
            self.training_state["best_val_epoch"] = checkpoint.get("best_val_epoch", -1)
            if "current_horizon" in checkpoint:
                self.training_state["current_horizon"] = checkpoint["current_horizon"]
            print(
                f"Resumed from epoch {self.start_epoch}, "
                f"step {self.training_state['global_step']}"
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

    def setup_curriculum_state(self) -> dict:
        curriculum = self.config.curriculum
        use_curriculum = curriculum is not None and len(curriculum) > 0

        if "current_horizon" in self.training_state:
            current_horizon = self.training_state["current_horizon"]
            max_horizon = self.training_state.get("max_horizon", self.config.horizon)
        elif use_curriculum and curriculum:
            current_horizon = curriculum[0]["horizon"]
            max_horizon = curriculum[-1]["horizon"]
        else:
            current_horizon = self.config.horizon
            max_horizon = self.config.horizon

        self.curriculum_state = {
            "use_curriculum": use_curriculum,
            "current_horizon": current_horizon,
            "max_horizon": max_horizon,
        }
        return self.curriculum_state

    def _loss_vs_t_output_dir(self) -> Path:
        """Directory for offline loss-vs-t diagnostics artifacts."""
        return Path(self.config.output_path) / "diagnostics" / "loss_vs_t"

    def _loss_vs_t_checkpoint_output_dir(self, filename: str) -> Path:
        """Directory for loss-vs-t artifacts associated with a checkpoint file."""
        return self._loss_vs_t_output_dir() / "checkpoints" / Path(filename).stem

    def save_training_checkpoint(
        self,
        filename: str,
        epoch: int,
        loss: float,
    ) -> None:
        encoder_ema = self.encoder_ema
        predictor_ema = self.predictor_ema
        optimizer = self.optimizer
        scaler = self.scaler
        if (
            encoder_ema is None
            or predictor_ema is None
            or optimizer is None
            or scaler is None
        ):
            raise RuntimeError("Training state is not initialized.")

        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "encoder_ema": encoder_ema.model.state_dict(),
            "predictor_ema": predictor_ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.training_state["global_step"],
            "loss": loss,
            "horizon": self.curriculum_state["max_horizon"],
            "current_horizon": self.curriculum_state["current_horizon"],
            "use_curriculum": self.curriculum_state["use_curriculum"],
            "best_loss": self.training_state["best_loss"],
            "best_epoch": self.training_state["best_epoch"],
            "best_val_loss": self.training_state["best_val_loss"],
            "best_val_epoch": self.training_state["best_val_epoch"],
        }
        checkpoint["config"] = self.config
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        diagnostics = self._checkpoint_flow_diagnostics
        if diagnostics is not None:
            artifacts = write_loss_vs_t_epoch_artifacts(
                output_dir=self._loss_vs_t_checkpoint_output_dir(filename),
                epoch=epoch,
                global_steps=diagnostics["global_steps"],
                t_values=diagnostics["t"],
                per_sample_flow_loss=diagnostics["per_sample_flow_loss"],
                num_bins=LOSS_VS_T_NUM_BINS,
            )
            if self.wandb_logger is not None:
                self.wandb_logger.log_image(
                    key=f"diagnostics/loss_vs_t/{Path(filename).stem}",
                    path=str(artifacts["epoch_plot"]),
                    step=self.training_state["global_step"],
                    caption=f"Loss vs t at epoch {epoch} ({filename})",
                )

    def incremental_flow_loss(
        self,
        batch: Dict[str, Any],
        epoch: int,
        stochastic_rollout: bool = True,
        use_cfg_dropout: bool = False,
        encoder: Optional[MotionHistoryEncoder] = None,
        predictor: Optional[FlowMatchingPredictor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        enc = encoder if encoder is not None else self.encoder
        pred_model = predictor if predictor is not None else self.predictor
        del stochastic_rollout
        device = self.device
        self._latest_flow_diagnostics = None

        motion_raw = batch["motion"].to(device)
        joints = batch["joints"].to(device)
        text = batch["text_clip"].to(device)
        B = motion_raw.shape[0]

        motion = (
            self.normalizer.normalize(motion_raw)
            if self.normalizer is not None
            else motion_raw
        )

        if (
            use_cfg_dropout
            and self.config.cfg_dropout > 0.0
            and torch.rand(1, device=device).item() <= self.config.cfg_dropout
        ):
            text_for_encoder = torch.zeros(
                B, self.config.encoder_text_dim, device=device, dtype=text.dtype
            )
        else:
            text_for_encoder = text[:, 0, :]

        motion = motion[:, 1:]
        joints = joints[:, 1:]
        j_len = joints.shape[1]

        pred_steps = j_len - 1
        hist = motion[:, :-1]
        hist_joints = joints[:, :-1]
        target_motion = motion[:, 1:]
        target_joints = joints[:, 1:]
        current_frame = hist[:, 0].detach().clone()

        with timer(self.timing_stats, "forward/gru_init"):
            context, h_state = enc.gru_step(hist[:, 0], text_for_encoder, h=None)

        rollout_prob = self._compute_rollout_probability(epoch)
        rollout_block_len = self._compute_rollout_block_length(epoch)
        ode_steps = max(1, int(self.config.rollout_integration_steps))
        current_positions = hist_joints[:, 0].detach().clone()
        rollout_steps_remaining = torch.zeros(B, dtype=torch.long, device=device)
        flow_contexts: list[torch.Tensor] = []
        flow_current_frame_features: list[torch.Tensor] = []
        flow_x0: list[torch.Tensor] = []
        flow_x1: list[torch.Tensor] = []
        flow_xt: list[torch.Tensor] = []
        flow_t: list[torch.Tensor] = []

        for step_idx in range(pred_steps):
            current_frame_features = extract_prev_frame_features(
                current_frame,
                normalizer=self.normalizer,
                normalize_output=self.normalizer is not None,
            )
            x1 = subset_271d_to_72d(
                target_motion[:, step_idx],
                prev_frame=current_frame,
                normalizer=self.normalizer,
            )
            x0 = torch.randn_like(x1)
            t = self._sample_training_timesteps(
                batch_size=B,
                device=device,
                dtype=x1.dtype,
                epoch=epoch,
            )
            t_ = t.view(B, 1)
            xt = t_ * x1 + (1 - t_) * x0
            flow_contexts.append(context)
            flow_current_frame_features.append(current_frame_features)
            flow_x0.append(x0)
            flow_x1.append(x1)
            flow_xt.append(xt)
            flow_t.append(t)

            p = float(max(0.0, min(1.0, rollout_prob)))
            active_rollout_mask = rollout_steps_remaining > 0
            eligible_mask = ~active_rollout_mask
            if p > 0.0 and eligible_mask.any():
                if p >= 1.0:
                    start_mask = eligible_mask
                else:
                    start_mask = eligible_mask & (torch.rand(B, device=device) < p)
                if start_mask.any():
                    rollout_steps_remaining[start_mask] = rollout_block_len
            rollout_mask = rollout_steps_remaining > 0

            next_input = target_motion[:, step_idx].clone()
            if rollout_mask.any():
                active_indices = rollout_mask.nonzero(as_tuple=False).squeeze(-1)
                active_context = context[active_indices]
                active_current_frame_features = current_frame_features[active_indices]
                active_text_for_encoder = text_for_encoder[active_indices]
                active_current_positions = current_positions[active_indices]
                predictor_was_training = pred_model.training
                pred_model.eval()
                try:
                    with timer(self.timing_stats, "forward/rollout_ode"):
                        with torch.no_grad():
                            x_t_roll = integrate_flow_ode(
                                predictor=pred_model,
                                track_features=active_context,
                                current_frame_features=active_current_frame_features,
                                text_embedding=active_text_for_encoder,
                                num_steps=ode_steps,
                                time_schedule_power=self.config.inference_t_schedule_power,
                                initial_state=torch.randn_like(x1[active_indices]),
                            )
                finally:
                    if predictor_was_training:
                        pred_model.train()

                with timer(self.timing_stats, "forward/pos_transform"):
                    x_t_roll_raw = (
                        self.normalizer.denormalize_flow_output(x_t_roll)
                        if self.normalizer is not None
                        else x_t_roll
                    )
                    current_frame_active = current_frame[active_indices]
                    current_frame_raw = (
                        self.normalizer.denormalize(current_frame_active)
                        if self.normalizer is not None
                        else current_frame_active
                    )
                    pred_positions_roll = flow_output_to_positions(
                        x_t_roll_raw,
                        prev_root_pos=active_current_positions[:, 0],
                        prev_root_rot_6d=current_frame_raw[:, 69:75],
                    )
                    rollout_frame, _, _ = generated_positions_to_271d(
                        new_positions=pred_positions_roll,
                        prev_positions=active_current_positions,
                        dataset_type="t2m",
                        normalizer=self.normalizer,
                    )

                next_input[active_indices] = rollout_frame

                next_positions = target_joints[:, step_idx].clone()
                next_positions[active_indices] = pred_positions_roll
                current_positions = next_positions.detach()
            else:
                current_positions = target_joints[:, step_idx].detach()

            rollout_steps_remaining[rollout_mask] -= 1

            current_frame = next_input.detach()

            with timer(self.timing_stats, "forward/gru_step"):
                context, h_state = enc.gru_step(
                    next_input.detach(), text_for_encoder, h_state
                )

        contexts_stacked = torch.stack(flow_contexts, dim=1)
        current_frame_features_stacked = torch.stack(flow_current_frame_features, dim=1)
        x0_stacked = torch.stack(flow_x0, dim=1)
        x1_stacked = torch.stack(flow_x1, dim=1)
        xt_stacked = torch.stack(flow_xt, dim=1)
        t_stacked = torch.stack(flow_t, dim=1)
        contexts_reshaped = contexts_stacked.flatten(0, 1)
        current_frame_features_reshaped = current_frame_features_stacked.flatten(0, 1)
        text_batched = (
            text_for_encoder.unsqueeze(1)
            .expand(-1, pred_steps, -1)
            .reshape(B * pred_steps, -1)
        )
        target_flow = (x1_stacked - x0_stacked).flatten(0, 1)
        xt = xt_stacked.flatten(0, 1)
        t = t_stacked.reshape(B * pred_steps)

        with timer(self.timing_stats, "forward/predictor"):
            pred, _, _ = pred_model.forward(
                track_features=contexts_reshaped,
                noisy_features=xt,
                timesteps=t,
                current_frame_features=current_frame_features_reshaped,
                text_embedding=text_batched,
                output_attentions=False,
                output_hidden_states=False,
            )
            per_sample_flow_loss = compute_per_sample_flow_loss(pred, target_flow)
            flow_loss = F.mse_loss(pred, target_flow)

        self._latest_flow_diagnostics = {
            "t": t.detach().cpu(),
            "per_sample_flow_loss": per_sample_flow_loss.detach().cpu(),
        }

        consistency_loss = flow_loss.new_zeros(())

        total_loss = flow_loss
        return total_loss, flow_loss, consistency_loss, pred_steps, rollout_prob

    def validate(
        self,
        encoder: Optional[MotionHistoryEncoder] = None,
        predictor: Optional[FlowMatchingPredictor] = None,
        epoch: int = 0,
        horizon: Optional[int] = None,
        num_batches: Optional[int] = None,
    ) -> dict:
        if self.val_dataloader is None:
            return {}

        if horizon is None:
            horizon = self.config.horizon
        if num_batches is None:
            num_batches = self.config.val_batches
        enc = encoder if encoder is not None else self.encoder
        pred = predictor if predictor is not None else self.predictor
        enc.eval()
        pred.eval()

        total_total_loss = 0.0
        total_flow_loss = 0.0
        total_consistency_loss = 0.0
        total_samples = 0

        pred_horizon = 1

        self.val_dataloader.dataset.set_horizon(1 + horizon + pred_horizon)  # type: ignore

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if num_batches > 0 and i >= num_batches:
                    break

                batch_size = batch["motion"].shape[0]

                total_loss, flow_loss, consistency_loss, _, _ = (
                    self.incremental_flow_loss(
                        batch=batch,
                        epoch=epoch,
                        stochastic_rollout=True,
                        use_cfg_dropout=False,
                        encoder=enc,
                        predictor=pred,
                    )
                )

                total_total_loss += total_loss.item() * batch_size
                total_flow_loss += flow_loss.item() * batch_size
                total_consistency_loss += consistency_loss.item() * batch_size
                total_samples += batch_size

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
        self.setup_training_environment()
        self.setup_curriculum_state()
        optimizer = self.optimizer
        scaler = self.scaler
        encoder_ema = self.encoder_ema
        predictor_ema = self.predictor_ema
        if (
            optimizer is None
            or scaler is None
            or encoder_ema is None
            or predictor_ema is None
        ):
            raise RuntimeError("Training environment is not initialized.")

        num_epochs = self.config.num_epochs
        lr = self.config.learning_rate
        max_grad_norm = self.config.gradient_clip
        val_interval = self.config.val_interval
        val_batches = self.config.val_batches
        val_use_ema = self.config.val_use_ema
        wandb_logger = self.wandb_logger

        epoch = self.start_epoch - 1
        try:
            for epoch in tqdm(
                range(self.start_epoch, num_epochs), desc="Training", unit="epoch"
            ):
                prev_horizon = self.curriculum_state["current_horizon"]
                if self.curriculum_state["use_curriculum"] and self.config.curriculum:
                    for level in reversed(self.config.curriculum):
                        if epoch <= level["epochs"]:
                            self.curriculum_state["current_horizon"] = level["horizon"]
                        else:
                            break
                    if self.curriculum_state["current_horizon"] != prev_horizon:
                        tqdm.write(
                            "Curriculum update: "
                            f"horizon {prev_horizon} -> {self.curriculum_state['current_horizon']}"
                        )

                self._checkpoint_flow_diagnostics = None
                epoch_loss = 0.0
                num_batches = 0
                pred_horizon = 1
                epoch_diag_steps: list[torch.Tensor] = []
                epoch_diag_t: list[torch.Tensor] = []
                epoch_diag_loss: list[torch.Tensor] = []

                self.dataloader.dataset.set_horizon(  # type: ignore
                    1 + self.curriculum_state["current_horizon"] + pred_horizon
                )

                pbar = tqdm(
                    self.dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch"
                )
                batch_start_time = time.time()
                timing_log_interval = max(1, int(self.config.timing_log_interval))

                for batch in pbar:
                    batch_size = batch["motion"].shape[0]
                    effective_horizon = self.curriculum_state["current_horizon"]
                    step_before_update = self.training_state["global_step"]

                    optimizer.zero_grad(set_to_none=True)

                    with timer(self.timing_stats, "forward"):
                        with torch.amp.autocast(
                            self.device_str,
                            dtype=self.amp_dtype,
                            enabled=self.use_amp,
                        ):  # type: ignore
                            (
                                loss,
                                flow_loss,
                                consistency_loss,
                                pred_horizon,
                                rollout_prob,
                            ) = self.incremental_flow_loss(
                                batch=batch,
                                epoch=epoch,
                                stochastic_rollout=True,
                                use_cfg_dropout=True,
                            )

                    nonfinite_losses = [
                        name
                        for name, value in {
                            "loss": loss,
                            "flow_loss": flow_loss,
                            "consistency_loss": consistency_loss,
                        }.items()
                        if not torch.isfinite(value).all()
                    ]
                    if nonfinite_losses:
                        tqdm.write(
                            f"[Epoch {epoch}] [Step {self.training_state['global_step']}] "
                            "Skipping batch with non-finite losses: "
                            + ", ".join(nonfinite_losses)
                        )
                        if wandb_logger is not None:
                            wandb_logger.log(
                                {"train/skipped_nonfinite_batch": 1},
                                step=self.training_state["global_step"],
                            )
                        pbar.set_postfix({"skip": ",".join(nonfinite_losses)})
                        optimizer.zero_grad(set_to_none=True)
                        self.training_state["global_step"] += 1
                        continue

                    diagnostics = self._latest_flow_diagnostics
                    if diagnostics is not None:
                        t_diag = diagnostics["t"]
                        loss_diag = diagnostics["per_sample_flow_loss"]
                        if t_diag.shape != loss_diag.shape:
                            raise RuntimeError(
                                "Flow diagnostics shape mismatch: "
                                f"{tuple(t_diag.shape)} vs {tuple(loss_diag.shape)}"
                            )
                        epoch_diag_t.append(t_diag)
                        epoch_diag_loss.append(loss_diag)
                        epoch_diag_steps.append(
                            torch.full(
                                t_diag.shape,
                                step_before_update,
                                dtype=torch.long,
                            )
                        )

                    with timer(self.timing_stats, "backward"):
                        scaler.scale(loss).backward()

                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters())
                            + list(self.predictor.parameters()),
                            max_grad_norm,
                        )

                        scaler.step(optimizer)
                        scaler.update()

                    with timer(self.timing_stats, "ema_update"):
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

                    if wandb_logger is not None:
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/epoch": epoch,
                            "train/grad_norm": (
                                grad_norm.item()
                                if hasattr(grad_norm, "item")
                                else grad_norm
                            ),
                            "train/batch_time": batch_time,
                            "train/samples_per_sec": (
                                batch_size / batch_time if batch_time > 0 else 0
                            ),
                            "train/current_horizon": self.curriculum_state[
                                "current_horizon"
                            ],
                            "train/effective_horizon": effective_horizon,
                            "train/num_pred_frames": pred_horizon,
                            "train/loss_L_total": loss.item(),
                            "train/loss_L_flow": flow_loss.item(),
                            "train/loss_L_consistency": consistency_loss.item(),
                            "train/loss_rollout_prob": rollout_prob,
                        }
                        wandb_logger.log(
                            log_dict, step=self.training_state["global_step"]
                        )

                    if (
                        self.timing_stats is not None
                        and (self.training_state["global_step"] + 1)
                        % timing_log_interval
                        == 0
                    ):
                        timing_dict = self.timing_stats.get_averages()
                        if wandb_logger is not None:
                            wandb_logger.log(
                                {
                                    f"time/{key}_ms": val
                                    for key, val in timing_dict.items()
                                },
                                step=self.training_state["global_step"],
                            )
                        tqdm.write(
                            f"[Step {self.training_state['global_step']}] {str(self.timing_stats)}"
                        )

                    if self.training_state["global_step"] % 100 == 0:
                        tqdm.write(
                            f"[Epoch {epoch}] [Step {self.training_state['global_step']}] "
                            f"loss={loss.item():.6f} lr={lr:.2e}"
                        )

                    epoch_loss += loss.item()
                    num_batches += 1
                    self.training_state["global_step"] += 1

                pbar.close()
                avg_epoch_loss = epoch_loss / max(1, num_batches)
                if epoch_diag_t and epoch_diag_loss and epoch_diag_steps:
                    self._checkpoint_flow_diagnostics = {
                        "global_steps": torch.cat(epoch_diag_steps),
                        "t": torch.cat(epoch_diag_t),
                        "per_sample_flow_loss": torch.cat(epoch_diag_loss),
                    }
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

                    try:
                        with timer(self.timing_stats, "validation"):
                            val_metrics = self.validate(
                                encoder=val_encoder,
                                predictor=val_predictor,
                                horizon=self.curriculum_state["current_horizon"],
                                num_batches=val_batches,
                                epoch=epoch,
                            )

                        val_loss = val_metrics["val_loss"]
                        tqdm.write(f"Validation loss: {val_loss:.6f}")

                        if wandb_logger is not None:
                            wandb_logger.log(
                                {
                                    "val/loss": val_loss,
                                    "val/epoch": epoch,
                                },
                                step=self.training_state["global_step"],
                            )
                    except Exception as e:
                        tqdm.write(f"Validation error: {str(e)}")
                        val_metrics = {}

                if wandb_logger is not None:
                    epoch_log = {
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/num": epoch,
                        "epoch/current_horizon": self.curriculum_state[
                            "current_horizon"
                        ],
                    }
                    if val_metrics and "val_loss" in val_metrics:
                        epoch_log["epoch/val_loss"] = val_metrics["val_loss"]
                    wandb_logger.log(
                        epoch_log,
                        step=self.training_state["global_step"],
                    )

                with timer(self.timing_stats, "checkpoint"):
                    if (
                        self.config.checkpoint_interval > 0
                        and (epoch + 1) % self.config.checkpoint_interval == 0
                    ):
                        self.save_training_checkpoint(
                            filename="latest.pt",
                            epoch=epoch,
                            loss=avg_epoch_loss,
                        )
                        if avg_epoch_loss < self.training_state["best_loss"]:
                            tqdm.write(
                                "New best model! "
                                f"(Loss: {self.training_state['best_loss']:.6f} -> {avg_epoch_loss:.6f})"
                            )
                            self.training_state["best_loss"] = avg_epoch_loss
                            self.training_state["best_epoch"] = epoch
                            self.save_training_checkpoint(
                                filename="best.pt",
                                epoch=epoch,
                                loss=avg_epoch_loss,
                            )

                    if (
                        self.config.save_best_val
                        and val_metrics
                        and "val_loss" in val_metrics
                    ):
                        val_loss = val_metrics["val_loss"]
                        if val_loss < self.training_state["best_val_loss"]:
                            tqdm.write(
                                "New best validation model! "
                                f"(Val Loss: {self.training_state['best_val_loss']:.6f} -> {val_loss:.6f})"
                            )
                            self.training_state["best_val_loss"] = val_loss
                            self.training_state["best_val_epoch"] = epoch
                            self.save_training_checkpoint(
                                filename="best_val.pt",
                                epoch=epoch,
                                loss=avg_epoch_loss,
                            )

                if self.timing_stats is not None:
                    tqdm.write(str(self.timing_stats))
                    if wandb_logger is not None:
                        epoch_timing = self.timing_stats.get_averages()
                        wandb_logger.log(
                            {
                                f"epoch_time/{key}_ms": val
                                for key, val in epoch_timing.items()
                            },
                            step=self.training_state["global_step"],
                        )
                    self.timing_stats.reset()

        except KeyboardInterrupt:
            tqdm.write("Training interrupted. Saving emergency checkpoint...")
            self.save_training_checkpoint(
                filename="latest_interrupted.pt",
                epoch=epoch,
                loss=0.0,
            )
            tqdm.write("Done.")

        if wandb_logger is not None:
            summary = {
                "best_loss": self.training_state["best_loss"],
                "best_epoch": self.training_state["best_epoch"],
                "best_val_loss": self.training_state["best_val_loss"],
                "best_val_epoch": self.training_state["best_val_epoch"],
            }
            if self.curriculum_state["use_curriculum"]:
                summary["final_horizon"] = self.curriculum_state["current_horizon"]
                summary["max_horizon"] = self.curriculum_state["max_horizon"]
            wandb_logger.log_summary(summary)
            wandb_logger.finish()

        return encoder_ema, predictor_ema
