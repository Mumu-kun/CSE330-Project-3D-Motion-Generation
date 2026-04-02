"""Tests for offline flow-loss-vs-t diagnostics."""

import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from utils.motion_utils import FeatureNormalizer, yaw_to_root_rot6d
from utils.train_utils import (
    EMAModel,
    Trainer,
    aggregate_loss_vs_t_bins,
    write_loss_vs_t_epoch_artifacts,
)


class DummyEncoder(nn.Module):
    """Minimal encoder stub for incremental_flow_loss testing."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward_all(
        self,
        motion_seq: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        del text_embedding
        batch_size, seq_len = motion_seq.shape[:2]
        context = torch.zeros(
            batch_size,
            seq_len,
            22,
            4,
            device=motion_seq.device,
            dtype=motion_seq.dtype,
        )
        return context + self.anchor * 0

    def forward(
        self,
        motion_seq: torch.Tensor,
        text_embedding: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        outputs = self.forward_all(motion_seq, text_embedding)
        return outputs if return_all else outputs[:, -1]


class DummyPredictor(nn.Module):
    """Minimal predictor stub returning a zero flow field."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        track_features: torch.Tensor,
        noisy_features: torch.Tensor,
        timesteps: torch.Tensor,
        current_frame_features: torch.Tensor,
        text_embedding: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, None, None]:
        del (
            track_features,
            timesteps,
            current_frame_features,
            text_embedding,
            output_attentions,
            output_hidden_states,
        )
        return noisy_features * 0 + self.anchor * 0, None, None


def _make_dummy_trainer(
    *, normalizer: FeatureNormalizer | None = None, use_consistency_loss: bool = False
) -> Trainer:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    config.num_epochs = 1
    config.use_consistency_loss = use_consistency_loss
    dataset = TensorDataset(torch.zeros(1))
    dataloader = DataLoader(dataset, batch_size=1)
    return Trainer(
        encoder=DummyEncoder(),  # type: ignore[arg-type]
        predictor=DummyPredictor(),  # type: ignore[arg-type]
        dataloader=dataloader,
        config=config,
        normalizer=normalizer,
    )


def test_incremental_flow_loss_collects_per_sample_t_and_loss() -> None:
    trainer = _make_dummy_trainer()
    batch = {
        "motion": torch.zeros(2, 4, 271, dtype=torch.float32),
        "joints": torch.zeros(2, 4, 22, 3, dtype=torch.float32),
        "text_clip": torch.zeros(2, 1, 512, dtype=torch.float32),
    }

    total_loss, flow_loss, consistency_loss, pred_steps, rollout_prob = (
        trainer.incremental_flow_loss(batch=batch, epoch=0)
    )

    diagnostics = trainer._latest_flow_diagnostics
    assert diagnostics is not None
    assert pred_steps == 2
    assert rollout_prob == 0.0
    assert diagnostics["t"].shape == diagnostics["per_sample_flow_loss"].shape
    assert diagnostics["t"].numel() == 4
    assert torch.all(diagnostics["t"] >= 0.0)
    assert torch.all(diagnostics["t"] <= 1.0)
    assert torch.isclose(
        flow_loss.detach().cpu(),
        diagnostics["per_sample_flow_loss"].mean(),
    )
    assert torch.isclose(total_loss.detach().cpu(), flow_loss.detach().cpu())
    assert consistency_loss.item() == 0.0


def test_incremental_flow_loss_consistency_uses_prev_frame_and_joint_targets() -> None:
    normalizer = FeatureNormalizer(torch.zeros(271), torch.ones(271))
    trainer = _make_dummy_trainer(
        normalizer=normalizer,
        use_consistency_loss=True,
    )
    trainer.config.consistency_loss_t_threshold = -1.0

    motion = torch.zeros(2, 4, 271, dtype=torch.float32)
    motion[..., 69:75] = yaw_to_root_rot6d(torch.zeros(2, 4, dtype=torch.float32))
    batch = {
        "motion": motion,
        "joints": torch.zeros(2, 4, 22, 3, dtype=torch.float32),
        "text_clip": torch.zeros(2, 1, 512, dtype=torch.float32),
    }

    total_loss, flow_loss, consistency_loss, pred_steps, rollout_prob = (
        trainer.incremental_flow_loss(batch=batch, epoch=0)
    )

    assert pred_steps == 2
    assert rollout_prob == 0.0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(flow_loss)
    assert torch.isfinite(consistency_loss)
    assert consistency_loss.item() >= 0.0


def test_aggregate_loss_vs_t_bins_uses_100_bin_indexing_and_nan_for_empty_bins() -> None:
    t_values = torch.tensor([0.0, 0.009, 0.011, 0.99, 1.0], dtype=torch.float32)
    losses = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], dtype=torch.float32)

    aggregated = aggregate_loss_vs_t_bins(t_values, losses, num_bins=100)

    assert aggregated["counts"][0].item() == 2
    assert math.isclose(aggregated["mean_flow_loss"][0].item(), 2.0)
    assert aggregated["counts"][1].item() == 1
    assert math.isclose(aggregated["mean_flow_loss"][1].item(), 5.0)
    assert aggregated["counts"][50].item() == 0
    assert math.isnan(aggregated["mean_flow_loss"][50].item())
    assert aggregated["counts"][99].item() == 2
    assert math.isclose(aggregated["mean_flow_loss"][99].item(), 8.0)


def test_write_loss_vs_t_epoch_artifacts_outputs_csv_and_png(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostics" / "loss_vs_t"
    global_steps = torch.tensor([12, 12, 13], dtype=torch.long)
    t_values = torch.tensor([0.001, 0.502, 1.0], dtype=torch.float32)
    losses = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    artifacts = write_loss_vs_t_epoch_artifacts(
        output_dir=output_dir,
        epoch=7,
        global_steps=global_steps,
        t_values=t_values,
        per_sample_flow_loss=losses,
        num_bins=100,
    )

    for artifact_path in artifacts.values():
        assert artifact_path.exists()
        assert artifact_path.stat().st_size > 0

    with artifacts["raw_csv"].open("r", newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    assert len(raw_rows) == 3
    assert raw_rows[0]["epoch"] == "7"
    assert raw_rows[0]["global_step"] == "12"

    with artifacts["binned_csv"].open("r", newline="", encoding="utf-8") as handle:
        binned_rows = list(csv.DictReader(handle))
    assert len(binned_rows) == 100
    populated = [row for row in binned_rows if int(row["count"]) > 0]
    assert len(populated) == 3
    empty = next(row for row in binned_rows if int(row["count"]) == 0)
    assert math.isnan(float(empty["mean_flow_loss"]))


def test_save_training_checkpoint_writes_loss_vs_t_artifacts_for_checkpoint(
    tmp_path: Path,
) -> None:
    config = Config(
        device="cpu",
        output_path=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    dataset = TensorDataset(torch.zeros(1))
    dataloader = DataLoader(dataset, batch_size=1)
    trainer = Trainer(
        encoder=DummyEncoder(),  # type: ignore[arg-type]
        predictor=DummyPredictor(),  # type: ignore[arg-type]
        dataloader=dataloader,
        config=config,
        normalizer=None,
    )

    trainer.encoder_ema = EMAModel(trainer.encoder, decay=0.9)
    trainer.predictor_ema = EMAModel(trainer.predictor, decay=0.9)
    trainer.optimizer = torch.optim.SGD(
        list(trainer.encoder.parameters()) + list(trainer.predictor.parameters()),
        lr=1e-3,
    )
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.training_state = {
        "global_step": 17,
        "best_loss": 0.4,
        "best_epoch": 2,
        "best_val_loss": 0.3,
        "best_val_epoch": 1,
    }
    trainer.curriculum_state = {
        "max_horizon": 40,
        "current_horizon": 10,
        "use_curriculum": True,
    }
    trainer._checkpoint_flow_diagnostics = {
        "global_steps": torch.tensor([16, 16, 17], dtype=torch.long),
        "t": torch.tensor([0.1, 0.6, 0.9], dtype=torch.float32),
        "per_sample_flow_loss": torch.tensor([0.3, 0.2, 0.4], dtype=torch.float32),
    }

    trainer.save_training_checkpoint(filename="latest.pt", epoch=3, loss=0.25)

    checkpoint_path = config.checkpoint_dir / "latest.pt"
    assert checkpoint_path.exists()

    diagnostics_dir = (
        config.output_path / "diagnostics" / "loss_vs_t" / "checkpoints" / "latest"
    )
    assert (diagnostics_dir / "loss_vs_t_raw.csv").exists()
    assert (diagnostics_dir / "loss_vs_t_binned.csv").exists()
    assert (diagnostics_dir / "loss_vs_t_epoch_003.png").exists()
    assert (diagnostics_dir / "loss_vs_t_latest.png").exists()
