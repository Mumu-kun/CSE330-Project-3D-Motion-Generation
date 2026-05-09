"""Tests for offline flow-loss-vs-t diagnostics."""

import csv
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import train_utils as train_utils_module
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
        self.step_inputs: list[torch.Tensor] = []

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
        context[..., 0] = motion_seq[..., 0].unsqueeze(-1).expand(-1, -1, 22)
        return context + self.anchor * 0

    def forward(
        self,
        motion_seq: torch.Tensor,
        text_embedding: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        outputs = self.forward_all(motion_seq, text_embedding)
        return outputs if return_all else outputs[:, -1]

    def step(
        self,
        x_t: torch.Tensor,
        text_embedding: torch.Tensor,
        frame_buffer: torch.Tensor | None,
        cache_state=None,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None]:
        del text_embedding
        self.step_inputs.append(x_t.detach().cpu().clone())
        if frame_buffer is None:
            next_frame_buffer = x_t.unsqueeze(1)
        else:
            next_frame_buffer = torch.cat([frame_buffer, x_t.unsqueeze(1)], dim=1)

        context = torch.zeros(
            x_t.shape[0],
            22,
            4,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        context[..., 0] = x_t[:, 0].unsqueeze(-1).expand(-1, 22)
        return context + self.anchor * 0, next_frame_buffer, cache_state


class DummyPredictor(nn.Module):
    """Minimal predictor stub returning a zero flow field."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.flow_dim = 68
        self.root_state_dim = 5
        self.current_frame_features_calls: list[torch.Tensor] = []

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
            text_embedding,
            output_attentions,
            output_hidden_states,
        )
        self.current_frame_features_calls.append(
            current_frame_features.detach().cpu().clone()
        )
        return noisy_features * 0 + self.anchor * 0, None, None


class DummyBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.horizon: int | None = None

    def __len__(self) -> int:
        return int(self.batch["motion"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.batch.items()}

    def set_horizon(self, horizon: int) -> None:
        self.horizon = horizon


def _make_batch(root_heights: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, seq_len = root_heights.shape
    motion = torch.zeros(batch_size, seq_len, 271, dtype=torch.float32)
    motion[..., 0] = root_heights
    motion[..., 69:75] = yaw_to_root_rot6d(
        torch.zeros(batch_size, seq_len, dtype=torch.float32)
    )

    joints = torch.zeros(batch_size, seq_len, 22, 3, dtype=torch.float32)
    joints[..., 0, 1] = root_heights

    return {
        "motion": motion,
        "joints": joints,
        "text_clip": torch.zeros(batch_size, 1, 512, dtype=torch.float32),
    }


def _install_simple_rollout_generation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    delta_height: float,
) -> dict[str, int]:
    call_state = {"integrate_flow_ode": 0}

    def fake_integrate_flow_ode(**kwargs) -> torch.Tensor:
        call_state["integrate_flow_ode"] += 1
        current_frame_features = kwargs["current_frame_features"]
        batch_size = current_frame_features.shape[0]
        output = torch.zeros(batch_size, 68, dtype=current_frame_features.dtype)
        output[:, 0] = current_frame_features[:, 0] + delta_height
        return output

    def fake_flow_output_to_positions(
        flow_output: torch.Tensor,
        prev_root_pos: torch.Tensor,
        prev_root_rot_6d: torch.Tensor,
    ) -> torch.Tensor:
        del prev_root_rot_6d
        batch_size = flow_output.shape[0]
        positions = torch.zeros(batch_size, 22, 3, dtype=flow_output.dtype)
        positions[:, 0, 0] = prev_root_pos[:, 0]
        positions[:, 0, 1] = flow_output[:, 0]
        positions[:, 0, 2] = prev_root_pos[:, 2]
        return positions

    def fake_generated_positions_to_271d(
        new_positions: torch.Tensor,
        prev_positions: torch.Tensor | None = None,
        normalizer: FeatureNormalizer | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del kwargs, prev_positions
        batch_size = new_positions.shape[0]
        next_frame = torch.zeros(batch_size, 271, dtype=new_positions.dtype)
        next_frame[:, 0] = new_positions[:, 0, 1]
        next_frame[:, 69:75] = yaw_to_root_rot6d(torch.zeros(batch_size))
        if normalizer is not None:
            next_frame = normalizer.normalize(next_frame)
        return next_frame, new_positions[:, 0], None

    monkeypatch.setattr(train_utils_module, "integrate_flow_ode", fake_integrate_flow_ode)
    monkeypatch.setattr(
        train_utils_module,
        "flow_output_to_positions",
        fake_flow_output_to_positions,
    )
    monkeypatch.setattr(
        train_utils_module,
        "generated_positions_to_271d",
        fake_generated_positions_to_271d,
    )
    return call_state


def _make_dummy_trainer(
    *,
    normalizer: FeatureNormalizer | None = None,
    use_consistency_loss: bool = False,
    val_dataloader: DataLoader | None = None,
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
        val_dataloader=val_dataloader,
    )


def test_incremental_flow_loss_collects_per_sample_t_and_loss() -> None:
    trainer = _make_dummy_trainer()
    batch = {
        "motion": torch.zeros(2, 4, 271, dtype=torch.float32),
        "joints": torch.zeros(2, 4, 22, 3, dtype=torch.float32),
        "text_clip": torch.zeros(2, 1, 512, dtype=torch.float32),
    }

    total_loss, flow_loss, consistency_loss, pred_steps, rollout_prob = (
        trainer.incremental_flow_loss(
            batch=batch,
            epoch=0,
            collect_diagnostics=True,
        )
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


def test_incremental_flow_loss_skips_diagnostics_when_not_collected() -> None:
    trainer = _make_dummy_trainer()
    batch = {
        "motion": torch.zeros(2, 4, 271, dtype=torch.float32),
        "joints": torch.zeros(2, 4, 22, 3, dtype=torch.float32),
        "text_clip": torch.zeros(2, 1, 512, dtype=torch.float32),
    }

    trainer.incremental_flow_loss(batch=batch, epoch=0)
    assert trainer._latest_flow_diagnostics is None


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


def test_rollout_subset_size_follows_config_fraction() -> None:
    trainer = _make_dummy_trainer()
    trainer.config.rollout_subset_fraction = 0.5
    trainer.config.rollout_prob_start = 1.0
    trainer.config.rollout_prob_end = 1.0
    trainer.config.rollout_loss_weight = 1.0
    trainer.config.rollout_block_len_start = 1
    trainer.config.rollout_block_len_end = 1

    batch = _make_batch(torch.zeros(6, 4, dtype=torch.float32))
    trainer.incremental_flow_loss(batch=batch, epoch=0)

    predictor = trainer.predictor
    assert isinstance(predictor, DummyPredictor)
    assert predictor.current_frame_features_calls[-1].shape[0] == 6


def test_incremental_flow_loss_rollout_self_feeds_generated_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalizer = FeatureNormalizer(torch.zeros(271), torch.ones(271))
    trainer = _make_dummy_trainer(
        normalizer=normalizer,
        use_consistency_loss=True,
    )
    trainer.config.rollout_prob_start = 1.0
    trainer.config.rollout_prob_end = 1.0
    trainer.config.rollout_subset_fraction = 1.0
    trainer.config.rollout_loss_weight = 1.0
    trainer.config.rollout_block_len_start = 2
    trainer.config.rollout_block_len_end = 2
    trainer.config.rollout_block_len_bias_power = 1000.0
    trainer.config.consistency_loss_t_threshold = -1.0

    call_state = _install_simple_rollout_generation(
        monkeypatch,
        delta_height=1.0,
    )

    batch = _make_batch(
        torch.tensor(
            [
                [0.0, 10.0, 20.0, 30.0],
                [0.0, 100.0, 200.0, 300.0],
            ],
            dtype=torch.float32,
        )
    )

    total_loss, flow_loss, consistency_loss, pred_steps, rollout_prob = (
        trainer.incremental_flow_loss(batch=batch, epoch=0)
    )

    encoder = trainer.encoder
    predictor = trainer.predictor
    assert isinstance(encoder, DummyEncoder)
    assert isinstance(predictor, DummyPredictor)
    assert pred_steps == 2
    assert rollout_prob == 1.0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(flow_loss)
    assert torch.isfinite(consistency_loss)

    step_root_heights = torch.stack(
        [step_input[:, 0] for step_input in encoder.step_inputs],
        dim=1,
    )
    expected_heights = torch.tensor(
        [
            [10.0, 11.0],
            [100.0, 101.0],
        ],
        dtype=torch.float32,
    )
    assert call_state["integrate_flow_ode"] >= 1
    assert torch.allclose(step_root_heights, expected_heights)

    rollout_current_frame_features = predictor.current_frame_features_calls[-1]
    rollout_root_heights = rollout_current_frame_features[:, 0].reshape(2, 2)
    assert torch.allclose(rollout_root_heights, expected_heights)


def test_validate_runs_with_rollout_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_batch(
        torch.tensor(
            [
                [0.0, 5.0, 10.0, 15.0],
                [0.0, 7.0, 14.0, 21.0],
            ],
            dtype=torch.float32,
        )
    )
    val_dataset = DummyBatchDataset(batch)
    val_loader = DataLoader(val_dataset, batch_size=2)

    trainer = _make_dummy_trainer(
        normalizer=FeatureNormalizer(torch.zeros(271), torch.ones(271)),
        use_consistency_loss=False,
        val_dataloader=val_loader,
    )
    trainer.config.rollout_prob_start = 1.0
    trainer.config.rollout_prob_end = 1.0
    trainer.config.rollout_subset_fraction = 1.0
    trainer.config.rollout_block_len_start = 2
    trainer.config.rollout_block_len_end = 2

    _install_simple_rollout_generation(
        monkeypatch,
        delta_height=0.5,
    )

    metrics = trainer.validate(epoch=0, num_batches=1)

    assert val_dataset.horizon is not None
    assert set(metrics) == {
        "val_loss",
        "val_flow_loss",
        "val_consistency_loss",
    }
    assert all(math.isfinite(value) for value in metrics.values())


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


def test_write_loss_vs_t_epoch_artifacts_outputs_binned_csv_and_png(
    tmp_path: Path,
) -> None:
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

    assert set(artifacts) == {"binned_csv", "epoch_plot", "latest_plot"}
    for artifact_path in artifacts.values():
        assert artifact_path.exists()
        assert artifact_path.stat().st_size > 0

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
    assert (diagnostics_dir / "loss_vs_t_binned.csv").exists()
    assert (diagnostics_dir / "loss_vs_t_epoch_003.png").exists()
    assert (diagnostics_dir / "loss_vs_t_latest.png").exists()


def test_save_training_checkpoint_skips_loss_vs_t_artifacts_when_unsampled(
    tmp_path: Path,
) -> None:
    config = Config(
        device="cpu",
        output_path=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    config.loss_vs_t_checkpoint_epoch_interval = 100
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
    trainer._checkpoint_flow_diagnostics = None

    trainer.save_training_checkpoint(filename="latest.pt", epoch=3, loss=0.25)

    checkpoint_path = config.checkpoint_dir / "latest.pt"
    assert checkpoint_path.exists()

    diagnostics_dir = (
        config.output_path / "diagnostics" / "loss_vs_t" / "checkpoints" / "latest"
    )
    assert not diagnostics_dir.exists()
