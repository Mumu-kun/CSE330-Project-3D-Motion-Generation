"""Tests for blocked rollout behavior in incremental training loss."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
import utils.train_utils as train_utils
from utils.motion_utils import (
    FeatureNormalizer,
    flow_output_to_positions,
    generated_positions_to_271d,
)
from utils.train_utils import Trainer


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "tests" / "dataset" / "humanml3d-subset-mini"


def _identity_normalizer() -> FeatureNormalizer:
    return FeatureNormalizer(torch.zeros(271), torch.ones(271))


def _load_motion_and_joints(sample_id: str = "000070") -> tuple[torch.Tensor, torch.Tensor]:
    motion = torch.from_numpy(
        np.load(DATASET_ROOT / "new_joint_vecs" / f"{sample_id}.npy")
    ).float()
    joints = torch.from_numpy(
        np.load(DATASET_ROOT / "new_joints" / f"{sample_id}.npy")
    ).float()
    return motion, joints


class RecordingEncoder(nn.Module):
    def __init__(self, joint_count: int = 22, feature_dim: int = 8) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.joint_count = joint_count
        self.feature_dim = feature_dim
        self.recorded_inputs: list[torch.Tensor] = []

    def gru_step(
        self,
        x_t: torch.Tensor,
        text_emb: torch.Tensor,
        h: torch.Tensor | None,
        use_normalization: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del text_emb, h, use_normalization
        self.recorded_inputs.append(x_t.detach().clone())
        batch_size = x_t.shape[0]
        context = torch.zeros(
            batch_size,
            self.joint_count,
            self.feature_dim,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        h_next = torch.zeros(1, batch_size, 1, device=x_t.device, dtype=x_t.dtype)
        return context + self.anchor * 0, h_next


class RecordingPredictor(nn.Module):
    def __init__(self, rollout_response: str = "zero") -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.rollout_response = rollout_response
        self.recorded_batch_sizes: list[int] = []

    def forward(
        self,
        noisy_features: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        track_features: torch.Tensor,
        current_frame_features: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        del (
            timesteps,
            text_embedding,
            track_features,
            current_frame_features,
            output_attentions,
            output_hidden_states,
            kwargs,
        )
        self.recorded_batch_sizes.append(int(noisy_features.shape[0]))
        if self.rollout_response == "cancel":
            return -noisy_features + self.anchor * 0, None, None
        return torch.zeros_like(noisy_features) + self.anchor * 0, None, None


def _make_trainer(config: Config, encoder: nn.Module, predictor: nn.Module) -> Trainer:
    dataset = TensorDataset(torch.zeros(1))
    dataloader = DataLoader(dataset, batch_size=1)
    return Trainer(
        encoder=encoder,  # type: ignore[arg-type]
        predictor=predictor,  # type: ignore[arg-type]
        dataloader=dataloader,
        config=config,
        normalizer=_identity_normalizer(),
    )


def test_incremental_flow_loss_keeps_teacher_forcing_when_rollout_prob_zero() -> None:
    motion, joints = _load_motion_and_joints()
    batch = {
        "motion": motion[1:5].unsqueeze(0),
        "joints": joints[1:5].unsqueeze(0),
        "text_clip": torch.zeros(1, 1, 512),
    }

    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 2
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    config.rollout_block_len_start = 2
    config.rollout_block_len_end = 2
    config.rollout_integration_steps = 1

    encoder = RecordingEncoder()
    predictor = RecordingPredictor(rollout_response="cancel")
    trainer = _make_trainer(config, encoder, predictor)

    trainer.incremental_flow_loss(batch=batch, epoch=1)

    assert len(encoder.recorded_inputs) == 3
    assert torch.allclose(encoder.recorded_inputs[1], batch["motion"][:, 2])
    assert torch.allclose(encoder.recorded_inputs[2], batch["motion"][:, 3])
    assert predictor.recorded_batch_sizes == [2]


def test_incremental_flow_loss_uses_contiguous_blocked_rollout(
    monkeypatch,
) -> None:
    motion, joints = _load_motion_and_joints()
    batch = {
        "motion": motion[1:6].unsqueeze(0),
        "joints": joints[1:6].unsqueeze(0),
        "text_clip": torch.zeros(1, 1, 512),
    }

    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 2
    config.rollout_prob_start = 0.5
    config.rollout_prob_end = 0.5
    config.rollout_block_len_start = 2
    config.rollout_block_len_end = 2
    config.rollout_integration_steps = 1

    rand_outputs = iter(
        [
            torch.tensor([0.25], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.30], dtype=torch.float32),
            torch.tensor([0.40], dtype=torch.float32),
            torch.tensor([0.9], dtype=torch.float32),
        ]
    )
    original_rand = torch.rand

    def fake_rand(*size, **kwargs):
        if len(size) == 1 and isinstance(size[0], int) and size[0] == 1:
            tensor = next(rand_outputs)
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", tensor.dtype)
            return tensor.to(device=device, dtype=dtype)
        return original_rand(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", fake_rand)
    rollout_calls: list[dict[str, object]] = []

    def fake_integrate_flow_ode(**kwargs):
        rollout_calls.append(
            {
                "batch_size": int(kwargs["track_features"].shape[0]),
                "num_steps": int(kwargs["num_steps"]),
                "time_schedule_power": float(kwargs["time_schedule_power"]),
                "initial_state_shape": tuple(kwargs["initial_state"].shape),
            }
        )
        return torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(train_utils, "integrate_flow_ode", fake_integrate_flow_ode)

    encoder = RecordingEncoder()
    predictor = RecordingPredictor(rollout_response="cancel")
    trainer = _make_trainer(config, encoder, predictor)

    trainer.incremental_flow_loss(batch=batch, epoch=1)

    zero_flow = torch.zeros(1, 68)
    rollout_positions_1 = flow_output_to_positions(
        zero_flow,
        prev_root_pos=batch["joints"][:, 1, 0],
        prev_root_rot_6d=batch["motion"][:, 1, 69:75],
    )
    rollout_frame_1, _, _ = generated_positions_to_271d(
        new_positions=rollout_positions_1,
        prev_positions=batch["joints"][:, 1],
        dataset_type="t2m",
        normalizer=trainer.normalizer,
    )
    rollout_positions_2 = flow_output_to_positions(
        zero_flow,
        prev_root_pos=rollout_positions_1[:, 0],
        prev_root_rot_6d=rollout_frame_1[:, 69:75],
    )
    rollout_frame_2, _, _ = generated_positions_to_271d(
        new_positions=rollout_positions_2,
        prev_positions=rollout_positions_1,
        dataset_type="t2m",
        normalizer=trainer.normalizer,
    )

    assert len(encoder.recorded_inputs) == 4
    assert torch.allclose(encoder.recorded_inputs[1], rollout_frame_1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(encoder.recorded_inputs[2], rollout_frame_2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(encoder.recorded_inputs[3], batch["motion"][:, 4], atol=1e-5, rtol=1e-5)
    assert rollout_calls == [
        {
            "batch_size": 1,
            "num_steps": 1,
            "time_schedule_power": float(config.inference_t_schedule_power),
            "initial_state_shape": (1, 68),
        },
        {
            "batch_size": 1,
            "num_steps": 1,
            "time_schedule_power": float(config.inference_t_schedule_power),
            "initial_state_shape": (1, 68),
        },
    ]
    assert predictor.recorded_batch_sizes == [3]


def test_incremental_flow_loss_runs_rollout_only_on_active_subset(
    monkeypatch,
) -> None:
    motion, joints = _load_motion_and_joints()
    batch = {
        "motion": torch.stack([motion[1:5], motion[1:5]], dim=0),
        "joints": torch.stack([joints[1:5], joints[1:5]], dim=0),
        "text_clip": torch.zeros(2, 1, 512),
    }

    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 2
    config.rollout_prob_start = 0.5
    config.rollout_prob_end = 0.5
    config.rollout_block_len_start = 2
    config.rollout_block_len_end = 2
    config.rollout_integration_steps = 1

    rand_outputs = iter(
        [
            torch.tensor([0.25, 0.75], dtype=torch.float32),
            torch.tensor([0.0, 0.9], dtype=torch.float32),
            torch.tensor([0.30, 0.60], dtype=torch.float32),
            torch.tensor([0.9, 0.9], dtype=torch.float32),
            torch.tensor([0.40, 0.20], dtype=torch.float32),
            torch.tensor([0.9, 0.9], dtype=torch.float32),
        ]
    )
    original_rand = torch.rand

    def fake_rand(*size, **kwargs):
        if len(size) == 1 and isinstance(size[0], int) and size[0] == 2:
            tensor = next(rand_outputs)
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", tensor.dtype)
            return tensor.to(device=device, dtype=dtype)
        return original_rand(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", fake_rand)
    rollout_batch_sizes: list[int] = []

    def fake_integrate_flow_ode(**kwargs):
        rollout_batch_sizes.append(int(kwargs["track_features"].shape[0]))
        return torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(train_utils, "integrate_flow_ode", fake_integrate_flow_ode)

    encoder = RecordingEncoder()
    predictor = RecordingPredictor(rollout_response="cancel")
    trainer = _make_trainer(config, encoder, predictor)

    trainer.incremental_flow_loss(batch=batch, epoch=1)

    assert rollout_batch_sizes == [1, 1]
    assert predictor.recorded_batch_sizes == [4]


def test_incremental_flow_loss_rollout_passes_integrate_flow_ode_args_for_active_subset(
    monkeypatch,
) -> None:
    motion, joints = _load_motion_and_joints()
    batch = {
        "motion": torch.stack([motion[1:5], motion[1:5]], dim=0),
        "joints": torch.stack([joints[1:5], joints[1:5]], dim=0),
        "text_clip": torch.zeros(2, 1, 512),
    }

    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 2
    config.rollout_prob_start = 0.5
    config.rollout_prob_end = 0.5
    config.rollout_block_len_start = 2
    config.rollout_block_len_end = 2
    config.rollout_integration_steps = 3
    config.inference_t_schedule_power = 4.0

    rand_outputs = iter(
        [
            torch.tensor([0.25, 0.75], dtype=torch.float32),
            torch.tensor([0.0, 0.9], dtype=torch.float32),
            torch.tensor([0.30, 0.60], dtype=torch.float32),
            torch.tensor([0.9, 0.9], dtype=torch.float32),
        ]
    )
    original_rand = torch.rand

    def fake_rand(*size, **kwargs):
        if len(size) == 1 and isinstance(size[0], int) and size[0] == 2:
            tensor = next(rand_outputs)
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", tensor.dtype)
            return tensor.to(device=device, dtype=dtype)
        return original_rand(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", fake_rand)

    rollout_calls: list[dict[str, object]] = []

    def fake_integrate_flow_ode(**kwargs):
        rollout_calls.append(
            {
                "batch_size": int(kwargs["track_features"].shape[0]),
                "num_steps": int(kwargs["num_steps"]),
                "time_schedule_power": float(kwargs["time_schedule_power"]),
                "initial_state_shape": tuple(kwargs["initial_state"].shape),
            }
        )
        return torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(train_utils, "integrate_flow_ode", fake_integrate_flow_ode)

    trainer = _make_trainer(
        config=config,
        encoder=RecordingEncoder(),
        predictor=RecordingPredictor(rollout_response="cancel"),
    )

    trainer.incremental_flow_loss(batch=batch, epoch=1)

    assert rollout_calls == [
        {
            "batch_size": 1,
            "num_steps": 3,
            "time_schedule_power": 4.0,
            "initial_state_shape": (1, 68),
        },
        {
            "batch_size": 1,
            "num_steps": 3,
            "time_schedule_power": 4.0,
            "initial_state_shape": (1, 68),
        },
    ]
