"""Tests for rollout scheduling with epoch-level warmup."""

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from utils.train_utils import Trainer


class DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


class DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def _make_trainer(config: Config) -> Trainer:
    dataset = TensorDataset(torch.zeros(1))
    dataloader = DataLoader(dataset, batch_size=1)
    return Trainer(
        encoder=DummyEncoder(),  # type: ignore[arg-type]
        predictor=DummyPredictor(),  # type: ignore[arg-type]
        dataloader=dataloader,
        config=config,
        normalizer=None,
    )


def test_config_clamps_rollout_warmup_fraction_into_valid_range() -> None:
    high = Config(device="cpu", rollout_warmup_fraction=3.5)
    low = Config(device="cpu", rollout_warmup_fraction=-0.25)

    assert 0.0 <= high.rollout_warmup_fraction < 1.0
    assert high.rollout_warmup_fraction == 1.0 - 1e-6
    assert low.rollout_warmup_fraction == 0.0


def test_config_clamps_t_sampling_controls_into_valid_range() -> None:
    config = Config(
        device="cpu",
        t_sampling_mode="PoWeR",
        t_sampling_power=-3.5,
        t_sampling_power_warmup_fraction=4.0,
    )

    assert config.t_sampling_mode == "power"
    assert config.t_sampling_power == 0.0
    assert config.t_sampling_power_warmup_fraction == 1.0


def test_rollout_schedule_applies_epoch_warmup_then_resumes_ramp() -> None:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 400
    config.rollout_prob_start = 0.2
    config.rollout_prob_end = 0.8
    config.rollout_warmup_fraction = 0.1
    trainer = _make_trainer(config)

    assert trainer._compute_rollout_probability(epoch=0) == 0.0
    assert trainer._compute_rollout_probability(epoch=39) == 0.0

    first_after_warmup = trainer._compute_rollout_probability(epoch=40)
    assert first_after_warmup > 0.0
    assert first_after_warmup > config.rollout_prob_start
    assert trainer._compute_rollout_probability(epoch=399) == config.rollout_prob_end


def test_rollout_schedule_matches_legacy_linear_ramp_when_warmup_disabled() -> None:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 5
    config.rollout_prob_start = 0.2
    config.rollout_prob_end = 0.6
    config.rollout_warmup_fraction = 0.0
    trainer = _make_trainer(config)

    for epoch in range(config.num_epochs):
        expected = config.rollout_prob_start + (
            (config.rollout_prob_end - config.rollout_prob_start)
            * (float(epoch) / float(config.num_epochs - 1))
        )
        assert math.isclose(
            trainer._compute_rollout_probability(epoch=epoch),
            expected,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


def test_rollout_schedule_handles_fixed_probability_and_single_epoch() -> None:
    fixed_config = Config(device="cpu", output_path=Path("./tests/output"))
    fixed_config.num_epochs = 10
    fixed_config.rollout_prob_start = 0.35
    fixed_config.rollout_prob_end = 0.35
    fixed_config.rollout_warmup_fraction = 0.2
    fixed_trainer = _make_trainer(fixed_config)

    assert fixed_trainer._compute_rollout_probability(epoch=0) == 0.0
    assert fixed_trainer._compute_rollout_probability(epoch=1) == 0.0
    assert math.isclose(
        fixed_trainer._compute_rollout_probability(epoch=2),
        0.35,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )

    single_epoch_config = Config(device="cpu", output_path=Path("./tests/output"))
    single_epoch_config.num_epochs = 1
    single_epoch_config.rollout_prob_end = 0.7
    single_epoch_config.rollout_warmup_fraction = 0.9
    single_epoch_trainer = _make_trainer(single_epoch_config)

    assert single_epoch_trainer._compute_rollout_probability(epoch=0) == 0.7


def test_config_clamps_rollout_block_length_range() -> None:
    config = Config(
        device="cpu",
        rollout_block_len_start=-3,
        rollout_block_len_end=0,
    )

    assert config.rollout_block_len_start == 1
    assert config.rollout_block_len_end == 1

    config = Config(
        device="cpu",
        rollout_block_len_start=4,
        rollout_block_len_end=2,
    )

    assert config.rollout_block_len_start == 4
    assert config.rollout_block_len_end == 4


def test_rollout_block_length_ramps_linearly_after_warmup() -> None:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 10
    config.rollout_warmup_fraction = 0.2
    config.rollout_block_len_start = 1
    config.rollout_block_len_end = 4
    trainer = _make_trainer(config)

    assert trainer._compute_rollout_block_length(epoch=0) == 1
    assert trainer._compute_rollout_block_length(epoch=1) == 1
    assert trainer._compute_rollout_block_length(epoch=2) == 1
    assert trainer._compute_rollout_block_length(epoch=5) == 2
    assert trainer._compute_rollout_block_length(epoch=6) == 3
    assert trainer._compute_rollout_block_length(epoch=9) == 4


def test_rollout_block_length_matches_linear_ramp_without_warmup() -> None:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 5
    config.rollout_warmup_fraction = 0.0
    config.rollout_block_len_start = 1
    config.rollout_block_len_end = 4
    trainer = _make_trainer(config)

    assert trainer._compute_rollout_block_length(epoch=0) == 1
    assert trainer._compute_rollout_block_length(epoch=1) == 2
    assert trainer._compute_rollout_block_length(epoch=2) == 3
    assert trainer._compute_rollout_block_length(epoch=3) == 3
    assert trainer._compute_rollout_block_length(epoch=4) == 4


def test_t_sampling_power_warmup_ramps_then_holds() -> None:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 9
    config.t_sampling_mode = "power"
    config.t_sampling_power = 2.0
    config.t_sampling_power_warmup_fraction = 0.5
    trainer = _make_trainer(config)

    assert trainer._compute_t_sampling_power(epoch=0) == 0.0
    assert math.isclose(
        trainer._compute_t_sampling_power(epoch=2),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert math.isclose(
        trainer._compute_t_sampling_power(epoch=4),
        2.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert math.isclose(
        trainer._compute_t_sampling_power(epoch=8),
        2.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_power_t_sampling_matches_uniform_when_effective_power_is_zero() -> None:
    config = Config(device="cpu", output_path=Path("./tests/output"))
    config.num_epochs = 100
    config.t_sampling_mode = "power"
    config.t_sampling_power = 2.0
    config.t_sampling_power_warmup_fraction = 0.25
    trainer = _make_trainer(config)

    torch.manual_seed(1234)
    sampled = trainer._sample_training_timesteps(
        batch_size=1024,
        device="cpu",
        dtype=torch.float32,
        epoch=0,
    )
    torch.manual_seed(1234)
    expected = torch.rand(1024, dtype=torch.float32)

    assert torch.allclose(sampled, expected)


def test_power_t_sampling_biases_samples_toward_high_values() -> None:
    uniform_config = Config(device="cpu", output_path=Path("./tests/output"))
    uniform_config.t_sampling_mode = "uniform"
    uniform_trainer = _make_trainer(uniform_config)

    power_config = Config(device="cpu", output_path=Path("./tests/output"))
    power_config.t_sampling_mode = "power"
    power_config.t_sampling_power = 2.0
    power_config.t_sampling_power_warmup_fraction = 0.0
    power_trainer = _make_trainer(power_config)

    torch.manual_seed(2026)
    uniform_t = uniform_trainer._sample_training_timesteps(
        batch_size=50000,
        device="cpu",
        dtype=torch.float32,
        epoch=0,
    )
    torch.manual_seed(2026)
    power_t = power_trainer._sample_training_timesteps(
        batch_size=50000,
        device="cpu",
        dtype=torch.float32,
        epoch=0,
    )

    uniform_tail = (uniform_t > 0.9).float().mean().item()
    power_tail = (power_t > 0.9).float().mean().item()

    assert power_t.mean().item() > uniform_t.mean().item()
    assert power_tail > uniform_tail
    assert uniform_tail < 0.12
    assert power_tail > 0.24
