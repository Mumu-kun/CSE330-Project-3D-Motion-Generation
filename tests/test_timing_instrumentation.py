"""Unit tests for optional timing instrumentation in Trainer."""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from utils.train_utils import TimingStats, Trainer, timer


def _make_dummy_trainer(enable_profiling: bool) -> Trainer:
    """Create a Trainer instance with minimal dependencies for unit tests."""
    config = Config()
    config.enable_profiling = enable_profiling

    encoder = nn.Identity()
    predictor = nn.Identity()
    dataset = TensorDataset(torch.zeros(2, 1))
    dataloader = DataLoader(dataset, batch_size=1)

    return Trainer(
        encoder=encoder,  # type: ignore[arg-type]
        predictor=predictor,  # type: ignore[arg-type]
        dataloader=dataloader,
        config=config,
    )


def test_trainer_profiling_toggle_initializes_timing_stats() -> None:
    """Trainer should only allocate TimingStats when profiling is enabled."""
    trainer_off = _make_dummy_trainer(enable_profiling=False)
    assert trainer_off.timing_stats is None

    trainer_on = _make_dummy_trainer(enable_profiling=True)
    assert trainer_on.timing_stats is not None
    assert isinstance(trainer_on.timing_stats, TimingStats)


def test_timer_records_expected_forward_breakdown_keys() -> None:
    """Timer utility should aggregate key timings and produce stable key names."""
    stats = TimingStats()
    keys = [
        "data_load",
        "text_prep",
        "forward",
        "forward/encoder_contexts",
        "forward/predictor",
        "forward/rollout_ode",
        "forward/rollout_ode_step",
        "forward/consistency_pred",
        "forward/pos_transform",
        "backward",
        "ema_update",
        "validation",
        "checkpoint",
    ]

    for key in keys:
        with timer(stats, key):
            _ = 1 + 1

    averages = stats.get_averages()
    assert set(keys).issubset(set(averages.keys()))
    assert all(value >= 0.0 for value in averages.values())

    per_step_payload = {f"time/{key}_ms": val for key, val in averages.items()}
    epoch_payload = {f"epoch_time/{key}_ms": val for key, val in averages.items()}

    assert "time/forward/encoder_contexts_ms" in per_step_payload
    assert "time/forward/rollout_ode_step_ms" in per_step_payload
    assert "epoch_time/forward/predictor_ms" in epoch_payload

    summary = str(stats)
    assert "=== Timing Summary ===" in summary
    assert "Total:" in summary


def test_timer_noop_when_stats_is_none() -> None:
    """timer(None, key) should be a no-op and never raise."""
    with timer(None, "forward"):
        value = 41 + 1
    assert value == 42
