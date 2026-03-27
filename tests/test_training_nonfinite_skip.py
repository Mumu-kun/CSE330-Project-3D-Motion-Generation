"""Regression coverage for skipping non-finite training batches safely."""

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from utils.train_utils import Trainer


class MinimalTrainingDataset(Dataset):
    def __init__(self, num_items: int = 2, seq_len: int = 3) -> None:
        self.num_items = num_items
        self.seq_len = seq_len
        self.last_horizon: int | None = None

    def set_horizon(self, horizon: int) -> None:
        self.last_horizon = horizon

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> dict:
        del idx
        return {
            "motion": torch.zeros(self.seq_len, 271, dtype=torch.float32),
            "joints": torch.zeros(self.seq_len, 22, 3, dtype=torch.float32),
            "text_clip": torch.zeros(1, 512, dtype=torch.float32),
            "lengths": torch.tensor(self.seq_len, dtype=torch.long),
        }


class CountingEMA:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.update_calls = 0

    def update(self, model: torch.nn.Module) -> None:
        del model
        self.update_calls += 1


class _ScaledLoss:
    def __init__(self, scaler: "CountingScaler", loss: torch.Tensor) -> None:
        self.scaler = scaler
        self.loss = loss

    def backward(self) -> None:
        self.scaler.backward_calls += 1
        self.loss.backward()


class CountingScaler:
    def __init__(self) -> None:
        self.backward_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss: torch.Tensor) -> _ScaledLoss:
        return _ScaledLoss(self, loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        self.unscale_calls += 1

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1


class NonFiniteOnceTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_calls = 0
        self.test_scaler: CountingScaler | None = None
        self.test_encoder_ema: CountingEMA | None = None
        self.test_predictor_ema: CountingEMA | None = None

    def setup_training_environment(self):
        optimizer = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=0.1,
        )
        scaler = CountingScaler()
        encoder_ema = CountingEMA(self.encoder)
        predictor_ema = CountingEMA(self.predictor)
        self.test_scaler = scaler
        self.test_encoder_ema = encoder_ema
        self.test_predictor_ema = predictor_ema
        training_state = {
            "global_step": 0,
            "best_loss": float("inf"),
            "best_epoch": -1,
            "best_val_loss": float("inf"),
            "best_val_epoch": -1,
        }
        self.encoder.train()
        self.predictor.train()
        return (
            "cpu",
            str(self.config.checkpoint_dir),
            None,
            encoder_ema,
            predictor_ema,
            optimizer,
            scaler,
            0,
            training_state,
            "cpu",
            False,
        )

    def incremental_flow_loss(self, *args, **kwargs):
        del args, kwargs
        self.loss_calls += 1
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        anchor = sum((param.sum() * 0 for param in params), start=torch.tensor(0.0))
        if self.loss_calls == 1:
            nan = anchor + torch.tensor(float("nan"))
            return nan, nan, nan, 1, 0.0
        return anchor + 1.0, anchor + 0.5, anchor + 0.25, 1, 0.0


def test_run_training_skips_nonfinite_batch_before_backward() -> None:
    config = Config(device="cpu")
    config.num_epochs = 1
    config.batch_size = 1
    config.horizon = 1
    config.curriculum = None
    config.checkpoint_interval = 0
    config.val_interval = 1000

    dataset = MinimalTrainingDataset(num_items=2, seq_len=3)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    encoder = torch.nn.Linear(1, 1)
    predictor = torch.nn.Linear(1, 1)

    trainer = NonFiniteOnceTrainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=dataloader,
        config=config,
        normalizer=None,
    )

    trainer._run_training()

    assert dataset.last_horizon == 3
    assert trainer.loss_calls == 2
    assert trainer.test_scaler is not None
    assert trainer.test_scaler.backward_calls == 1
    assert trainer.test_scaler.unscale_calls == 1
    assert trainer.test_scaler.step_calls == 1
    assert trainer.test_scaler.update_calls == 1
    assert trainer.test_encoder_ema is not None
    assert trainer.test_predictor_ema is not None
    assert trainer.test_encoder_ema.update_calls == 1
    assert trainer.test_predictor_ema.update_calls == 1
