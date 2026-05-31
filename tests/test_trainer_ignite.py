"""Tests for IgniteMotionTrainer in src/utils/trainer_ignite.py."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, FlowMatchingPredictorConfig
from models import FlowMatchingPredictor, MotionHistoryEncoder
from utils.motion_utils import FeatureNormalizer
from utils.trainer_ignite import (
    EMAModel,
    EngineStateCheckpointProxy,
    IgniteMotionTrainer,
    TrainingPhase,
    TrainingStrategy,
    PretrainStrategy,
    FinetuneStrategy,
    TwoStepMotionTrainer,
    build_ignite_trainer,
)


def _create_test_config() -> Config:
    """Create minimal config for fast CPU testing."""
    config = Config(device="cpu")
    config.encoder_config.hidden_size = 32
    config.encoder_config.intermediate_size = 64
    config.encoder_config.per_joint_output_dim = 8
    config.encoder_config.num_hidden_layers = 1
    config.encoder_config.dropout = 0.0
    config.encoder_config.attention_dropout = 0.0
    config.predictor_config = FlowMatchingPredictorConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        attention_dropout=0.0,
        track_dimensionality=3,
        global_cond_dim=config.encoder_config.text_embedding_dim,
    )
    config.num_epochs = 1
    config.batch_size = 2
    config.learning_rate = 1e-4
    config.weight_decay = 1e-5
    config.ema_decay = 0.999
    config.horizon = 4
    return config


def _create_mock_normalizer() -> FeatureNormalizer:
    """Create a simple normalizer for testing."""
    mean = torch.zeros(271, dtype=torch.float32)
    std = torch.ones(271, dtype=torch.float32)
    return FeatureNormalizer(mean, std)


def _create_dataloader(num_samples: int = 4) -> DataLoader:
    """Create a dataloader with correct batch structure for trainer."""
    motion = torch.randn(num_samples, 4, 271)
    text_clip = torch.randn(num_samples, 1, 512)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, motions, texts):
            self.motions = motions
            self.texts = texts

        def __len__(self):
            return len(self.motions)

        def __getitem__(self, idx):
            return {"motion": self.motions[idx], "text_clip": self.texts[idx]}

    dataset = DictDataset(motion, text_clip)
    return DataLoader(dataset, batch_size=2)


def test_import_trainer_ignite() -> None:
    """Verify module imports without errors."""
    assert IgniteMotionTrainer is not None
    assert EMAModel is not None
    assert EngineStateCheckpointProxy is not None
    assert build_ignite_trainer is not None


def test_ema_model_update() -> None:
    """Test EMAModel wrapper updates correctly."""
    config = _create_test_config()
    model = MotionHistoryEncoder(config.encoder_config)

    ema = EMAModel(model, decay=0.999)
    first_param = next(model.parameters())
    old_weight = first_param.data.clone()

    first_param.data.fill_(1.0)
    ema.update(model)

    ema_param = next(ema.model.parameters())
    assert not torch.allclose(ema_param.data, old_weight)
    assert ema_param.data.mean() > old_weight.mean()


def test_engine_state_checkpoint_proxy() -> None:
    """Test state save/load for checkpoint proxy."""

    class MockState:
        def __init__(self):
            self.global_step = 10
            self.epoch = 2
            self.current_horizon = 5
            self.last_train_loss = 0.5
            self.last_val_loss = 0.3
            self.best_train_loss = 0.4
            self.best_val_loss = 0.25
            self.latest_metrics = {"loss": 0.5}

    mock_state = MockState()
    proxy = EngineStateCheckpointProxy(mock_state)

    snapshot = proxy.state_dict()
    assert snapshot["global_step"] == 10
    assert snapshot["epoch"] == 2

    new_mock = MockState()
    new_mock.global_step = 0
    new_mock.epoch = 0
    proxy2 = EngineStateCheckpointProxy(new_mock)
    proxy2.load_state_dict(snapshot)

    assert proxy2.engine_state.global_step == 10
    assert proxy2.engine_state.epoch == 2


def test_train_step_private() -> None:
    """Test _train_step executes without errors."""
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder = MotionHistoryEncoder(config.encoder_config)
    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
    )

    train_loader = _create_dataloader(num_samples=2)

    trainer = IgniteMotionTrainer(
        encoder=encoder,
        predictor=predictor,
        config=config,
        normalizer=normalizer,
        dataloader=train_loader,
    )

    engine = trainer.build_train_engine()
    batch = next(iter(train_loader))
    output = trainer._train_step(engine, batch)

    assert "loss" in output
    assert not torch.isnan(output["loss"])


def test_val_step_private() -> None:
    """Test _val_step executes without errors."""
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder = MotionHistoryEncoder(config.encoder_config)
    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
    )

    val_loader = _create_dataloader(num_samples=2)

    trainer = IgniteMotionTrainer(
        encoder=encoder,
        predictor=predictor,
        config=config,
        normalizer=normalizer,
        val_dataloader=val_loader,
    )

    engine = trainer.build_validation_engine()
    batch = next(iter(val_loader))
    output = trainer._val_step(engine, batch)

    assert "val_loss" in output
    assert not torch.isnan(output["val_loss"])


def test_build_engines() -> None:
    """Test engine construction."""
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder = MotionHistoryEncoder(config.encoder_config)
    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
    )

    trainer = IgniteMotionTrainer(
        encoder=encoder,
        predictor=predictor,
        config=config,
        normalizer=normalizer,
    )

    train_engine = trainer.build_train_engine()
    assert train_engine is not None
    assert trainer.trainer is train_engine

    val_engine = trainer.build_validation_engine()
    assert val_engine is not None
    assert trainer.evaluator is val_engine


def test_full_training_run() -> None:
    """Test complete training run with checkpointing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _create_test_config()

        normalizer = _create_mock_normalizer()
        encoder = MotionHistoryEncoder(config.encoder_config)
        predictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
        )

        train_loader = _create_dataloader(num_samples=2)
        val_loader = _create_dataloader(num_samples=2)

        trainer = IgniteMotionTrainer(
            encoder=encoder,
            predictor=predictor,
            config=config,
            normalizer=normalizer,
            checkpoint_dir=tmpdir,
            log_to_console=False,
        )

        trainer.run(train_dataloader=train_loader, val_dataloader=val_loader, max_epochs=1)

        assert trainer.state is not None
        assert trainer.state.epoch == 1


def test_build_ignite_trainer_factory() -> None:
    """Test factory function creates trainer correctly."""
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder = MotionHistoryEncoder(config.encoder_config)
    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
    )

    trainer = build_ignite_trainer(
        encoder=encoder,
        predictor=predictor,
        config=config,
        normalizer=normalizer,
    )

    assert isinstance(trainer, IgniteMotionTrainer)
    assert trainer.log_to_console is True
    assert trainer.use_ema_for_validation is True


def test_training_phase_enum() -> None:
    """Test TrainingPhase enum values."""
    assert TrainingPhase.PRETRAINING.value == "pretrain"
    assert TrainingPhase.FINETUNING.value == "finetune"


def test_pretrain_strategy_defaults() -> None:
    """Test PretrainStrategy default values."""
    strategy = PretrainStrategy()
    assert strategy.learning_rate == 1e-4
    assert strategy.weight_decay == 1e-5
    assert strategy.num_epochs == 100
    assert strategy.ema_decay == 0.999
    assert strategy.get_checkpoint_prefix() == "pretrain_latest"


def test_pretrain_strategy_custom() -> None:
    """Test PretrainStrategy custom values."""
    strategy = PretrainStrategy(
        learning_rate=5e-5,
        weight_decay=1e-6,
        num_epochs=50,
        ema_decay=0.99,
    )
    assert strategy.learning_rate == 5e-5
    assert strategy.weight_decay == 1e-6
    assert strategy.num_epochs == 50
    assert strategy.ema_decay == 0.99


def test_finetune_strategy_defaults() -> None:
    """Test FinetuneStrategy default values."""
    strategy = FinetuneStrategy()
    assert strategy.learning_rate == 5e-5
    assert strategy.weight_decay == 1e-5
    assert strategy.num_epochs == 50
    assert strategy.ema_decay == 0.999
    assert strategy.get_checkpoint_prefix() == "finetune_latest"


def test_two_step_trainer_init() -> None:
    """Test TwoStepMotionTrainer initialization."""
    config = _create_test_config()
    encoder = MotionHistoryEncoder(config.encoder_config)
    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
    )

    pretrain_strategy = PretrainStrategy()
    finetune_strategy = FinetuneStrategy()

    trainer = TwoStepMotionTrainer(
        encoder=encoder,
        predictor=predictor,
        config=config,
        pretrain_strategy=pretrain_strategy,
        finetune_strategy=finetune_strategy,
    )

    assert trainer.pretrain_strategy is pretrain_strategy
    assert trainer.finetune_strategy is finetune_strategy
    assert trainer._current_phase is None


def test_two_step_trainer_save_checkpoint() -> None:
    """Test IgniteMotionTrainer save_checkpoint method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _create_test_config()
        normalizer = _create_mock_normalizer()
        encoder = MotionHistoryEncoder(config.encoder_config)
        predictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
        )

        trainer = IgniteMotionTrainer(
            encoder=encoder,
            predictor=predictor,
            config=config,
            normalizer=normalizer,
            checkpoint_dir=tmpdir,
            log_to_console=False,
        )

        trainer.build_train_engine()
        checkpoint_path = trainer.save_checkpoint("test_checkpoint")

        assert Path(checkpoint_path).exists()
        state = torch.load(checkpoint_path, weights_only=False)
        assert "encoder" in state
        assert "predictor" in state
        assert "encoder_ema" in state
        assert "predictor_ema" in state


def test_two_step_trainer_load_checkpoint() -> None:
    """Test IgniteMotionTrainer load_checkpoint method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _create_test_config()
        normalizer = _create_mock_normalizer()
        encoder = MotionHistoryEncoder(config.encoder_config)
        predictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
        )

        trainer = IgniteMotionTrainer(
            encoder=encoder,
            predictor=predictor,
            config=config,
            normalizer=normalizer,
            checkpoint_dir=tmpdir,
            log_to_console=False,
        )

        trainer.build_train_engine()
        trainer._train_step(trainer.trainer, next(iter(_create_dataloader())))
        original_weight = next(encoder.parameters()).data.clone()

        trainer2 = IgniteMotionTrainer(
            encoder=encoder,
            predictor=predictor,
            config=config,
            normalizer=normalizer,
            checkpoint_dir=tmpdir,
            log_to_console=False,
        )

        checkpoint_path = trainer.save_checkpoint("test_load")
        trainer2.build_train_engine()
        trainer2.load_checkpoint(checkpoint_path)

        loaded_weight = next(encoder.parameters()).data.clone()
        assert torch.allclose(loaded_weight, original_weight)


def test_two_step_trainer_pretrain_run() -> None:
    """Test TwoStepMotionTrainer pretrain phase execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _create_test_config()
        config.num_epochs = 1
        normalizer = _create_mock_normalizer()
        encoder = MotionHistoryEncoder(config.encoder_config)
        predictor = FlowMatchingPredictor(
            feature_size=config.get_predictor_feature_size(),
            config=config.predictor_config,
        )

        train_loader = _create_dataloader(num_samples=2)
        val_loader = _create_dataloader(num_samples=2)

        pretrain_strategy = PretrainStrategy(num_epochs=1, learning_rate=1e-4)
        finetune_strategy = FinetuneStrategy()

        trainer = TwoStepMotionTrainer(
            encoder=encoder,
            predictor=predictor,
            config=config,
            pretrain_strategy=pretrain_strategy,
            finetune_strategy=finetune_strategy,
            checkpoint_dir=tmpdir,
            log_to_console=False,
        )

        trainer.run_pretrain(train_dataloader=train_loader, val_dataloader=val_loader, max_epochs=1)

        assert trainer._current_phase == TrainingPhase.PRETRAINING
        assert trainer.pretrain_checkpoint_path.exists()