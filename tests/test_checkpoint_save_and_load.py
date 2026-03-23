"""Checkpoint save-overhead and HumanMotionGenerator load-correctness tests."""

import os
import sys
import tempfile
import time
from typing import Dict

import numpy as np
import pytest
import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, FlowMatchingPredictorConfig
from models import FlowMatchingPredictor, HumanMotionGenerator, MotionHistoryEncoder
from utils.motion_utils import FeatureNormalizer
from utils.train_utils import EMAModel, Trainer


def _empty_dataloader() -> DataLoader:
    """Create a typed empty dataloader for Trainer construction in utility tests."""
    dataset = TensorDataset(torch.empty(0, 1))
    return DataLoader(dataset, batch_size=1)


def _seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _create_mock_normalizer() -> FeatureNormalizer:
    mean = np.zeros(271, dtype=np.float32)
    std = np.ones(271, dtype=np.float32)
    return FeatureNormalizer(
        torch.from_numpy(mean).float(),
        torch.from_numpy(std).float(),
    )


def _create_test_config() -> Config:
    config = Config(device="cpu")
    config.encoder_num_layers = 2
    config.encoder_hidden_dim = 128
    config.encoder_text_proj_dim = 64
    config.encoder_per_joint_dim = 32
    config.encoder_dropout = 0.0
    config.predictor_config = FlowMatchingPredictorConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_dropout=0.0,
        track_dimensionality=3,
        global_cond_dim=config.encoder_text_dim,
        head_dim=None,
    )
    return config


def _create_models(
    config: Config,
    normalizer: FeatureNormalizer,
) -> tuple[MotionHistoryEncoder, FlowMatchingPredictor]:
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
    ).to("cpu")

    predictor = FlowMatchingPredictor(
        feature_size=config.get_predictor_feature_size(),
        config=config.predictor_config,
        out_channels=3,
        use_relative_shift=True,
    ).to("cpu")

    return encoder, predictor


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state_dict.items()}


def _offset_float_tensors(
    state_dict: Dict[str, torch.Tensor],
    delta: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            out[k] = v + delta
        else:
            out[k] = v.clone()
    return out


def _assert_state_dict_close(
    got: Dict[str, torch.Tensor],
    expected: Dict[str, torch.Tensor],
) -> None:
    assert got.keys() == expected.keys()
    for key in got:
        g = got[key]
        e = expected[key]
        if torch.is_floating_point(g):
            assert torch.allclose(g, e, atol=1e-7, rtol=1e-6), f"Mismatch at {key}"
        else:
            assert torch.equal(g, e), f"Mismatch at {key}"


def _build_training_state() -> tuple[dict, dict]:
    curriculum_state = {
        "use_curriculum": False,
        "current_horizon": 5,
        "max_horizon": 5,
    }
    training_state = {
        "global_step": 0,
        "best_loss": float("inf"),
        "best_epoch": -1,
        "best_val_loss": float("inf"),
        "best_val_epoch": -1,
    }
    return curriculum_state, training_state


def _run_tiny_train_loop(with_checkpoint_saves: bool, steps: int = 4) -> float:
    _seed_all(7)
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, predictor = _create_models(config, normalizer)

    encoder.train()
    predictor.train()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scaler = GradScaler("cuda", enabled=False)
    encoder_ema = EMAModel(encoder, decay=0.999)
    predictor_ema = EMAModel(predictor, decay=0.999)

    curriculum_state, training_state = _build_training_state()
    trainer = Trainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=_empty_dataloader(),
        config=config,
        normalizer=normalizer,
    )

    B = 2
    T = 4
    history = torch.randn(B, T, 271)
    text_emb = torch.randn(B, 512)
    target_shift = torch.randn(B, 22, 3)

    with tempfile.TemporaryDirectory() as tmpdir:
        start = time.perf_counter()

        for step in range(steps):
            features = encoder(history, text_emb)
            t = torch.rand(B)
            x0 = torch.randn_like(target_shift)
            x_t = t.view(B, 1, 1) * target_shift + (1 - t.view(B, 1, 1)) * x0

            pred, _, _ = predictor(
                noised_tracks=x_t,
                timesteps=t,
                text_embedding=text_emb,
                track_features=features,
                relative_shifts=target_shift,
            )
            target_vel = target_shift - x0
            loss = torch.nn.functional.mse_loss(pred, target_vel)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            encoder_ema.update(encoder)
            predictor_ema.update(predictor)

            if with_checkpoint_saves:
                for save_idx in range(2):
                    trainer.save_training_checkpoint(
                        save_dir=tmpdir,
                        filename=f"step_{step}_{save_idx}.pt",
                        encoder_ema=encoder_ema,
                        predictor_ema=predictor_ema,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=step,
                        global_step=step,
                        loss=float(loss.item()),
                        curriculum_state=curriculum_state,
                        training_state=training_state,
                    )

        return time.perf_counter() - start


def test_checkpoint_save_latency_metrics() -> None:
    _seed_all(101)
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, predictor = _create_models(config, normalizer)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4
    )
    scaler = GradScaler("cuda", enabled=False)
    encoder_ema = EMAModel(encoder, decay=0.999)
    predictor_ema = EMAModel(predictor, decay=0.999)
    curriculum_state, training_state = _build_training_state()
    trainer = Trainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=_empty_dataloader(),
        config=config,
        normalizer=normalizer,
    )

    save_durations = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in range(5):
            t0 = time.perf_counter()
            trainer.save_training_checkpoint(
                save_dir=tmpdir,
                filename=f"latency_{idx}.pt",
                encoder_ema=encoder_ema,
                predictor_ema=predictor_ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=idx,
                global_step=idx,
                loss=1.0,
                curriculum_state=curriculum_state,
                training_state=training_state,
            )
            save_durations.append(time.perf_counter() - t0)

        ordered = sorted(save_durations)
        p95_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
        mean_latency = float(np.mean(save_durations))
        p95_latency = ordered[p95_index]

    print(
        "Checkpoint save latency metrics: "
        f"mean={mean_latency:.6f}s, p95={p95_latency:.6f}s"
    )
    assert all(d >= 0.0 for d in save_durations)
    assert mean_latency > 0.0
    assert p95_latency > 0.0


def test_checkpointing_overhead_vs_no_checkpointing() -> None:
    without_save = _run_tiny_train_loop(with_checkpoint_saves=False, steps=4)
    with_save = _run_tiny_train_loop(with_checkpoint_saves=True, steps=4)
    overhead_ratio = with_save / max(without_save, 1e-9)

    print(
        "Tiny train-loop timing: "
        f"without_save={without_save:.6f}s, with_save={with_save:.6f}s, "
        f"overhead_ratio={overhead_ratio:.3f}"
    )
    assert without_save > 0.0
    assert with_save > 0.0
    # Saving two checkpoints per step should introduce measurable overhead.
    assert overhead_ratio > 1.0


def test_load_from_checkpoint_prefers_ema_weights() -> None:
    _seed_all(11)
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, predictor = _create_models(config, normalizer)

    std_encoder = _clone_state_dict(encoder.state_dict())
    std_predictor = _clone_state_dict(predictor.state_dict())
    ema_encoder = _offset_float_tensors(std_encoder, delta=0.1)
    ema_predictor = _offset_float_tensors(std_predictor, delta=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "ema_preferred.pt")
        torch.save(
            {
                "encoder": std_encoder,
                "predictor": std_predictor,
                "encoder_ema": ema_encoder,
                "predictor_ema": ema_predictor,
                "config": config,
            },
            ckpt,
        )

        loaded = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path=ckpt,
            config=Config(device="cpu"),
            device="cpu",
            normalizer=normalizer,
        )

    _assert_state_dict_close(loaded.encoder.state_dict(), ema_encoder)
    _assert_state_dict_close(loaded.predictor.state_dict(), ema_predictor)


def test_load_from_checkpoint_falls_back_to_standard_weights() -> None:
    _seed_all(12)
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, predictor = _create_models(config, normalizer)

    std_encoder = _clone_state_dict(encoder.state_dict())
    std_predictor = _clone_state_dict(predictor.state_dict())

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "standard_only.pt")
        torch.save(
            {
                "encoder": std_encoder,
                "predictor": std_predictor,
                "config": config,
            },
            ckpt,
        )

        loaded = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path=ckpt,
            config=config,
            device="cpu",
            normalizer=normalizer,
        )

    _assert_state_dict_close(loaded.encoder.state_dict(), std_encoder)
    _assert_state_dict_close(loaded.predictor.state_dict(), std_predictor)


def test_load_from_checkpoint_uses_embedded_config() -> None:
    _seed_all(13)
    config_in_ckpt = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, predictor = _create_models(config_in_ckpt, normalizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "config_override.pt")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "config": config_in_ckpt,
            },
            ckpt,
        )

        different_passed_config = Config(device="cpu")
        different_passed_config.encoder_num_layers = 4
        different_passed_config.predictor_config.num_hidden_layers = 4

        loaded = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path=ckpt,
            config=different_passed_config,
            device="cpu",
            normalizer=normalizer,
        )

    assert loaded.config.encoder_num_layers == config_in_ckpt.encoder_num_layers
    assert (
        loaded.config.predictor_config.num_hidden_layers
        == config_in_ckpt.predictor_config.num_hidden_layers
    )


def test_load_from_checkpoint_missing_required_key_fails() -> None:
    _seed_all(14)
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, _ = _create_models(config, normalizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "missing_predictor.pt")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "config": config,
            },
            ckpt,
        )

        with pytest.raises(KeyError):
            HumanMotionGenerator.load_from_checkpoint(
                checkpoint_path=ckpt,
                config=config,
                device="cpu",
                normalizer=normalizer,
            )


def test_load_from_full_training_checkpoint_payload() -> None:
    _seed_all(15)
    config = _create_test_config()
    normalizer = _create_mock_normalizer()
    encoder, predictor = _create_models(config, normalizer)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scaler = GradScaler("cuda", enabled=False)
    encoder_ema = EMAModel(encoder, decay=0.999)
    predictor_ema = EMAModel(predictor, decay=0.999)
    curriculum_state, training_state = _build_training_state()
    trainer = Trainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=_empty_dataloader(),
        config=config,
        normalizer=normalizer,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_training_checkpoint(
            save_dir=tmpdir,
            filename="full_training.pt",
            encoder_ema=encoder_ema,
            predictor_ema=predictor_ema,
            optimizer=optimizer,
            scaler=scaler,
            epoch=0,
            global_step=3,
            loss=0.123,
            curriculum_state=curriculum_state,
            training_state=training_state,
        )

        loaded = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path=os.path.join(tmpdir, "full_training.pt"),
            config=config,
            device="cpu",
            normalizer=normalizer,
        )

    text_emb = torch.randn(1, 1, 512)
    input_positions = torch.randn(1, 3, 22, 3)
    with torch.no_grad():
        positions, features, rel_shifts = loaded.generate_sequence(
            text=text_emb,
            num_frames=2,
            num_steps=2,
            input_positions=input_positions,
        )

    assert positions.shape == (1, 5, 22, 3)
    assert features.shape == (1, 5, 271)
    assert rel_shifts.shape == (1, 5, 22, 3)
