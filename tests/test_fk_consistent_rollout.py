import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
import utils.train_utils as train_utils
from utils.motion_utils import (  # noqa: E402
    FeatureNormalizer,
    extract_prev_frame_features,
    features_to_positions,
    flow_output_to_positions,
    generated_positions_to_271d,
    get_fk_offsets,
    sequence_joints_to_features,
    subset_271d_to_72d,
)
from utils.train_utils import Trainer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _identity_normalizer() -> FeatureNormalizer:
    return FeatureNormalizer(torch.zeros(271), torch.ones(271))


def _load_sample(sample_id: str = "000070") -> tuple[torch.Tensor, torch.Tensor]:
    gt_features = torch.from_numpy(
        np.load(ROOT / "sample_data" / f"{sample_id}_vec.npy")
    ).float()
    gt_positions = features_to_positions(gt_features)
    return gt_features, gt_positions


class RecordingEncoder(torch.nn.Module):
    def __init__(self, joint_count: int = 22, feature_dim: int = 8) -> None:
        super().__init__()
        self.joint_count = joint_count
        self.feature_dim = feature_dim
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.recorded_inputs: list[torch.Tensor] = []

    def gru_step(
        self,
        x_t: torch.Tensor,
        text_emb: torch.Tensor,
        h: torch.Tensor | None,
        use_normalization: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return context + self.anchor.view(1, 1, 1) * 0, h_next


class RecordingPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.flow_dim = 68
        self.recorded_current_frame_features: list[torch.Tensor] = []
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
        del timesteps, text_embedding, track_features, output_attentions, output_hidden_states, kwargs
        self.recorded_batch_sizes.append(int(noisy_features.shape[0]))
        self.recorded_current_frame_features.append(
            current_frame_features.detach().clone()
        )
        return -noisy_features + self.anchor.view(1, 1) * 0, None, None


class ZeroPredictor(RecordingPredictor):
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
        del timesteps, text_embedding, track_features, output_attentions, output_hidden_states, kwargs
        self.recorded_batch_sizes.append(int(noisy_features.shape[0]))
        self.recorded_current_frame_features.append(
            current_frame_features.detach().clone()
        )
        return (
            torch.zeros_like(noisy_features) + self.anchor.view(1, 1) * 0,
            None,
            None,
        )


class ZeroThenZeroEndpointPredictor(RecordingPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

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
        del text_embedding, track_features, output_attentions, output_hidden_states, kwargs
        self.recorded_batch_sizes.append(int(noisy_features.shape[0]))
        self.recorded_current_frame_features.append(
            current_frame_features.detach().clone()
        )
        self.forward_calls += 1
        if self.forward_calls == 2:
            denom = (1 - timesteps.view(-1, 1)).clamp(min=1e-6)
            return (
                (-noisy_features / denom) + self.anchor.view(1, 1) * 0,
                None,
                None,
            )
        return (
            torch.zeros_like(noisy_features) + self.anchor.view(1, 1) * 0,
            None,
            None,
        )


def _make_trainer(
    predictor: torch.nn.Module | None = None, config: Config | None = None
) -> Trainer:
    if config is None:
        config = Config(device="cpu")
    return Trainer(
        encoder=RecordingEncoder(),
        predictor=RecordingPredictor() if predictor is None else predictor,
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )


def test_generated_positions_to_271d_returns_fk_canonical_frame() -> None:
    _, gt_positions = _load_sample("000070")
    prev_positions = gt_positions[9:10]
    new_positions = gt_positions[10:11].clone()
    new_positions[:, 5, 0] += 0.15
    new_positions[:, 8, 2] -= 0.08

    fk_offsets = get_fk_offsets(gt_positions.unsqueeze(0))

    frame_out, _, fk_positions = generated_positions_to_271d(
        new_positions=new_positions,
        prev_positions=prev_positions,
        dataset_type="t2m",
        fk_offsets=fk_offsets,
        normalizer=None,
    )

    assert fk_positions is not None

    expected_frame = sequence_joints_to_features(
        torch.cat([prev_positions, fk_positions], dim=0), dataset_type="t2m"
    )[1:2]
    raw_frame = sequence_joints_to_features(
        torch.cat([prev_positions, new_positions], dim=0), dataset_type="t2m"
    )[1:2]

    assert torch.allclose(frame_out, expected_frame, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(frame_out, raw_frame, atol=1e-5, rtol=1e-5)


def test_detect_degenerate_pose_mask_accepts_valid_pose() -> None:
    _, gt_positions = _load_sample("000070")
    trainer = _make_trainer()

    prev_positions = gt_positions[10:11]
    new_positions = prev_positions.clone()
    reference_shifts = torch.zeros_like(prev_positions)

    mask = trainer.detect_degenerate_pose_mask(
        new_positions=new_positions,
        prev_positions=prev_positions,
        fk_offsets=get_fk_offsets(gt_positions.unsqueeze(0)),
        reference_shifts=reference_shifts,
    )

    assert torch.equal(mask, torch.tensor([False]))


def test_detect_degenerate_pose_mask_flags_nonfinite_positions() -> None:
    _, gt_positions = _load_sample("000070")
    trainer = _make_trainer()

    prev_positions = gt_positions[10:11]
    new_positions = prev_positions.clone()
    new_positions[0, 0, 0] = float("nan")

    mask = trainer.detect_degenerate_pose_mask(
        new_positions=new_positions,
        prev_positions=prev_positions,
        fk_offsets=get_fk_offsets(gt_positions.unsqueeze(0)),
        reference_shifts=torch.zeros_like(prev_positions),
    )

    assert torch.equal(mask, torch.tensor([True]))


def test_detect_degenerate_pose_mask_flags_collapsed_bone() -> None:
    _, gt_positions = _load_sample("000070")
    trainer = _make_trainer()

    prev_positions = gt_positions[10:11]
    new_positions = prev_positions.clone()
    new_positions[:, 5] = new_positions[:, 2]

    mask = trainer.detect_degenerate_pose_mask(
        new_positions=new_positions,
        prev_positions=prev_positions,
        fk_offsets=get_fk_offsets(gt_positions.unsqueeze(0)),
        reference_shifts=torch.zeros_like(prev_positions),
    )

    assert torch.equal(mask, torch.tensor([True]))


def test_detect_degenerate_pose_mask_flags_tiny_across_vector() -> None:
    _, gt_positions = _load_sample("000070")
    trainer = _make_trainer()

    prev_positions = gt_positions[10:11]
    new_positions = prev_positions.clone()
    new_positions[:, 1] = new_positions[:, 2]
    new_positions[:, 16] = new_positions[:, 17]

    mask = trainer.detect_degenerate_pose_mask(
        new_positions=new_positions,
        prev_positions=prev_positions,
        fk_offsets=get_fk_offsets(gt_positions.unsqueeze(0)),
        reference_shifts=torch.zeros_like(prev_positions),
    )

    assert torch.equal(mask, torch.tensor([True]))


def test_detect_degenerate_pose_mask_flags_absurd_step() -> None:
    _, gt_positions = _load_sample("000070")
    trainer = _make_trainer()

    prev_positions = gt_positions[10:11]
    new_positions = prev_positions.clone()
    new_positions[:, 8] = new_positions[:, 8] + torch.tensor([50.0, 0.0, 0.0])

    mask = trainer.detect_degenerate_pose_mask(
        new_positions=new_positions,
        prev_positions=prev_positions,
        fk_offsets=get_fk_offsets(gt_positions.unsqueeze(0)),
        reference_shifts=torch.zeros_like(prev_positions),
    )

    assert torch.equal(mask, torch.tensor([True]))


def test_incremental_flow_loss_rollout_writes_generated_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gt_features, gt_positions = _load_sample("000070")
    base_motion = gt_features[1:5]
    base_joints = gt_positions[1:5]

    translated_joints = base_joints + torch.tensor([1.25, 0.0, -0.5])

    motion = torch.stack([base_motion, base_motion], dim=0)
    joints = torch.stack([base_joints, translated_joints], dim=0)
    text_for_encoder = torch.zeros(2, 512)

    config = Config(device="cpu")
    config.num_epochs = 2
    config.use_fk = False
    config.rollout_prob_start = 0.5
    config.rollout_prob_end = 0.5
    config.rollout_integration_steps = 1
    config.use_consistency_loss = False

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
    rollout_frame_features: list[torch.Tensor] = []

    def fake_integrate_flow_ode(**kwargs):
        rollout_frame_features.append(
            kwargs["current_frame_features"].detach().clone()
        )
        return torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(train_utils, "integrate_flow_ode", fake_integrate_flow_ode)

    encoder = RecordingEncoder()
    predictor = RecordingPredictor()
    trainer = Trainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    batch = {
        "motion": motion,
        "joints": joints,
        "text_clip": text_for_encoder.unsqueeze(1),
    }

    trainer.incremental_flow_loss(batch=batch, epoch=1)

    current_positions_step0 = joints[:1, 1]
    zero_flow = torch.zeros(1, 68)
    rollout_positions = flow_output_to_positions(
        zero_flow,
        prev_root_pos=current_positions_step0[:, 0],
        prev_root_rot_6d=motion[:1, 1, 69:75],
    )
    expected_rollout_frame, _, _ = generated_positions_to_271d(
        new_positions=rollout_positions,
        prev_positions=current_positions_step0,
        dataset_type="t2m",
        normalizer=trainer.normalizer,
    )

    assert len(encoder.recorded_inputs) == 3
    assert torch.allclose(
        encoder.recorded_inputs[1][:1], expected_rollout_frame, atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        encoder.recorded_inputs[1][1:2], motion[1:2, 2], atol=1e-5, rtol=1e-5
    )
    assert len(rollout_frame_features) == 2
    assert torch.allclose(
        rollout_frame_features[1],
        extract_prev_frame_features(expected_rollout_frame),
        atol=1e-6,
        rtol=1e-6,
    )

def test_incremental_flow_loss_batches_main_flow_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gt_features, gt_positions = _load_sample("000070")
    motion = torch.stack([gt_features[1:5], gt_features[1:5]], dim=0)
    joints = torch.stack([gt_positions[1:5], gt_positions[1:5]], dim=0)
    text_for_encoder = torch.zeros(2, 512)

    config = Config(device="cpu")
    config.num_epochs = 2
    config.use_fk = False
    config.rollout_prob_start = 0.5
    config.rollout_prob_end = 0.5
    config.rollout_integration_steps = 1
    config.use_consistency_loss = False

    predictor = RecordingPredictor()
    trainer = Trainer(
        encoder=RecordingEncoder(),
        predictor=predictor,
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    batch = {
        "motion": motion,
        "joints": joints,
        "text_clip": text_for_encoder.unsqueeze(1),
    }

    rollout_rand_outputs = iter(
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
            tensor = next(rollout_rand_outputs)
            device = kwargs.get("device")
            dtype = kwargs.get("dtype", tensor.dtype)
            return tensor.to(device=device, dtype=dtype)
        return original_rand(*size, **kwargs)

    def fake_integrate_flow_ode(**kwargs):
        return torch.zeros_like(kwargs["initial_state"])

    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(train_utils, "integrate_flow_ode", fake_integrate_flow_ode)
    trainer.incremental_flow_loss(batch=batch, epoch=1)

    pred_steps = batch["joints"].shape[1] - 2
    expected_batched_size = motion.shape[0] * pred_steps
    assert predictor.recorded_batch_sizes.count(expected_batched_size) == 1
    assert predictor.recorded_batch_sizes == [expected_batched_size]


def test_incremental_flow_loss_rollout_targets_ground_truth_from_effective_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gt_features, gt_positions = _load_sample("000070")
    motion = gt_features[1:5].unsqueeze(0)
    joints = gt_positions[1:5].unsqueeze(0)
    text_for_encoder = torch.zeros(1, 512)

    config = Config(device="cpu")
    config.num_epochs = 1
    config.use_fk = False
    config.rollout_prob_start = 1.0
    config.rollout_prob_end = 1.0
    config.rollout_integration_steps = 1
    config.use_consistency_loss = False

    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    trainer = Trainer(
        encoder=RecordingEncoder(),
        predictor=ZeroPredictor(),
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    batch = {
        "motion": motion,
        "joints": joints,
        "text_clip": text_for_encoder.unsqueeze(1),
    }

    _, flow_loss, _, pred_steps, _ = trainer.incremental_flow_loss(batch=batch, epoch=0)

    rollout_positions = flow_output_to_positions(
        torch.zeros(1, 68),
        prev_root_pos=joints[:, 1, 0],
        prev_root_rot_6d=motion[:, 1, 69:75],
    )
    rollout_frame, _, _ = generated_positions_to_271d(
        new_positions=rollout_positions,
        prev_positions=joints[:, 1],
        dataset_type="t2m",
        normalizer=trainer.normalizer,
    )
    expected_targets = torch.stack(
        [
            subset_271d_to_72d(motion[:, 2], prev_frame=motion[:, 1]),
            subset_271d_to_72d(motion[:, 3], prev_frame=rollout_frame),
        ],
        dim=1,
    )
    expected_flow_loss = expected_targets.square().mean()
    assert torch.isclose(flow_loss, expected_flow_loss, atol=1e-6, rtol=1e-6)
    assert pred_steps == expected_targets.shape[1]


def test_incremental_flow_loss_returns_zero_consistency_for_now() -> None:
    gt_features, gt_positions = _load_sample("000070")
    motion = gt_features[1:4].unsqueeze(0)
    joints = gt_positions[1:4].unsqueeze(0)
    text_for_encoder = torch.zeros(1, 512)

    config = Config(device="cpu")
    config.num_epochs = 1
    config.use_fk = False
    config.rollout_prob_start = 1.0
    config.rollout_prob_end = 1.0
    config.rollout_integration_steps = 1
    config.use_consistency_loss = True
    config.consistency_loss_weight = 2.5

    trainer = Trainer(
        encoder=RecordingEncoder(),
        predictor=ZeroPredictor(),
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    batch = {
        "motion": motion,
        "joints": joints,
        "text_clip": text_for_encoder.unsqueeze(1),
    }

    total_loss, flow_loss, consistency_loss, _, _ = trainer.incremental_flow_loss(
        batch=batch,
        epoch=0,
    )

    assert torch.isclose(consistency_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
    assert torch.isclose(total_loss, flow_loss, atol=1e-6, rtol=1e-6)


def test_generated_positions_to_271d_stays_finite_for_degenerate_poses() -> None:
    _, gt_positions = _load_sample("000070")
    batch_size = 3
    prev_positions = torch.zeros(batch_size, 22, 3)
    new_positions = torch.zeros(batch_size, 22, 3)

    # Symmetric/collapsed structures that previously produced zero quaternions.
    new_positions[1, 1] = torch.tensor([1.0, 0.0, 0.0])
    new_positions[1, 2] = torch.tensor([1.0, 0.0, 0.0])
    new_positions[2, 0] = torch.tensor([0.0, 1.0, 0.0])
    new_positions[2, 1] = torch.tensor([0.0, 1.0, 0.0])
    new_positions[2, 2] = torch.tensor([0.0, 1.0, 0.0])

    fk_offsets = get_fk_offsets(gt_positions.unsqueeze(0)).expand(batch_size, -1, -1)

    frame_out, root_out, fk_positions = generated_positions_to_271d(
        new_positions=new_positions,
        prev_positions=prev_positions,
        dataset_type="t2m",
        fk_offsets=fk_offsets,
        normalizer=None,
    )

    assert torch.isfinite(frame_out).all()
    assert torch.isfinite(root_out).all()
    assert fk_positions is not None
    assert torch.isfinite(fk_positions).all()
