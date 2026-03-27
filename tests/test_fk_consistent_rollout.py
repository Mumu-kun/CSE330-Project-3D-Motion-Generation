import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from utils.motion_utils import (  # noqa: E402
    FeatureNormalizer,
    features_to_positions,
    generated_positions_to_271d,
    get_fk_offsets,
    sequence_joints_to_features,
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
        self.recorded_prev_relative_shifts: list[torch.Tensor] = []

    def forward(
        self,
        noised_tracks: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        track_features: torch.Tensor,
        prev_relative_shifts: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        if prev_relative_shifts is not None:
            self.recorded_prev_relative_shifts.append(
                prev_relative_shifts.detach().clone()
            )
        # One Euler step with dt=1 should land exactly at zero displacement.
        return -noised_tracks + self.anchor.view(1, 1, 1) * 0, None, None


class ZeroThenZeroEndpointPredictor(RecordingPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        noised_tracks: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        track_features: torch.Tensor,
        prev_relative_shifts: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        del text_embedding, track_features, output_attentions, output_hidden_states, kwargs
        if prev_relative_shifts is not None:
            self.recorded_prev_relative_shifts.append(
                prev_relative_shifts.detach().clone()
            )
        self.forward_calls += 1
        if self.forward_calls == 2:
            denom = (1 - timesteps.view(-1, 1, 1)).clamp(min=1e-6)
            return (-noised_tracks / denom) + self.anchor.view(1, 1, 1) * 0, None, None
        return torch.zeros_like(noised_tracks) + self.anchor.view(1, 1, 1) * 0, None, None


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


def test_incremental_flow_loss_rollout_writes_canonical_history() -> None:
    gt_features, gt_positions = _load_sample("000070")
    base_motion = gt_features[1:5]
    base_joints = gt_positions[1:5]

    translated_joints = base_joints + torch.tensor([1.25, 0.0, -0.5])

    motion = torch.stack([base_motion, base_motion], dim=0)
    joints = torch.stack([base_joints, translated_joints], dim=0)
    relative_shifts = torch.stack(
        [
            gt_positions[1:5] - gt_positions[0:4],
            gt_positions[1:5] - gt_positions[0:4],
        ],
        dim=0,
    )
    text_for_encoder = torch.zeros(2, 512)

    config = Config(device="cpu")
    config.use_fk = True
    config.rollout_prob_start = 0.5
    config.rollout_prob_end = 0.5
    config.rollout_integration_steps = 1
    config.use_consistency_loss = False

    encoder = RecordingEncoder()
    predictor = RecordingPredictor()
    trainer = Trainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    trainer.incremental_flow_loss(
        motion=motion,
        joints=joints,
        relative_shifts=relative_shifts,
        text_for_encoder=text_for_encoder,
        epoch=0,
        total_epochs=2,
        device="cpu",
        stochastic_rollout=False,
    )

    fk_offsets = get_fk_offsets(joints)
    current_positions_step0 = joints[:1, 0]
    expected_rollout_frame, _, expected_rollout_joints = generated_positions_to_271d(
        new_positions=current_positions_step0,
        prev_positions=current_positions_step0,
        dataset_type="t2m",
        fk_offsets=fk_offsets[:1],
        normalizer=trainer.normalizer,
    )

    assert expected_rollout_joints is not None
    assert len(encoder.recorded_inputs) == 3
    assert torch.allclose(
        encoder.recorded_inputs[1][:1], expected_rollout_frame, atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        encoder.recorded_inputs[1][1:2], motion[1:2, 1], atol=1e-5, rtol=1e-5
    )

    expected_prev_shift_step1 = expected_rollout_joints - current_positions_step0
    assert len(predictor.recorded_prev_relative_shifts) >= 2
    assert torch.allclose(
        predictor.recorded_prev_relative_shifts[1],
        expected_prev_shift_step1,
        atol=1e-4,
        rtol=1e-4,
    )


def test_rollout_guard_teacher_forces_only_degenerate_samples() -> None:
    gt_features, gt_positions = _load_sample("000070")
    zero_frame = torch.zeros_like(gt_features[0])
    zero_joints = torch.zeros_like(gt_positions[0])

    sample0_motion = torch.stack([zero_frame, gt_features[1], gt_features[2], gt_features[3]], dim=0)
    sample0_joints = torch.stack([zero_joints, gt_positions[1], gt_positions[2], gt_positions[3]], dim=0)
    sample1_motion = gt_features[1:5]
    sample1_joints = gt_positions[1:5]

    motion = torch.stack([sample0_motion, sample1_motion], dim=0)
    joints = torch.stack([sample0_joints, sample1_joints], dim=0)
    relative_shifts = torch.stack(
        [
            torch.zeros_like(sample0_joints),
            gt_positions[1:5] - gt_positions[0:4],
        ],
        dim=0,
    )
    text_for_encoder = torch.zeros(2, 512)

    config = Config(device="cpu")
    config.use_fk = True
    config.use_degenerate_pose_guard = True
    config.rollout_prob_start = 1.0
    config.rollout_prob_end = 1.0
    config.rollout_integration_steps = 1
    config.use_consistency_loss = False

    encoder = RecordingEncoder()
    predictor = RecordingPredictor()
    trainer = Trainer(
        encoder=encoder,
        predictor=predictor,
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    trainer.incremental_flow_loss(
        motion=motion,
        joints=joints,
        relative_shifts=relative_shifts,
        text_for_encoder=text_for_encoder,
        epoch=0,
        total_epochs=2,
        device="cpu",
        stochastic_rollout=False,
    )

    fk_offsets = get_fk_offsets(joints)
    expected_valid_frame, _, _ = generated_positions_to_271d(
        new_positions=joints[1:2, 0],
        prev_positions=joints[1:2, 0],
        dataset_type="t2m",
        fk_offsets=fk_offsets[1:2],
        normalizer=trainer.normalizer,
    )

    assert len(encoder.recorded_inputs) >= 2
    assert torch.allclose(encoder.recorded_inputs[1][0], motion[0, 1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        encoder.recorded_inputs[1][1:2], expected_valid_frame, atol=1e-4, rtol=1e-4
    )
    assert trainer.last_guard_stats["degenerate_rollout_count"] == 1.0
    assert trainer.last_guard_stats["degenerate_rollout_ratio"] == 0.25


def test_consistency_guard_masks_degenerate_samples() -> None:
    gt_features, gt_positions = _load_sample("000070")
    zero_frame = torch.zeros_like(gt_features[0])
    zero_joints = torch.zeros_like(gt_positions[0])

    motion = torch.stack(
        [
            torch.stack([zero_frame, zero_frame], dim=0),
            torch.stack([gt_features[1], gt_features[1]], dim=0),
        ],
        dim=0,
    )
    joints = torch.stack(
        [
            torch.stack([zero_joints, zero_joints], dim=0),
            torch.stack([gt_positions[1], gt_positions[1]], dim=0),
        ],
        dim=0,
    )
    relative_shifts = torch.zeros(2, 2, 22, 3)
    text_for_encoder = torch.zeros(2, 512)

    config = Config(device="cpu")
    config.use_fk = True
    config.use_degenerate_pose_guard = True
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    config.rollout_integration_steps = 1
    config.use_consistency_loss = True

    trainer = Trainer(
        encoder=RecordingEncoder(),
        predictor=ZeroThenZeroEndpointPredictor(),
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    _, _, consistency_loss, _, _ = trainer.incremental_flow_loss(
        motion=motion,
        joints=joints,
        relative_shifts=relative_shifts,
        text_for_encoder=text_for_encoder,
        epoch=0,
        total_epochs=2,
        device="cpu",
        stochastic_rollout=False,
    )

    assert torch.isclose(consistency_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
    assert trainer.last_guard_stats["degenerate_consistency_count"] == 1.0
    assert trainer.last_guard_stats["degenerate_consistency_ratio"] == 0.5


def test_consistency_guard_returns_zero_when_all_samples_are_degenerate() -> None:
    zero_frame = torch.zeros(271)
    zero_joints = torch.zeros(22, 3)
    motion = torch.stack(
        [
            torch.stack([zero_frame, zero_frame], dim=0),
            torch.stack([zero_frame, zero_frame], dim=0),
        ],
        dim=0,
    )
    joints = torch.stack(
        [
            torch.stack([zero_joints, zero_joints], dim=0),
            torch.stack([zero_joints, zero_joints], dim=0),
        ],
        dim=0,
    )
    relative_shifts = torch.zeros(2, 2, 22, 3)
    text_for_encoder = torch.zeros(2, 512)

    config = Config(device="cpu")
    config.use_fk = False
    config.use_degenerate_pose_guard = True
    config.rollout_prob_start = 0.0
    config.rollout_prob_end = 0.0
    config.use_consistency_loss = True

    trainer = Trainer(
        encoder=RecordingEncoder(),
        predictor=ZeroThenZeroEndpointPredictor(),
        dataloader=None,
        config=config,
        normalizer=_identity_normalizer(),
    )

    _, _, consistency_loss, _, _ = trainer.incremental_flow_loss(
        motion=motion,
        joints=joints,
        relative_shifts=relative_shifts,
        text_for_encoder=text_for_encoder,
        epoch=0,
        total_epochs=2,
        device="cpu",
        stochastic_rollout=False,
    )

    assert torch.isclose(consistency_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
    assert trainer.last_guard_stats["degenerate_consistency_count"] == 2.0
    assert trainer.last_guard_stats["degenerate_consistency_ratio"] == 1.0


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
