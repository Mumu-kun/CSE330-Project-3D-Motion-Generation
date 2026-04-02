import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import (  # noqa: E402
    FeatureNormalizer,
    extract_prev_frame_features,
    flow_output_to_positions,
    generated_positions_to_271d,
    root_rot6d_to_yaw,
    root_features_to_root_positions,
    root_positions_to_root_features,
    subset_271d_to_72d,
    yaw_to_root_rot6d,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_mini_dataset_normalizer() -> FeatureNormalizer:
    dataset_root = ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
    mean = torch.from_numpy(np.load(dataset_root / "Mean.npy")).float()
    std = torch.from_numpy(np.load(dataset_root / "Std.npy")).float()
    return FeatureNormalizer(mean=mean, std=std)


def _load_mini_sample(sample_id: str = "000070") -> tuple[torch.Tensor, torch.Tensor]:
    dataset_root = ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
    features = torch.from_numpy(
        np.load(dataset_root / "new_joint_vecs" / f"{sample_id}.npy")
    ).float()
    joints = torch.from_numpy(
        np.load(dataset_root / "new_joints" / f"{sample_id}.npy")
    ).float()
    return features, joints


def test_root_motion_round_trip_sequence() -> None:
    prev_root_pos = torch.tensor(
        [
            [10.0, 1.0, 20.0],
            [-3.0, 5.0, 4.0],
        ]
    )
    root_features = torch.tensor(
        [
            [
                [2.0, 0.5, -1.0],
                [2.1, 0.25, 0.5],
                [1.9, -0.75, 1.25],
            ],
            [
                [5.5, 1.0, 2.0],
                [5.0, -0.5, -1.5],
                [4.8, 0.25, 0.75],
            ],
        ]
    )

    root_positions = root_features_to_root_positions(root_features, prev_root_pos)
    recovered_features = root_positions_to_root_features(root_positions, prev_root_pos)

    expected_positions = torch.tensor(
        [
            [
                [10.5, 2.0, 19.0],
                [10.75, 2.1, 19.5],
                [10.0, 1.9, 20.75],
            ],
            [
                [-2.0, 5.5, 6.0],
                [-2.5, 5.0, 4.5],
                [-2.25, 4.8, 5.25],
            ],
        ]
    )

    assert torch.allclose(root_positions, expected_positions)
    assert torch.allclose(recovered_features, root_features)


def test_root_motion_round_trip_single_frame() -> None:
    prev_root_pos = torch.tensor([[1.5, 0.0, -2.0]])
    root_features = torch.tensor([[3.25, -0.5, 1.75]])

    root_positions = root_features_to_root_positions(root_features, prev_root_pos)
    recovered_features = root_positions_to_root_features(root_positions, prev_root_pos)

    assert torch.allclose(root_positions, torch.tensor([[1.0, 3.25, -0.25]]))
    assert torch.allclose(recovered_features, root_features)


def test_root_yaw_round_trip_through_6d() -> None:
    yaw = torch.tensor([-1.5, -0.25, 0.0, 0.75, 2.25])
    root_rot_6d = yaw_to_root_rot6d(yaw)
    recovered_yaw = root_rot6d_to_yaw(root_rot_6d)

    assert torch.allclose(recovered_yaw, yaw, atol=1e-6, rtol=1e-6)


def test_subset_271d_to_72d_returns_68d_delta_yaw_layout() -> None:
    prev_frame = torch.zeros(1, 271)
    current_frame = torch.zeros(1, 271)
    prev_frame[:, 69:75] = yaw_to_root_rot6d(torch.tensor([0.0]))
    current_frame[:, 69:75] = yaw_to_root_rot6d(torch.tensor([torch.pi / 2]))
    current_frame[:, 0:3] = torch.tensor([[1.25, -0.5, 0.75]])
    current_frame[:, 6:69] = torch.arange(63, dtype=torch.float32).view(1, 63)

    reduced = subset_271d_to_72d(current_frame, prev_frame=prev_frame)

    assert reduced.shape == (1, 68)
    assert torch.allclose(reduced[:, 0:3], current_frame[:, 0:3])
    assert torch.allclose(
        reduced[:, 3:5],
        torch.tensor([[1.0, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(reduced[:, 5:68], current_frame[:, 6:69])


def test_flow_output_to_positions_integrates_delta_yaw() -> None:
    prev_root_pos = torch.tensor([[0.0, 0.0, 0.0]])
    prev_root_rot_6d = yaw_to_root_rot6d(torch.tensor([0.0]))
    flow_output = torch.zeros(1, 68)
    flow_output[:, 0:3] = torch.tensor([[1.0, 0.0, 0.0]])
    flow_output[:, 3:5] = torch.tensor([[1.0, 0.0]])  # +90 degrees
    flow_output[:, 5:8] = torch.tensor([[0.0, 0.0, 1.0]])

    positions = flow_output_to_positions(
        flow_output,
        prev_root_pos=prev_root_pos,
        prev_root_rot_6d=prev_root_rot_6d,
    )

    assert torch.allclose(positions[:, 0], torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(
        positions[:, 1],
        torch.tensor([[-1.0, 1.0, 0.0]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_normalize_flow_output_round_trip_with_delta_yaw() -> None:
    mean = torch.zeros(271)
    std = torch.linspace(1.0, 2.0, 271)
    normalizer = FeatureNormalizer(mean=mean, std=std)

    raw_flow = torch.randn(2, 68)
    raw_flow[:, 3:5] = torch.tensor([[0.5, 0.8660254], [-0.25, 0.9682458]])

    normalized = normalizer.normalize_flow_output(raw_flow)
    recovered = normalizer.denormalize_flow_output(normalized)

    assert torch.allclose(recovered, raw_flow, atol=1e-6, rtol=1e-6)


def test_normalized_training_reduced_state_round_trip_matches_sample_positions() -> None:
    normalizer = _load_mini_dataset_normalizer()
    features_raw, positions = _load_mini_sample("000070")
    features_norm = normalizer.normalize(features_raw)

    for t in (1, 10, 25, features_raw.shape[0] - 1):
        prev_frame_norm = features_norm[t - 1 : t]
        frame_norm = features_norm[t : t + 1]

        reduced_norm = subset_271d_to_72d(
            frame_norm,
            prev_frame=prev_frame_norm,
            normalizer=normalizer,
        )
        reduced_raw = normalizer.denormalize_flow_output(reduced_norm)
        reconstructed_positions = flow_output_to_positions(
            reduced_raw,
            prev_root_pos=positions[t - 1 : t, 0],
            prev_root_rot_6d=features_raw[t - 1 : t, 69:75],
        )
        roundtrip_frame_raw, _, _ = generated_positions_to_271d(
            new_positions=reconstructed_positions,
            prev_positions=positions[t - 1 : t],
            normalizer=None,
        )
        roundtrip_frame_norm = normalizer.normalize(roundtrip_frame_raw)
        reduced_roundtrip_norm = subset_271d_to_72d(
            roundtrip_frame_norm,
            prev_frame=prev_frame_norm,
            normalizer=normalizer,
        )

        assert torch.allclose(
            reconstructed_positions,
            positions[t : t + 1],
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            reduced_roundtrip_norm,
            reduced_norm,
            atol=1e-5,
            rtol=1e-5,
        )


def test_normalized_inference_conditioning_uses_raw_yaw_but_preserves_normalized_channels() -> None:
    normalizer = _load_mini_dataset_normalizer()
    features_raw, _ = _load_mini_sample("000121")
    frame_raw = features_raw[20:21]
    frame_norm = normalizer.normalize(frame_raw)

    conditioning_norm = extract_prev_frame_features(frame_norm, normalizer=normalizer)
    conditioning_raw = extract_prev_frame_features(frame_raw, normalizer=None)
    expected_joint_channels = torch.cat(
        [frame_norm[:, 6:69], frame_norm[:, 75:201], frame_norm[:, 204:267]], dim=-1
    )

    assert torch.allclose(conditioning_norm[:, 0:3], frame_norm[:, 0:3], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        conditioning_norm[:, 5:],
        expected_joint_channels,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(conditioning_norm[:, 3:5], conditioning_raw[:, 3:5], atol=1e-6, rtol=1e-6)
