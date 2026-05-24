import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.checkpoint_eval import aggregate_position_metrics, root_aligned_mpjpe
from utils.motion_utils import sequence_271d_to_263d, yaw_to_root_rot6d


def test_sequence_271d_to_263d_preserves_legacy_slices() -> None:
    features = torch.zeros(2, 271)
    features[:, 0] = torch.tensor([1.25, 1.50])
    features[:, 1] = torch.tensor([0.10, 0.20])
    features[:, 2] = torch.tensor([0.30, 0.40])
    features[:, 6:69] = torch.arange(63, dtype=torch.float32).view(1, 63)
    features[:, 75:201] = torch.arange(126, dtype=torch.float32).view(1, 126) + 100.0
    features[:, 201:267] = torch.arange(66, dtype=torch.float32).view(1, 66) + 200.0
    features[:, 267:271] = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    features[:, 69:75] = yaw_to_root_rot6d(torch.tensor([0.0, math.pi / 2]))

    converted = sequence_271d_to_263d(features)

    assert converted.shape == (2, 263)
    assert torch.allclose(converted[0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(converted[1, 0], torch.tensor(math.pi / 2), atol=1e-6)
    assert torch.allclose(converted[:, 1:3], features[:, 1:3])
    assert torch.allclose(converted[:, 3:4], features[:, 0:1])
    assert torch.allclose(converted[:, 4:67], features[:, 6:69])
    assert torch.allclose(converted[:, 67:193], features[:, 75:201])
    assert torch.allclose(converted[:, 193:259], features[:, 201:267])
    assert torch.allclose(converted[:, 259:263], features[:, 267:271])


def test_root_aligned_mpjpe_is_translation_invariant() -> None:
    target = torch.zeros(2, 22, 3)
    pred = target + torch.tensor([1.0, 0.0, 0.0])

    mpjpe = root_aligned_mpjpe(pred, target)
    metrics = aggregate_position_metrics(pred, target)

    assert torch.allclose(mpjpe, torch.zeros(2), atol=1e-6)
    assert metrics["mpjpe"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["joint_l2_mean"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["root_l2_mean"] == pytest.approx(1.0, abs=1e-6)
