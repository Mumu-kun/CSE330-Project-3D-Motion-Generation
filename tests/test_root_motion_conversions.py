import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import (  # noqa: E402
    root_features_to_root_positions,
    root_positions_to_root_features,
)


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
