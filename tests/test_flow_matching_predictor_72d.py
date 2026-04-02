import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import FlowMatchingPredictorConfig
from models import FlowMatchingPredictor
from utils.motion_utils import extract_prev_frame_features, yaw_to_root_rot6d


def _build_predictor(feature_size: int = 16) -> FlowMatchingPredictor:
    config = FlowMatchingPredictorConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_dropout=0.0,
        global_cond_dim=512,
    )
    return FlowMatchingPredictor(feature_size=feature_size, config=config)


def test_extract_prev_frame_features_layout() -> None:
    frame = torch.randn(3, 271)
    frame[:, 69:75] = yaw_to_root_rot6d(torch.tensor([0.0, 0.5, -0.75]))
    features = extract_prev_frame_features(frame)

    assert features.shape == (3, 257)
    assert torch.allclose(
        features[:, :5],
        torch.cat(
            [
                frame[:, 0:1],
                frame[:, 1:3],
                torch.stack(
                    [
                        torch.sin(torch.tensor([0.0, 0.5, -0.75])),
                        torch.cos(torch.tensor([0.0, 0.5, -0.75])),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        ),
    )
    assert torch.allclose(
        features[:, 5:],
        torch.cat([frame[:, 6:69], frame[:, 75:201], frame[:, 204:267]], dim=-1),
    )


def test_flow_matching_predictor_accepts_68d_and_257d_inputs() -> None:
    predictor = _build_predictor()
    batch_size = 4
    noisy_features = torch.randn(batch_size, 68)
    timesteps = torch.rand(batch_size)
    text_embedding = torch.randn(batch_size, 512)
    track_features = torch.randn(batch_size, 22, 16)
    current_frame_features = torch.randn(batch_size, 257)

    pred, hidden_states, attentions = predictor(
        noisy_features=noisy_features,
        timesteps=timesteps,
        text_embedding=text_embedding,
        track_features=track_features,
        current_frame_features=current_frame_features,
    )

    assert pred.shape == (batch_size, 68)
    assert hidden_states is None
    assert attentions is None


@pytest.mark.parametrize(
    ("bad_noisy_shape", "bad_frame_shape"),
    [((2, 67), (2, 257)), ((2, 68), (2, 256)), ((2, 68), (2, 258))],
)
def test_flow_matching_predictor_rejects_malformed_inputs(
    bad_noisy_shape: tuple[int, ...],
    bad_frame_shape: tuple[int, ...],
) -> None:
    predictor = _build_predictor()
    batch_size = bad_noisy_shape[0]
    timesteps = torch.rand(batch_size)
    text_embedding = torch.randn(batch_size, 512)
    track_features = torch.randn(batch_size, 22, 16)
    noisy_features = torch.randn(*bad_noisy_shape)
    current_frame_features = torch.randn(*bad_frame_shape)

    with pytest.raises(ValueError):
        predictor(
            noisy_features=noisy_features,
            timesteps=timesteps,
            text_embedding=text_embedding,
            track_features=track_features,
            current_frame_features=current_frame_features,
        )
