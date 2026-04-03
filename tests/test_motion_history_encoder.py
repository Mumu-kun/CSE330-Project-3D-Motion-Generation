import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import MotionHistoryEncoderConfig
from models import MotionHistoryEncoder


def _make_encoder() -> MotionHistoryEncoder:
    config = MotionHistoryEncoderConfig(
        frame_feature_dim=271,
        text_embedding_dim=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        per_joint_output_dim=8,
        joint_count=22,
    )
    return MotionHistoryEncoder(config).eval()


def test_forward_output_shape_and_forward_all_last_match() -> None:
    encoder = _make_encoder()
    motion = torch.randn(2, 6, 271)
    text = torch.randn(2, 512)

    with torch.no_grad():
        all_contexts = encoder(motion, text, return_all=True)
        latest_context = encoder(motion, text)

    assert all_contexts.shape == (2, 6, 22, 8)
    assert latest_context.shape == (2, 22, 8)
    assert torch.allclose(all_contexts[:, -1], latest_context, atol=1e-6, rtol=1e-6)


def test_step_matches_forward_for_each_prefix_and_grows_buffer() -> None:
    encoder = _make_encoder()
    motion = torch.randn(2, 6, 271)
    text = torch.randn(2, 512)

    frame_buffer = None
    cache_state = None

    with torch.no_grad():
        for prefix_len in range(1, motion.shape[1] + 1):
            step_context, frame_buffer, cache_state = encoder.step(
                motion[:, prefix_len - 1],
                text,
                frame_buffer=frame_buffer,
                cache_state=cache_state,
            )
            forward_context = encoder(motion[:, :prefix_len], text)

            assert torch.allclose(
                step_context, forward_context, atol=1e-6, rtol=1e-6
            )
            assert frame_buffer is not None
            assert frame_buffer.shape == (
                motion.shape[0],
                prefix_len,
                motion.shape[-1],
            )

    assert cache_state is not None
    assert len(cache_state.layers) == encoder.num_layers
