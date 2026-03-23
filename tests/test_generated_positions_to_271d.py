"""Tests for optional normalization in generated_positions_to_271d."""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import FeatureNormalizer, generated_positions_to_271d


def test_generated_positions_to_271d_optional_normalizer():
    """The output frame is normalized only when a normalizer is provided."""
    torch.manual_seed(7)

    batch_size = 2
    prev_positions = torch.randn(batch_size, 22, 3)
    new_positions = prev_positions + 0.05 * torch.randn(batch_size, 22, 3)

    raw_frame, raw_root = generated_positions_to_271d(
        new_positions=new_positions,
        prev_positions=prev_positions,
        dataset_type="t2m",
    )

    mean = torch.linspace(-1.0, 1.0, 271)
    std = torch.linspace(0.5, 1.5, 271)
    normalizer = FeatureNormalizer(mean=mean, std=std)

    norm_frame, norm_root = generated_positions_to_271d(
        new_positions=new_positions,
        prev_positions=prev_positions,
        dataset_type="t2m",
        normalizer=normalizer,
    )

    expected_norm = (raw_frame - mean.to(raw_frame.device, raw_frame.dtype)) / std.to(
        raw_frame.device, raw_frame.dtype
    )

    assert raw_frame.shape == (batch_size, 271)
    assert norm_frame.shape == (batch_size, 271)
    assert raw_root.shape == (batch_size, 3)
    assert norm_root.shape == (batch_size, 3)

    assert raw_frame.dtype == norm_frame.dtype
    assert raw_root.dtype == norm_root.dtype

    assert torch.allclose(norm_frame, expected_norm, atol=1e-5)
    assert not torch.allclose(raw_frame, norm_frame)
    assert torch.allclose(raw_root, norm_root, atol=1e-6)
