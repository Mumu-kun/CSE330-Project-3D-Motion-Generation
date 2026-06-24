"""Tests for 68D format and conversion functions in motion_utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.motion_utils import (
    FeatureNormalizer,
    Features,
    root_rot6d_to_yaw,
    root_rot6d_to_yaw_sin_cos,
    sin_cos_to_yaw,
    x68_to_positions,
    x68_to_x271,
    x271_to_x68,
    yaw_to_root_rot6d,
    yaw_to_sin_cos,
)

DATASET_PATH = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"


class TestFeatureLayout:
    """Test 68D feature layout dimensions."""

    def test_d68_slices_sum_to_68(self):
        assert Features.D68_ROOT_Y.stop - Features.D68_ROOT_Y.start == 1
        assert Features.D68_ROOT_VX.stop - Features.D68_ROOT_VX.start == 1
        assert Features.D68_ROOT_VZ.stop - Features.D68_ROOT_VZ.start == 1
        assert Features.D68_YAW_SINCOS.stop - Features.D68_YAW_SINCOS.start == 2
        assert Features.D68_JOINTS_VEL.stop - Features.D68_JOINTS_VEL.start == 63
        total = (
            (Features.D68_ROOT_Y.stop - Features.D68_ROOT_Y.start)
            + (Features.D68_ROOT_VX.stop - Features.D68_ROOT_VX.start)
            + (Features.D68_ROOT_VZ.stop - Features.D68_ROOT_VZ.start)
            + (Features.D68_YAW_SINCOS.stop - Features.D68_YAW_SINCOS.start)
            + (Features.D68_JOINTS_VEL.stop - Features.D68_JOINTS_VEL.start)
        )
        assert total == 68, f"68D layout should sum to 68, got {total}"

    def test_d68_yaw_sin_cos_slice(self):
        assert Features.D68_YAW_SINCOS == slice(3, 5)
        assert Features.D68_JOINTS_VEL == slice(5, 68)


class TestYawConversions:
    """Test yaw <-> rotation conversions."""

    def test_yaw_to_root_rot6d_roundtrip(self):
        yaw = torch.tensor([0.5, 1.0, -0.3, 2.5])
        rot6d = yaw_to_root_rot6d(yaw)
        assert rot6d.shape == (4, 6)
        yaw_back = root_rot6d_to_yaw(rot6d)
        assert torch.allclose(yaw, yaw_back, atol=1e-6)

    def test_sin_cos_to_yaw_roundtrip(self):
        yaw = torch.tensor([0.5, 1.0, -0.3, 2.5])
        sin_cos = yaw_to_sin_cos(yaw)
        assert sin_cos.shape == (4, 2)
        yaw_back = sin_cos_to_yaw(sin_cos)
        assert torch.allclose(yaw, yaw_back, atol=1e-6)

    def test_yaw_sin_cos_consistency(self):
        yaw = torch.tensor([0.5, 1.0, -0.3])
        sin_cos = root_rot6d_to_yaw_sin_cos(yaw_to_root_rot6d(yaw))
        expected = yaw_to_sin_cos(yaw)
        assert torch.allclose(sin_cos, expected, atol=1e-6)


class TestNormalizer:
    """Test FeatureNormalizer for 68D features."""

    def test_normalizer_68d_stats_shape(self):
        mean_271 = torch.zeros(271)
        std_271 = torch.ones(271)
        normalizer = FeatureNormalizer(mean_271, std_271)
        assert normalizer._mean_68d.shape == (68,)
        assert normalizer._std_68d.shape == (68,)

    def test_normalizer_68d_root_y(self):
        mean_271 = torch.zeros(271)
        mean_271[Features.ROOT_Y] = 0.5
        std_271 = torch.ones(271)
        std_271[Features.ROOT_Y] = 0.2
        normalizer = FeatureNormalizer(mean_271, std_271)
        assert torch.allclose(normalizer._mean_68d[0:1], torch.tensor([0.5]))
        assert torch.allclose(normalizer._std_68d[0:1], torch.tensor([0.2]))

    def test_normalizer_68d_joint_vel(self):
        mean_271 = torch.zeros(271)
        mean_271[Features.JOINT_VEL] = 0.1
        std_271 = torch.ones(271)
        std_271[Features.JOINT_VEL] = 0.3
        normalizer = FeatureNormalizer(mean_271, std_271)
        assert torch.allclose(normalizer._mean_68d[5:68], torch.full((63,), 0.1))
        assert torch.allclose(normalizer._std_68d[5:68], torch.full((63,), 0.3))

    def test_normalize_denormalize_x68_inverse(self):
        mean_271 = torch.randn(271) * 0.5
        std_271 = torch.abs(torch.randn(271)) + 0.5
        normalizer = FeatureNormalizer(mean_271, std_271)
        x68 = torch.randn(10, 68)
        normalized = normalizer.normalize_x68(x68)
        denormalized = normalizer.denormalize_x68(normalized)
        assert torch.allclose(x68, denormalized, atol=1e-6)


class TestRoundTripConversion:
    """Test round-trip conversions between positions, x271, and x68."""

    @classmethod
    def setup_class(cls):
        mean_path = DATASET_PATH / "Mean.npy"
        std_path = DATASET_PATH / "Std.npy"
        cls.normalizer = FeatureNormalizer.load_from_files(str(mean_path), str(std_path))

    def test_x68_to_positions_single_frame(self):
        x68 = torch.randn(1, 68)
        prev_x271 = torch.zeros(1, 271)
        positions = x68_to_positions(x68, self.normalizer, prev_x271)
        assert positions.shape == (1, 22, 3)
        assert torch.isfinite(positions).all()

    def test_x68_to_positions_batch(self):
        x68 = torch.randn(4, 68)
        prev_x271 = torch.zeros(4, 271)
        positions = x68_to_positions(x68, self.normalizer, prev_x271)
        assert positions.shape == (4, 22, 3)

    def test_positions_to_x68_to_positions(self):
        joint_vec_path = DATASET_PATH / "new_joint_vecs" / "000009.npy"
        joint_pos_path = DATASET_PATH / "new_joints" / "000009.npy"
        x271_data = np.load(joint_vec_path)
        positions_data = np.load(joint_pos_path)
        x271 = torch.from_numpy(x271_data).float()
        original_positions = torch.from_numpy(positions_data).float()

        x68_frames = []
        prev_pos = None
        prev_x271 = None
        for i in range(len(x271)):
            x68_frame = x271_to_x68(
                x271[i : i + 1], self.normalizer, prev_positions=prev_pos, prev_x271=prev_x271
            )
            x68_frames.append(x68_frame)
            prev_pos = original_positions[i : i + 1]
            prev_x271 = x271[i : i + 1]

        x68 = torch.cat(x68_frames, dim=0)
        assert x68.shape[-1] == 68

        reconstructed_frames = []
        for i in range(len(x68)):
            if i == 0:
                prev_pos_frame = original_positions[i : i + 1]
            else:
                prev_pos_frame = reconstructed_frames[-1]
            rec = x68_to_positions(x68[i : i + 1], self.normalizer, x271[i : i + 1], prev_pos_frame)
            reconstructed_frames.append(rec)

        reconstructed = torch.cat(reconstructed_frames, dim=0)
        assert reconstructed.shape == original_positions.shape
        assert torch.isfinite(reconstructed).all()

    def test_multiple_samples_roundtrip(self):
        joint_vecs_dir = DATASET_PATH / "new_joint_vecs"
        joint_pos_dir = DATASET_PATH / "new_joints"
        files = ["000016.npy", "000017.npy", "000019.npy"]
        for fname in files:
            x271_data = np.load(joint_vecs_dir / fname)
            positions_data = np.load(joint_pos_dir / fname)
            x271 = torch.from_numpy(x271_data).float()
            original_positions = torch.from_numpy(positions_data).float()

            x68_frames = []
            prev_pos = None
            prev_x271 = None
            for i in range(len(x271)):
                x68_frame = x271_to_x68(
                    x271[i : i + 1], self.normalizer, prev_positions=prev_pos, prev_x271=prev_x271
                )
                x68_frames.append(x68_frame)
                prev_pos = original_positions[i : i + 1]
                prev_x271 = x271[i : i + 1]

            x68 = torch.cat(x68_frames, dim=0)

            reconstructed_frames = []
            for i in range(len(x68)):
                if i == 0:
                    prev_pos_frame = original_positions[i : i + 1]
                else:
                    prev_pos_frame = reconstructed_frames[-1]
                rec = x68_to_positions(x68[i : i + 1], self.normalizer, x271[i : i + 1], prev_pos_frame)
                reconstructed_frames.append(rec)

            reconstructed = torch.cat(reconstructed_frames, dim=0)
            assert torch.isfinite(reconstructed).all()
            assert reconstructed.shape == original_positions.shape

    def test_x68_to_x271_shape(self):
        x68 = torch.randn(2, 68)
        prev_x271 = torch.zeros(2, 271)
        x271 = x68_to_x271(x68, self.normalizer, prev_x271)
        assert x271.shape == (2, 271)


class TestSyntheticPositionTest:
    """Test x68 -> positions with synthetic input."""

    def test_synthetic_x68_positions(self):
        normalizer = FeatureNormalizer(
            torch.zeros(271),
            torch.ones(271),
        )
        x68 = torch.zeros(1, 68)
        x68[:, 0] = 0.9
        x68[:, Features.D68_YAW_SINCOS] = torch.tensor([0.0, 1.0])
        prev_x271 = torch.zeros(1, 271)
        positions = x68_to_positions(x68, normalizer, prev_x271)
        assert positions.shape == (1, 22, 3)
        assert torch.allclose(positions[0, 0, 1], torch.tensor(0.9), atol=1e-5)

    def test_root_y_in_reasonable_range(self):
        normalizer = FeatureNormalizer(
            torch.zeros(271),
            torch.ones(271),
        )
        x68 = torch.randn(10, 68)
        x68[:, 0] = 1.0
        prev_x271 = torch.zeros(10, 271)
        positions = x68_to_positions(x68, normalizer, prev_x271)
        for i in range(10):
            assert -10 < positions[i, 0, 1].item() < 10