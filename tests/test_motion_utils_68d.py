"""Tests for 72D (~75D) format and conversion functions in motion_utils.py."""

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
    x72_to_positions,
    x72_to_x271,
    x271_to_x72,
    x68_to_positions,
    x271_to_x68,
    yaw_to_root_rot6d,
    yaw_to_sin_cos,
)

DATASET_PATH = (
    PROJECT_ROOT / "src" / "dataset" / "humanml3d-subset"
    if (PROJECT_ROOT / "src" / "dataset" / "humanml3d-subset" / "Mean.npy").exists()
    else PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
)


class TestFeatureLayout:
    """Test 72D feature layout dimensions."""

    def test_d72_slices_sum_to_72(self):
        assert Features.D72_ROOT_Y.stop - Features.D72_ROOT_Y.start == 1
        assert Features.D72_ROOT_VX.stop - Features.D72_ROOT_VX.start == 1
        assert Features.D72_ROOT_VZ.stop - Features.D72_ROOT_VZ.start == 1
        assert Features.D72_YAW_SINCOS.stop - Features.D72_YAW_SINCOS.start == 2
        assert Features.D72_JOINTS_RIC.stop - Features.D72_JOINTS_RIC.start == 63
        assert Features.D72_CONTACTS.stop - Features.D72_CONTACTS.start == 4
        total = (
            (Features.D72_ROOT_Y.stop - Features.D72_ROOT_Y.start)
            + (Features.D72_ROOT_VX.stop - Features.D72_ROOT_VX.start)
            + (Features.D72_ROOT_VZ.stop - Features.D72_ROOT_VZ.start)
            + (Features.D72_YAW_SINCOS.stop - Features.D72_YAW_SINCOS.start)
            + (Features.D72_JOINTS_RIC.stop - Features.D72_JOINTS_RIC.start)
            + (Features.D72_CONTACTS.stop - Features.D72_CONTACTS.start)
        )
        assert total == 72, f"72D layout should sum to 72, got {total}"

    def test_d72_slices(self):
        assert Features.D72_YAW_SINCOS == slice(3, 5)
        assert Features.D72_JOINTS_RIC == slice(5, 68)
        assert Features.D72_CONTACTS == slice(68, 72)


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
    """Test FeatureNormalizer for 72D features."""

    def test_normalizer_72d_stats_shape(self):
        mean_271 = torch.zeros(271)
        std_271 = torch.ones(271)
        normalizer = FeatureNormalizer(mean_271, std_271)
        assert normalizer._mean_72d.shape == (72,)
        assert normalizer._std_72d.shape == (72,)

    def test_normalizer_72d_root_y(self):
        mean_271 = torch.zeros(271)
        mean_271[Features.ROOT_Y] = 0.5
        std_271 = torch.ones(271)
        std_271[Features.ROOT_Y] = 0.2
        normalizer = FeatureNormalizer(mean_271, std_271)
        assert torch.allclose(normalizer._mean_72d[0:1], torch.tensor([0.5]))
        assert torch.allclose(normalizer._std_72d[0:1], torch.tensor([0.2]))

    def test_normalizer_72d_joint_ric(self):
        mean_271 = torch.zeros(271)
        mean_271[Features.JOINT_RIC] = 0.1
        std_271 = torch.ones(271)
        std_271[Features.JOINT_RIC] = 0.3
        normalizer = FeatureNormalizer(mean_271, std_271)
        assert torch.allclose(normalizer._mean_72d[5:68], torch.full((63,), 0.1))
        assert torch.allclose(normalizer._std_72d[5:68], torch.full((63,), 0.3))

    def test_normalize_denormalize_x72_inverse(self):
        mean_271 = torch.randn(271) * 0.5
        std_271 = torch.abs(torch.randn(271)) + 0.5
        normalizer = FeatureNormalizer(mean_271, std_271)
        x72 = torch.randn(10, 72)
        normalized = normalizer.normalize_x72(x72)
        denormalized = normalizer.denormalize_x72(normalized)
        assert torch.allclose(x72, denormalized, atol=1e-6)


class TestRoundTripConversion:
    """Test round-trip conversions between positions, x271, and x72."""

    @classmethod
    def setup_class(cls):
        mean_path = DATASET_PATH / "Mean.npy"
        std_path = DATASET_PATH / "Std.npy"
        cls.normalizer = FeatureNormalizer.load_from_files(str(mean_path), str(std_path))

    def test_x72_to_positions_single_frame(self):
        x72 = torch.randn(1, 72)
        prev_x271 = torch.zeros(1, 271)
        positions = x72_to_positions(x72, self.normalizer, prev_x271)
        assert positions.shape == (1, 22, 3)
        assert torch.isfinite(positions).all()

    def test_x72_to_positions_batch(self):
        x72 = torch.randn(4, 72)
        prev_x271 = torch.zeros(4, 271)
        positions = x72_to_positions(x72, self.normalizer, prev_x271)
        assert positions.shape == (4, 22, 3)

    def test_positions_to_x72_to_positions(self):
        joint_vec_path = DATASET_PATH / "new_joint_vecs" / "000009.npy"
        joint_pos_path = DATASET_PATH / "new_joints" / "000009.npy"
        x271_data = np.load(joint_vec_path)
        positions_data = np.load(joint_pos_path)
        x271 = torch.from_numpy(x271_data).float()
        original_positions = torch.from_numpy(positions_data).float()

        x72_frames = []
        prev_pos = None
        prev_x271 = None
        for i in range(len(x271)):
            x72_frame = x271_to_x72(
                x271[i : i + 1], self.normalizer, prev_positions=prev_pos, prev_x271=prev_x271
            )
            x72_frames.append(x72_frame)
            prev_pos = original_positions[i : i + 1]
            prev_x271 = x271[i : i + 1]

        x72 = torch.cat(x72_frames, dim=0)
        assert x72.shape[-1] == 72

        reconstructed_frames = []
        for i in range(len(x72)):
            if i == 0:
                prev_pos_frame = original_positions[i : i + 1]
            else:
                prev_pos_frame = reconstructed_frames[-1]
            rec = x72_to_positions(x72[i : i + 1], self.normalizer, x271[i : i + 1], prev_pos_frame)
            reconstructed_frames.append(rec)

        reconstructed = torch.cat(reconstructed_frames, dim=0)
        assert reconstructed.shape == original_positions.shape
        assert torch.isfinite(reconstructed).all()

    def test_x72_to_x271_shape(self):
        x72 = torch.randn(2, 72)
        prev_x271 = torch.zeros(2, 271)
        x271 = x72_to_x271(x72, self.normalizer, prev_x271)
        assert x271.shape == (2, 271)


class TestLongSequenceDriftStability:
    """Regression test: verifies direct RIC position representation eliminates dead-reckoning drift."""

    @classmethod
    def setup_class(cls):
        mean_path = DATASET_PATH / "Mean.npy"
        std_path = DATASET_PATH / "Std.npy"
        cls.normalizer = FeatureNormalizer.load_from_files(str(mean_path), str(std_path))

    def test_drift_growth_ratio_is_flat(self):
        joint_vec_path = DATASET_PATH / "new_joint_vecs" / "000009.npy"
        joint_pos_path = DATASET_PATH / "new_joints" / "000009.npy"
        x271_data = np.load(joint_vec_path)
        pos_data = np.load(joint_pos_path)

        # Tile to 200+ frames
        repeats = (240 // len(x271_data)) + 1
        x271_seq = torch.from_numpy(np.tile(x271_data, (repeats, 1))[:240]).float()
        pos_gt_seq = torch.from_numpy(np.tile(pos_data, (repeats, 1, 1))[:240]).float()

        curr_pos = pos_gt_seq[0:1]
        curr_frame = x271_seq[0:1]

        mpjpe_per_frame = []

        for t in range(len(x271_seq) - 1):
            next_frame_gt = x271_seq[t + 1 : t + 2]
            target_72d = x271_to_x72(next_frame_gt, self.normalizer, prev_positions=curr_pos, prev_x271=curr_frame)

            new_pos = x72_to_positions(target_72d, self.normalizer, curr_frame, curr_pos)

            gt_pos_t = pos_gt_seq[t + 1 : t + 2]
            rel_new = new_pos - new_pos[:, 0:1]
            rel_gt = gt_pos_t - gt_pos_t[:, 0:1]
            err_t = torch.norm(rel_new - rel_gt, dim=-1).mean().item()
            mpjpe_per_frame.append(err_t)

            curr_pos = new_pos
            curr_frame = next_frame_gt

        early_err = np.mean(mpjpe_per_frame[:40])
        late_err = np.mean(mpjpe_per_frame[-40:])
        growth_ratio = late_err / max(early_err, 1e-6)

        print(f"Regression Test - Early Relative MPJPE (1-40): {early_err:.6f}, Late Relative MPJPE (160-200): {late_err:.6f}, Growth Ratio: {growth_ratio:.2f}x")

        # Growth ratio should stay flat (< 1.5x), not explode ~15x
        assert growth_ratio < 1.5, f"Drift growth ratio exploded to {growth_ratio:.2f}x (expected < 1.5x)"