"""Sample-data regression checks for generated_positions_to_271d.

Covers:
- fk_offsets branch over full sequence
- teacher-forced and autoregressive prev-position modes
- full 271D frame error and FK-position error
- sequence round-trip reconstruction consistency
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import (  # noqa: E402
    features_to_positions,
    generated_positions_to_271d,
    get_fk_offsets,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_sample(sample_id: str = "000070") -> Dict[str, torch.Tensor]:
    gt_features = torch.from_numpy(
        np.load(ROOT / "sample_data" / f"{sample_id}_vec.npy")
    ).float()
    gt_positions = features_to_positions(gt_features)
    gt_joint_file = torch.from_numpy(
        np.load(ROOT / "sample_data" / f"{sample_id}_joint.npy")
    ).float()
    return {
        "features": gt_features,
        "positions": gt_positions,
        "joint_file": gt_joint_file,
    }


def _run_variant(
    gt_features: torch.Tensor,
    gt_positions: torch.Tensor,
    mode: str,
    use_fk_offsets: bool,
) -> Dict[str, float]:
    fk_offsets = get_fk_offsets(gt_positions.unsqueeze(0))

    frame_fro_errors = []
    fk_joint_errors = []
    generated_frames = []

    ar_prev_positions: Optional[torch.Tensor] = None

    n_frames = gt_features.shape[0]

    for t in range(n_frames):
        new_positions = gt_positions[t : t + 1]

        if t == 0:
            prev_positions = None
        elif mode == "teacher_forced":
            prev_positions = gt_positions[t - 1 : t]
        elif mode == "autoregressive":
            prev_positions = ar_prev_positions
        else:
            raise ValueError(f"Unknown mode: {mode}")

        frame_out, _, fk_positions = generated_positions_to_271d(
            new_positions=new_positions,
            prev_positions=prev_positions,
            dataset_type="t2m",
            fk_offsets=fk_offsets if use_fk_offsets else None,
            normalizer=None,
        )

        pred_frame = frame_out.squeeze(0)
        gt_frame = gt_features[t]

        diff = pred_frame - gt_frame
        frame_fro_errors.append(float(torch.linalg.vector_norm(diff, ord=2).item()))

        if use_fk_offsets:
            assert fk_positions is not None
            fk_err = torch.linalg.vector_norm(
                fk_positions.squeeze(0) - gt_positions[t], ord=2, dim=-1
            )
            fk_joint_errors.append(float(fk_err.mean().item()))
        else:
            assert fk_positions is None

        generated_frames.append(pred_frame)

        if mode == "autoregressive":
            seq_pred = torch.stack(generated_frames, dim=0)
            seq_pos = features_to_positions(seq_pred)
            ar_prev_positions = seq_pos[-1:].detach()

    seq_pred = torch.stack(generated_frames, dim=0)
    seq_pos = features_to_positions(seq_pred)
    seq_joint_l2 = torch.linalg.vector_norm(seq_pos - gt_positions, ord=2, dim=-1)

    out = {
        "feature_fro_mean": float(np.mean(frame_fro_errors)),
        "feature_fro_max": float(np.max(frame_fro_errors)),
        "roundtrip_joint_l2_mean": float(seq_joint_l2.mean().item()),
        "roundtrip_joint_l2_max": float(seq_joint_l2.max().item()),
    }

    if fk_joint_errors:
        out["fk_joint_l2_mean"] = float(np.mean(fk_joint_errors))
        out["fk_joint_l2_max"] = float(np.max(fk_joint_errors))

    return out


def test_sample_data_ground_truth_alignment_is_tight() -> None:
    payload = _load_sample("000070")
    gt_positions = payload["positions"]
    gt_joint_file = payload["joint_file"]

    joint_l2 = torch.linalg.vector_norm(gt_positions - gt_joint_file, ord=2, dim=-1)
    print("GT position to joint file L2 errors:")
    print(f"  mean: {joint_l2.mean().item():.6f}")
    print(f"  max: {joint_l2.max().item():.6f}")
    assert float(joint_l2.mean().item()) < 1e-6
    assert float(joint_l2.max().item()) < 1e-5


def test_generated_positions_to_271d_full_sequence_fk_and_roundtrip() -> None:
    payload = _load_sample("000070")
    gt_features = payload["features"]
    gt_positions = payload["positions"]

    variants = [
        ("teacher_forced", False),
        ("teacher_forced", True),
        ("autoregressive", False),
        ("autoregressive", True),
    ]

    results = {}
    for mode, use_fk in variants:
        key = f"{mode}|fk={int(use_fk)}"
        results[key] = _run_variant(
            gt_features=gt_features,
            gt_positions=gt_positions,
            mode=mode,
            use_fk_offsets=use_fk,
        )

    print("Results for generated_positions_to_271d regression test:")

    # Full-feature errors should remain very small for all variants.
    for key, metrics in results.items():
        print(f"  {key}:")
        print(f"    feature_fro_mean: {metrics['feature_fro_mean']:.6f}")
        print(f"    feature_fro_max: {metrics['feature_fro_max']:.6f}")
        print(f"    roundtrip_joint_l2_mean: {metrics['roundtrip_joint_l2_mean']:.6f}")
        print(f"    roundtrip_joint_l2_max: {metrics['roundtrip_joint_l2_max']:.6f}")
        assert metrics["feature_fro_mean"] < 1e-4, key
        assert metrics["feature_fro_max"] < 1e-3, key
        assert metrics["roundtrip_joint_l2_mean"] < 1e-5, key
        assert metrics["roundtrip_joint_l2_max"] < 1e-4, key

    # FK branch should also stay close to GT positions.
    for key in ("teacher_forced|fk=1", "autoregressive|fk=1"):
        print(f"  {key}:")
        print(f"    fk_joint_l2_mean: {results[key]['fk_joint_l2_mean']:.6f}")
        print(f"    fk_joint_l2_max: {results[key]['fk_joint_l2_max']:.6f}")
        assert results[key]["fk_joint_l2_mean"] < 1e-5, key
        assert results[key]["fk_joint_l2_max"] < 1e-4, key
