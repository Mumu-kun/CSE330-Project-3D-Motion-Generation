"""Evaluate generated_positions_to_271d FK branch against sample-data ground truth.

Runs four variants across a full sequence:
- teacher_forced + fk_offsets=False
- teacher_forced + fk_offsets=True
- autoregressive + fk_offsets=False
- autoregressive + fk_offsets=True

Outputs:
- Per-frame CSV with feature/position/rotation/FK errors
- Console summary table for quick comparison

Usage:
    C:/Python311/python.exe -u scripts/evaluate_generated_positions_to_271d_fk.py \
        --sample-id 000070 \
        --csv output/generated_positions_fk_eval_000070.csv
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.motion_utils import (  # noqa: E402
    features_to_positions,
    generated_positions_to_271d,
    get_fk_offsets,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalVariant:
    mode: str
    use_fk_offsets: bool


def _load_ground_truth(sample_id: str, device: torch.device) -> Dict[str, torch.Tensor]:
    vec_path = ROOT / "sample_data" / f"{sample_id}_vec.npy"
    joint_path = ROOT / "sample_data" / f"{sample_id}_joint.npy"

    if not vec_path.exists():
        raise FileNotFoundError(f"Missing feature file: {vec_path}")
    if not joint_path.exists():
        raise FileNotFoundError(f"Missing joint file: {joint_path}")

    gt_features = torch.from_numpy(np.load(vec_path)).to(
        device=device, dtype=torch.float32
    )
    gt_joint_from_file = torch.from_numpy(np.load(joint_path)).to(
        device=device, dtype=torch.float32
    )

    # Per request: source positions come from feature reconstruction.
    gt_positions = features_to_positions(gt_features)

    return {
        "gt_features": gt_features,
        "gt_positions": gt_positions,
        "gt_joint_from_file": gt_joint_from_file,
    }


def _frame_feature_metrics(
    pred_frame: torch.Tensor, gt_frame: torch.Tensor
) -> Dict[str, float]:
    diff = pred_frame - gt_frame
    return {
        "feature_fro": float(torch.linalg.vector_norm(diff, ord=2).item()),
        "feature_mae": float(diff.abs().mean().item()),
        "feature_max_abs": float(diff.abs().max().item()),
    }


def _rotation_metrics(
    pred_frame: torch.Tensor, gt_frame: torch.Tensor
) -> Dict[str, float]:
    pred_rot = pred_frame[69:201].reshape(22, 6)
    gt_rot = gt_frame[69:201].reshape(22, 6)
    rot_l2 = torch.linalg.vector_norm(pred_rot - gt_rot, ord=2, dim=-1)
    return {
        "rotation_l2_mean": float(rot_l2.mean().item()),
        "rotation_l2_max": float(rot_l2.max().item()),
    }


def _compute_joint_l2_frame(
    pred_positions: torch.Tensor, gt_positions: torch.Tensor
) -> Dict[str, float]:
    # Inputs are (22, 3)
    per_joint = torch.linalg.vector_norm(pred_positions - gt_positions, ord=2, dim=-1)
    return {
        "joint_l2_mean": float(per_joint.mean().item()),
        "joint_l2_max": float(per_joint.max().item()),
        "root_l2": float(per_joint[0].item()),
        "nonroot_joint_l2_mean": float(per_joint[1:].mean().item()),
    }


def _run_variant(
    variant: EvalVariant,
    gt_features: torch.Tensor,
    gt_positions: torch.Tensor,
    fk_offsets: torch.Tensor,
    feet_thre: float,
) -> List[Dict[str, float]]:
    n_frames = gt_features.shape[0]
    rows: List[Dict[str, float]] = []
    generated_frames: List[torch.Tensor] = []

    # AR state is previous predicted global positions.
    ar_prev_positions: Optional[torch.Tensor] = None

    for t in range(n_frames):
        new_positions = gt_positions[t : t + 1]

        if t == 0:
            prev_positions = None
        elif variant.mode == "teacher_forced":
            prev_positions = gt_positions[t - 1 : t]
        elif variant.mode == "autoregressive":
            prev_positions = ar_prev_positions
        else:
            raise ValueError(f"Unknown mode: {variant.mode}")

        frame_out, _, fk_positions = generated_positions_to_271d(
            new_positions=new_positions,
            prev_positions=prev_positions,
            dataset_type="t2m",
            feet_thre=feet_thre,
            fk_offsets=fk_offsets if variant.use_fk_offsets else None,
            normalizer=None,
        )

        pred_frame = frame_out.squeeze(0)
        gt_frame = gt_features[t]

        row: Dict[str, float] = {
            "frame_index": float(t),
            "mode": variant.mode,
            "use_fk_offsets": float(1 if variant.use_fk_offsets else 0),
        }
        row.update(_frame_feature_metrics(pred_frame, gt_frame))
        row.update(_rotation_metrics(pred_frame, gt_frame))

        if fk_positions is not None:
            fk_pos = fk_positions.squeeze(0)
            fk_joint_l2 = torch.linalg.vector_norm(
                fk_pos - gt_positions[t], ord=2, dim=-1
            )
            row["fk_joint_l2_mean"] = float(fk_joint_l2.mean().item())
            row["fk_joint_l2_max"] = float(fk_joint_l2.max().item())
        else:
            row["fk_joint_l2_mean"] = float("nan")
            row["fk_joint_l2_max"] = float("nan")

        generated_frames.append(pred_frame)

        # Update AR state from round-trip reconstruction of generated sequence prefix.
        if variant.mode == "autoregressive":
            seq_pred = torch.stack(generated_frames, dim=0)
            seq_pos = features_to_positions(seq_pred)
            ar_prev_positions = seq_pos[-1:].detach()

        rows.append(row)

    # Sequence-level round-trip position error per frame.
    seq_pred = torch.stack(generated_frames, dim=0)
    seq_pos = features_to_positions(seq_pred)

    cumulative_error = 0.0
    for t in range(n_frames):
        joint_metrics = _compute_joint_l2_frame(seq_pos[t], gt_positions[t])
        rows[t].update(joint_metrics)

        cumulative_error += rows[t]["feature_fro"]
        rows[t]["feature_fro_cumulative"] = cumulative_error
        rows[t]["roundtrip_pos_l2_mean"] = joint_metrics["joint_l2_mean"]
        rows[t]["roundtrip_pos_l2_max"] = joint_metrics["joint_l2_max"]

    return rows


def _write_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_index",
        "mode",
        "use_fk_offsets",
        "feature_fro",
        "feature_mae",
        "feature_max_abs",
        "rotation_l2_mean",
        "rotation_l2_max",
        "joint_l2_mean",
        "joint_l2_max",
        "root_l2",
        "nonroot_joint_l2_mean",
        "fk_joint_l2_mean",
        "fk_joint_l2_max",
        "roundtrip_pos_l2_mean",
        "roundtrip_pos_l2_max",
        "feature_fro_cumulative",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: List[float]) -> float:
    return (
        float(np.mean(np.array(values, dtype=np.float64))) if values else float("nan")
    )


def _format_stat(rows: List[Dict[str, float]], key: str) -> str:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    if not vals:
        return "n/a"
    return f"{_mean(vals):.6f}"


def _summary_block(rows: List[Dict[str, float]]) -> List[str]:
    if not rows:
        return ["No rows."]
    return [
        f"frames={len(rows)}",
        f"feature_fro_mean={_format_stat(rows, 'feature_fro')}",
        f"feature_mae_mean={_format_stat(rows, 'feature_mae')}",
        f"rotation_l2_mean={_format_stat(rows, 'rotation_l2_mean')}",
        f"joint_l2_mean={_format_stat(rows, 'joint_l2_mean')}",
        f"root_l2_mean={_format_stat(rows, 'root_l2')}",
        f"fk_joint_l2_mean={_format_stat(rows, 'fk_joint_l2_mean')}",
        f"roundtrip_pos_l2_mean={_format_stat(rows, 'roundtrip_pos_l2_mean')}",
        f"final_feature_fro_cumulative={rows[-1]['feature_fro_cumulative']:.6f}",
    ]


def _print_summary(
    all_rows: List[Dict[str, float]], sample_id: str, gt_delta: float
) -> None:
    print("\n" + "=" * 84)
    print(f"generated_positions_to_271d FK evaluation | sample={sample_id}")
    print("=" * 84)
    print("Ground-truth sanity: reconstructed_positions vs *_joint.npy")
    print(f"  mean_joint_l2={gt_delta:.6f}")

    groups: Dict[str, List[Dict[str, float]]] = {}
    for row in all_rows:
        key = f"{row['mode']}|fk={int(row['use_fk_offsets'])}"
        groups.setdefault(key, []).append(row)

    for key in sorted(groups.keys()):
        print("-" * 84)
        print(key)
        for line in _summary_block(groups[key]):
            print(f"  {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated_positions_to_271d for FK offsets branch"
    )
    parser.add_argument(
        "--sample-id", default="000070", help="Sample id from sample_data"
    )
    parser.add_argument(
        "--csv",
        default="output/generated_positions_fk_eval_000070.csv",
        help="CSV output path",
    )
    parser.add_argument("--feet-thre", type=float, default=0.002)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    payload = _load_ground_truth(args.sample_id, device=device)
    gt_features = payload["gt_features"]
    gt_positions = payload["gt_positions"]
    gt_joint_from_file = payload["gt_joint_from_file"]

    gt_delta = torch.linalg.vector_norm(
        gt_positions - gt_joint_from_file, ord=2, dim=-1
    )
    gt_delta_mean = float(gt_delta.mean().item())

    fk_offsets = get_fk_offsets(gt_positions.unsqueeze(0))  # (1, 22, 3)

    variants = [
        EvalVariant(mode="teacher_forced", use_fk_offsets=False),
        EvalVariant(mode="teacher_forced", use_fk_offsets=True),
        EvalVariant(mode="autoregressive", use_fk_offsets=False),
        EvalVariant(mode="autoregressive", use_fk_offsets=True),
    ]

    all_rows: List[Dict[str, float]] = []
    for variant in variants:
        rows = _run_variant(
            variant=variant,
            gt_features=gt_features,
            gt_positions=gt_positions,
            fk_offsets=fk_offsets,
            feet_thre=args.feet_thre,
        )
        all_rows.extend(rows)

    csv_path = (ROOT / args.csv).resolve()
    _write_csv(all_rows, csv_path)

    _print_summary(all_rows, sample_id=args.sample_id, gt_delta=gt_delta_mean)
    print("-" * 84)
    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
