from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Config
from utils.checkpoint_eval import (
    ScenarioSpec,
    aggregate_position_metrics,
    build_default_scenarios,
    compute_drift_metrics,
    config_to_jsonable,
    joint_name,
    load_checkpoint_config,
    load_checkpoint_payload,
    load_feature_normalizer,
    load_generator_from_checkpoint,
    load_sample,
    load_text_embedding_cache,
    mean_per_joint_l2,
    per_joint_l2,
    predict_next_positions,
    resolve_dataset_path,
    sample_length,
    select_sample_ids_by_length_quantiles,
)
from utils.motion_utils import generated_positions_to_271d
from utils.visualization import plot_3d_motion


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float:
    return float(value) if not isinstance(value, bool) else float(int(value))


def _numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    keys: list[str] = []
    for key in rows[0].keys():
        values = [row.get(key) for row in rows]
        if all(
            isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
            for value in values
        ):
            keys.append(key)
    return keys


def _build_per_frame_rows(
    *,
    sample_id: str,
    scenario: ScenarioSpec,
    pred_eval: torch.Tensor,
    gt_eval: torch.Tensor,
    eval_start_frame: int,
    teacher_force_mask: list[bool] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_idx in range(pred_eval.shape[0]):
        pred_frame = pred_eval[local_idx]
        gt_frame = gt_eval[local_idx]
        frame_metrics = aggregate_position_metrics(pred_frame, gt_frame)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "scenario": scenario.name,
            "scenario_mode": scenario.mode,
            "seed_length": scenario.seed_length,
            "num_steps": scenario.num_steps,
            "scenario_frame_index": local_idx,
            "global_frame_index": eval_start_frame + local_idx,
        }
        row.update(frame_metrics)
        if teacher_force_mask is not None:
            row["teacher_force_used"] = int(teacher_force_mask[local_idx])
        rows.append(row)
    return rows


def _build_per_joint_rows(
    *,
    sample_id: str,
    scenario: ScenarioSpec,
    per_joint_mean_l2: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint_index, value in enumerate(per_joint_mean_l2.tolist()):
        rows.append(
            {
                "sample_id": sample_id,
                "scenario": scenario.name,
                "scenario_mode": scenario.mode,
                "seed_length": scenario.seed_length,
                "num_steps": scenario.num_steps,
                "joint_index": joint_index,
                "joint_name": joint_name(joint_index),
                "mean_l2": float(value),
            }
        )
    return rows


def _evaluate_scenario_outputs(
    *,
    sample_id: str,
    caption: str,
    sample_length_value: int,
    scenario: ScenarioSpec,
    predicted_sequence: torch.Tensor,
    ground_truth_sequence: torch.Tensor,
    eval_start_frame: int,
    teacher_force_mask: list[bool] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_start_frame >= sample_length_value:
        raise ValueError(
            f"Scenario '{scenario.name}' has eval_start_frame={eval_start_frame} "
            f"for sample length {sample_length_value}."
        )

    pred_eval = predicted_sequence[eval_start_frame:]
    gt_eval = ground_truth_sequence[eval_start_frame:]
    frame_joint_l2 = per_joint_l2(pred_eval, gt_eval)
    frame_joint_l2_mean = frame_joint_l2.mean(dim=-1)
    scenario_metrics = aggregate_position_metrics(pred_eval, gt_eval)
    scenario_metrics.update(compute_drift_metrics(frame_joint_l2_mean))

    scenario_row: dict[str, Any] = {
        "sample_id": sample_id,
        "caption": caption,
        "sample_length": sample_length_value,
        "scenario": scenario.name,
        "scenario_mode": scenario.mode,
        "seed_length": scenario.seed_length,
        "num_steps": scenario.num_steps,
        "eval_start_frame": eval_start_frame,
        "num_eval_frames": int(pred_eval.shape[0]),
        "teacher_force_probability": (
            float(scenario.teacher_force_prob)
            if scenario.teacher_force_prob is not None
            else 0.0
        ),
        "teacher_force_count": (
            int(sum(teacher_force_mask)) if teacher_force_mask is not None else 0
        ),
        "teacher_force_ratio": (
            float(sum(teacher_force_mask) / max(1, len(teacher_force_mask)))
            if teacher_force_mask is not None
            else 0.0
        ),
    }
    scenario_row.update(scenario_metrics)

    aligned_mask = teacher_force_mask
    per_frame_rows = _build_per_frame_rows(
        sample_id=sample_id,
        scenario=scenario,
        pred_eval=pred_eval,
        gt_eval=gt_eval,
        eval_start_frame=eval_start_frame,
        teacher_force_mask=aligned_mask,
    )
    per_joint_rows = _build_per_joint_rows(
        sample_id=sample_id,
        scenario=scenario,
        per_joint_mean_l2=frame_joint_l2.mean(dim=0),
    )
    return scenario_row, per_frame_rows, per_joint_rows


def _run_teacher_forced_one_step(
    generator,
    motion_norm: torch.Tensor,
    joints: torch.Tensor,
    text_clip: torch.Tensor,
    scenario: ScenarioSpec,
) -> tuple[torch.Tensor, int, list[bool] | None]:
    positions = [joints[:, 0]]
    with torch.no_grad():
        h_state = None
        for frame_idx in range(joints.shape[1] - 1):
            current_frame_norm = motion_norm[:, frame_idx]
            current_positions = joints[:, frame_idx]
            context, h_state = generator.encoder.gru_step(
                current_frame_norm,
                text_clip[:, 0, :],
                h_state,
            )
            pred_positions, _ = predict_next_positions(
                generator=generator,
                context=context,
                current_positions=current_positions,
                current_frame_norm=current_frame_norm,
                text_embedding=text_clip[:, 0, :],
                num_steps=scenario.num_steps,
            )
            positions.append(pred_positions)
    return torch.stack(positions, dim=1), 1, None


def _run_mixed_teacher_force_rollout(
    generator,
    motion_norm: torch.Tensor,
    joints: torch.Tensor,
    text_clip: torch.Tensor,
    scenario: ScenarioSpec,
) -> tuple[torch.Tensor, int, list[bool]]:
    positions = [joints[:, 0]]
    teacher_force_mask: list[bool] = []
    with torch.no_grad():
        h_state = None
        current_positions = joints[:, 0]
        current_frame = motion_norm[:, 0]

        for frame_idx in range(joints.shape[1] - 1):
            context, h_state = generator.encoder.gru_step(
                current_frame,
                text_clip[:, 0, :],
                h_state,
            )
            pred_positions, _ = predict_next_positions(
                generator=generator,
                context=context,
                current_positions=current_positions,
                current_frame_norm=current_frame,
                text_embedding=text_clip[:, 0, :],
                num_steps=scenario.num_steps,
            )
            gt_next_positions = joints[:, frame_idx + 1]
            use_teacher_force = bool(
                torch.rand(1, device=joints.device).item()
                < float(scenario.teacher_force_prob)
            )
            teacher_force_mask.append(use_teacher_force)
            next_positions = gt_next_positions if use_teacher_force else pred_positions
            positions.append(next_positions)

            if use_teacher_force:
                next_frame = motion_norm[:, frame_idx + 1]
            else:
                next_frame, _, _ = generated_positions_to_271d(
                    new_positions=next_positions,
                    prev_positions=current_positions,
                    dataset_type="t2m",
                    normalizer=generator.normalizer,
                )
            current_positions = next_positions
            current_frame = next_frame

    return torch.stack(positions, dim=1), 1, teacher_force_mask


def _run_rollout_scenario(
    generator,
    joints: torch.Tensor,
    text_clip: torch.Tensor,
    scenario: ScenarioSpec,
) -> tuple[torch.Tensor, int, list[bool] | None]:
    sample_length_value = int(joints.shape[1])
    seed_length = min(max(1, scenario.seed_length), sample_length_value - 1)
    rollout_frames = sample_length_value - seed_length
    with torch.no_grad():
        rollout_positions, _, _ = generator.generate_sequence(
            text=text_clip,
            num_frames=rollout_frames,
            num_steps=scenario.num_steps,
            horizon=seed_length,
            input_positions=joints[:, :seed_length],
            guidance_scale=1.0,
            dataset_type="t2m",
            use_fk=False,
        )
    return rollout_positions, seed_length, None


def run_scenario(
    *,
    generator,
    sample: dict[str, Any],
    motion_norm: torch.Tensor,
    scenario: ScenarioSpec,
) -> tuple[torch.Tensor, int, list[bool] | None]:
    joints = sample["joints"]
    text_clip = sample["text_clip"]

    if scenario.mode == "teacher_forced_one_step":
        return _run_teacher_forced_one_step(
            generator=generator,
            motion_norm=motion_norm,
            joints=joints,
            text_clip=text_clip,
            scenario=scenario,
        )
    if scenario.mode == "mixed_teacher_force_rollout":
        return _run_mixed_teacher_force_rollout(
            generator=generator,
            motion_norm=motion_norm,
            joints=joints,
            text_clip=text_clip,
            scenario=scenario,
        )
    if scenario.mode == "rollout":
        return _run_rollout_scenario(
            generator=generator,
            joints=joints,
            text_clip=text_clip,
            scenario=scenario,
        )
    raise ValueError(f"Unknown scenario mode: {scenario.mode}")


def aggregate_scenario_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    scenario_aggregates: dict[str, dict[str, Any]] = {}
    for scenario_name, scenario_rows in grouped.items():
        numeric_keys = _numeric_keys(scenario_rows)
        aggregate: dict[str, Any] = {
            "sample_count": len(scenario_rows),
            "sample_ids": [str(row["sample_id"]) for row in scenario_rows],
        }
        for key in numeric_keys:
            aggregate[key] = float(
                np.mean([_to_float(row[key]) for row in scenario_rows])
            )
        scenario_aggregates[scenario_name] = aggregate
    return scenario_aggregates


def aggregate_top_joints(
    per_joint_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in per_joint_rows:
        scenario_name = str(row["scenario"])
        joint_index = int(row["joint_index"])
        grouped.setdefault(scenario_name, {}).setdefault(joint_index, []).append(
            float(row["mean_l2"])
        )

    out: dict[str, list[dict[str, Any]]] = {}
    for scenario_name, joint_map in grouped.items():
        ranked = []
        for joint_index, values in joint_map.items():
            ranked.append(
                {
                    "joint_index": joint_index,
                    "joint_name": joint_name(joint_index),
                    "mean_l2": float(np.mean(values)),
                }
            )
        ranked.sort(key=lambda row: row["mean_l2"], reverse=True)
        out[scenario_name] = ranked
    return out


def apply_teacher_forced_deltas(rows: list[dict[str, Any]]) -> None:
    delta_keys = [
        "joint_mse",
        "joint_mae",
        "joint_l2_mean",
        "root_l2_mean",
        "nonroot_joint_l2_mean",
        "first_third_joint_l2_mean",
        "last_third_joint_l2_mean",
        "drift_delta",
        "final_joint_l2_mean",
        "worst_frame_joint_l2_mean",
    ]
    baselines = {
        str(row["sample_id"]): row
        for row in rows
        if row["scenario"] == "teacher_forced_one_step"
    }
    for row in rows:
        baseline = baselines[str(row["sample_id"])]
        for key in delta_keys:
            row[f"delta_vs_teacher_forced_{key}"] = float(row[key] - baseline[key])


def render_report(summary: dict[str, Any]) -> str:
    checkpoint = summary["checkpoint"]
    config = summary["config"]
    scenario_aggregates = summary["scenario_aggregates"]
    top_joints = summary["scenario_top_joints"]
    selected_samples = summary["selected_samples"]

    teacher = scenario_aggregates["teacher_forced_one_step"]
    rollout = scenario_aggregates["rollout_default"]
    mixed = scenario_aggregates["mixed_teacher_force_rollout"]
    short_context = scenario_aggregates["rollout_short_context"]
    high_steps = scenario_aggregates["rollout_high_steps"]

    findings: list[str] = []
    improvements: list[str] = []

    if rollout["joint_l2_mean"] > max(
        teacher["joint_l2_mean"] * 3.0, teacher["joint_l2_mean"] + 0.1
    ):
        findings.append(
            "Rollout drift dominates: `rollout_default` is much worse than "
            f"`teacher_forced_one_step` ({rollout['joint_l2_mean']:.4f} vs {teacher['joint_l2_mean']:.4f})."
        )
        improvements.append(
            "Increase exposure to generated states during training by revisiting "
            "`rollout_prob_start`, `rollout_prob_end`, and longer rollout branches."
        )

    if mixed["joint_l2_mean"] < rollout["joint_l2_mean"] * 0.5:
        findings.append(
            "`mixed_teacher_force_rollout` recovers much of the gap, which points to exposure bias "
            "more than pure one-step modeling error."
        )
        improvements.append(
            "Prioritize rollout robustness and consistency-style objectives before spending time on "
            "small decoder architecture tweaks."
        )

    if short_context["joint_l2_mean"] > rollout["joint_l2_mean"] * 1.05:
        findings.append(
            "Reducing rollout context hurts quality, so the model benefits from a longer conditioning window."
        )
        improvements.append(
            "If shorter seeds are important, add more training coverage for low-history regimes or use a shorter-context validation target explicitly."
        )

    if high_steps["joint_l2_mean"] < rollout["joint_l2_mean"] * 0.95:
        findings.append(
            "More ODE steps help somewhat, so inference integration error is part of the rollout gap."
        )
        improvements.append(
            "Keep a higher-step inference option available for analysis runs and check whether training with more rollout integration steps narrows the gap further."
        )
    elif high_steps["joint_l2_mean"] > rollout["joint_l2_mean"] * 1.05:
        findings.append(
            "More ODE steps do not help, which suggests the main issue is model drift rather than solver resolution."
        )
        improvements.append(
            "Do not prioritize larger inference-step counts as the primary fix; focus on state robustness and training-time rollout behavior."
        )

    root_ratio = rollout["root_l2_mean"] / max(rollout["nonroot_joint_l2_mean"], 1e-8)
    if root_ratio > 1.2:
        findings.append(
            "Root motion errors are larger than non-root errors in `rollout_default`."
        )
        improvements.append(
            "Inspect root velocity and delta-yaw behavior, since root motion is likely amplifying downstream drift."
        )
    elif root_ratio < (1.0 / 1.2):
        findings.append(
            "Non-root articulation errors dominate over root motion in `rollout_default`."
        )
        improvements.append(
            "Inspect limb articulation quality and the joints with the highest rollout error before adjusting root-motion losses."
        )

    rollout_top_joints = top_joints.get("rollout_default", [])[:3]
    if not findings:
        findings.append("No single failure mode dominated the default scenario matrix.")
    if not improvements:
        improvements.append(
            "The evaluator did not detect a dominant correction path; inspect per-sample outputs directly."
        )

    lines = [
        "# Checkpoint Scenario Evaluation",
        "",
        "## Summary",
        f"- Checkpoint: `{checkpoint['path']}`",
        f"- Split: `{summary['split']}`",
        f"- Samples: {', '.join(sample['sample_id'] for sample in selected_samples)}",
        f"- Embedded horizon: `{config['horizon']}`",
        f"- Embedded num_inference_steps: `{config['num_inference_steps']}`",
        f"- Embedded use_fk: `{config['use_fk']}`",
        "",
        "## Findings",
    ]
    lines.extend([f"- {finding}" for finding in findings])
    lines.extend(
        [
            "",
            "## Top Failure Joints",
        ]
    )
    if rollout_top_joints:
        lines.extend(
            [
                f"- `{row['joint_name']}` (joint {row['joint_index']}): mean_l2={row['mean_l2']:.4f}"
                for row in rollout_top_joints
            ]
        )
    else:
        lines.append("- No joint ranking data was available.")
    lines.extend(
        [
            "",
            "## Likely Improvements",
        ]
    )
    lines.extend([f"- {item}" for item in improvements])
    return "\n".join(lines) + "\n"


def run_checkpoint_scenario_evaluation(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    split: str = "test",
    num_samples: int = 3,
    sample_ids: list[str] | None = None,
    device: str = "cpu",
    seed: int = 0,
    render_videos: bool = True,
) -> dict[str, Any]:
    _seed_everything(seed)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_config = load_checkpoint_config(checkpoint_path, map_location=device)
    resolved_dataset_path = resolve_dataset_path(
        checkpoint_path,
        dataset_path=dataset_path,
        map_location=device,
    )
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else PROJECT_ROOT / "output" / "checkpoint_evaluation" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = load_feature_normalizer(resolved_dataset_path, device=device)
    generator = load_generator_from_checkpoint(
        checkpoint_path,
        dataset_path=resolved_dataset_path,
        device=device,
        normalizer=normalizer,
    )
    generator_config: Config = generator.config
    scenarios = build_default_scenarios(generator_config)

    if sample_ids is None:
        selected_sample_ids = select_sample_ids_by_length_quantiles(
            resolved_dataset_path,
            split=split,
            num_samples=num_samples,
        )
    else:
        selected_sample_ids = list(dict.fromkeys(sample_ids))

    text_cache = load_text_embedding_cache(resolved_dataset_path)
    representative_sample_id = max(
        selected_sample_ids,
        key=lambda sample_id: sample_length(resolved_dataset_path, sample_id),
    )

    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    selected_samples: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    per_joint_rows: list[dict[str, Any]] = []
    representative_sequences: dict[str, torch.Tensor] = {}
    representative_caption = ""

    for sample_id in selected_sample_ids:
        sample = load_sample(
            resolved_dataset_path,
            sample_id,
            device=device,
            text_cache=text_cache,
        )
        selected_samples.append(
            {
                "sample_id": sample_id,
                "sample_length": int(sample["sample_length"]),
                "caption": str(sample["caption"]),
            }
        )

        motion_norm = normalizer.normalize(sample["motion"])
        gt_sequence = sample["joints"].squeeze(0)

        if sample_id == representative_sample_id:
            representative_caption = str(sample["caption"])
            representative_sequences["ground_truth"] = gt_sequence.detach().cpu()

        for scenario in scenarios:
            predicted_sequence, eval_start_frame, teacher_force_mask = run_scenario(
                generator=generator,
                sample=sample,
                motion_norm=motion_norm,
                scenario=scenario,
            )
            predicted_unbatched = predicted_sequence.squeeze(0)
            scenario_row, frame_rows, joint_rows = _evaluate_scenario_outputs(
                sample_id=sample_id,
                caption=str(sample["caption"]),
                sample_length_value=int(sample["sample_length"]),
                scenario=scenario,
                predicted_sequence=predicted_unbatched,
                ground_truth_sequence=gt_sequence,
                eval_start_frame=eval_start_frame,
                teacher_force_mask=teacher_force_mask,
            )
            scenario_rows.append(scenario_row)
            per_frame_rows.extend(frame_rows)
            per_joint_rows.extend(joint_rows)

            if sample_id == representative_sample_id and scenario.name in {
                "teacher_forced_one_step",
                "rollout_default",
            }:
                representative_sequences[scenario.name] = (
                    predicted_unbatched.detach().cpu()
                )

    apply_teacher_forced_deltas(scenario_rows)
    scenario_aggregates = aggregate_scenario_rows(scenario_rows)
    scenario_top_joints = aggregate_top_joints(per_joint_rows)

    summary = {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "stem": checkpoint_path.stem,
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
            "loss": payload.get("loss"),
        },
        "dataset_path": str(resolved_dataset_path.resolve()),
        "split": split,
        "seed": seed,
        "selected_samples": selected_samples,
        "config": config_to_jsonable(checkpoint_config),
        "scenario_definitions": [asdict(scenario) for scenario in scenarios],
        "scenario_metrics": scenario_rows,
        "scenario_aggregates": scenario_aggregates,
        "scenario_top_joints": scenario_top_joints,
    }

    report_text = render_report(summary)
    summary["report_preview"] = report_text.splitlines()[:12]

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv_rows(output_dir / "scenario_metrics.csv", scenario_rows)
    _write_csv_rows(output_dir / "per_frame_metrics.csv", per_frame_rows)
    _write_csv_rows(output_dir / "per_joint_metrics.csv", per_joint_rows)
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")

    if render_videos:
        fps = float(getattr(generator_config, "fps", 20))
        video_items = [
            ("ground_truth", "Ground Truth"),
            ("teacher_forced_one_step", "Teacher-Forced One-Step"),
            ("rollout_default", "Rollout Default"),
        ]
        for key, title in video_items:
            if key not in representative_sequences:
                continue
            plot_3d_motion(
                motion=representative_sequences[key].numpy(),
                fps=fps,
                title=f"{title} | {representative_sample_id} | {representative_caption}",
                save_path=output_dir / f"{key}_{representative_sample_id}.mp4",
            )

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint across a small scenario matrix."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset root. Defaults to the embedded checkpoint config path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to output/checkpoint_evaluation/<checkpoint_stem>.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of deterministic length-quantile samples to evaluate.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Explicit sample ids to evaluate instead of split-based selection.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation RNG seed.")
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip MP4 rendering and only write numeric/report artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_checkpoint_scenario_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        sample_ids=args.sample_ids,
        device=args.device,
        seed=args.seed,
        render_videos=not args.skip_videos,
    )
    print(
        f"Wrote evaluation bundle to {args.output_dir or 'default output directory'}."
    )
    print(json.dumps(summary["scenario_aggregates"], indent=2))


if __name__ == "__main__":
    main()
