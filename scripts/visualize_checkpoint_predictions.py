from __future__ import annotations

"""Visualize one-step and rollout predictions from a saved checkpoint."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import HumanMotionGenerator
from utils.checkpoint_eval import (
    aggregate_position_metrics,
    joint_name,
    load_feature_normalizer as shared_load_feature_normalizer,
    load_generator_from_checkpoint as shared_load_generator_from_checkpoint,
    predict_next_positions as shared_predict_next_positions,
)
from utils.dataset import Text2MotionDataset, text2motion_collate_fn
from utils.motion_utils import FeatureNormalizer, generated_positions_to_271d
from utils.visualization import plot_3d_motion

PREDICTOR_STEPS = 5
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "checkpoints"
    / "latest.pt"
    # / "before_submit"
    # / "best_val_noss_fullset.pt"
)
DATASET_PATH = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "checkpoint_visualizations"
    / (CHECKPOINT_PATH.stem + f"_{PREDICTOR_STEPS*2}_steps")
)
MASKED_TEACHER_FORCE_PROB = 0.4


def _load_feature_normalizer(
    dataset_path: str | Path,
    device: str | torch.device = "cpu",
) -> FeatureNormalizer:
    return shared_load_feature_normalizer(dataset_path, device=device)


def _load_generator_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_path: str | Path,
    device: str | torch.device = "cpu",
    normalizer: FeatureNormalizer | None = None,
) -> HumanMotionGenerator:
    return shared_load_generator_from_checkpoint(
        checkpoint_path,
        dataset_path=dataset_path,
        device=device,
        normalizer=normalizer,
    )


def _load_long_sample(
    config,
    normalizer: FeatureNormalizer,
) -> dict[str, torch.Tensor | str | int]:
    dataset = Text2MotionDataset(
        config=config,
        mean=normalizer.mean.cpu().numpy(),
        std=normalizer.std.cpu().numpy(),
        split="train",
    )
    min_required_len = int(config.horizon) + 8
    candidate_indices = np.where(dataset.length_arr >= min_required_len)[0]
    if len(candidate_indices) == 0:
        raise RuntimeError(
            f"No sample in {DATASET_PATH} is long enough for horizon={config.horizon}."
        )

    sample_index = int(candidate_indices[-3])
    sample_name = dataset.name_list[sample_index]
    sample_len = int(dataset.data_dict[sample_name]["length"])

    dataset.set_horizon(sample_len)
    relative_index = sample_index - dataset.pointer
    if relative_index < 0:
        raise RuntimeError(
            f"Selected sample '{sample_name}' fell outside the filtered horizon view."
        )
    batch = text2motion_collate_fn([dataset[relative_index]])
    batch["sample_index"] = sample_index
    batch["sample_name"] = sample_name
    batch["sample_length"] = sample_len
    return batch


def _predict_next_positions(
    *,
    generator: HumanMotionGenerator,
    context: torch.Tensor,
    current_positions: torch.Tensor,
    current_frame_norm: torch.Tensor,
    text_embedding: torch.Tensor,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return shared_predict_next_positions(
        generator=generator,
        context=context,
        current_positions=current_positions,
        current_frame_norm=current_frame_norm,
        text_embedding=text_embedding,
        num_steps=num_steps,
    )


def _joint_metrics(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> dict[str, float]:
    return aggregate_position_metrics(pred_positions, target_positions)


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    flat = values.astype(np.float64, copy=False).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "abs_mean": float(np.abs(flat).mean()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "min": float(flat.min()),
        "p10": float(np.quantile(flat, 0.10)),
        "median": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "max": float(flat.max()),
    }


def _noise_candidates(base_std: float) -> dict[str, float]:
    return {
        "0.5pct": float(base_std * 0.005),
        "1pct": float(base_std * 0.01),
        "2pct": float(base_std * 0.02),
        "5pct": float(base_std * 0.05),
    }


def _summarize_history_contexts(
    history_contexts: torch.Tensor,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if history_contexts.ndim != 4:
        raise ValueError(
            "Expected history contexts shape (B, T, J, D), "
            f"got {tuple(history_contexts.shape)}."
        )

    contexts_np = history_contexts.detach().cpu().float().numpy()
    temporal_std_per_joint_dim = contexts_np.std(axis=1, dtype=np.float64)
    per_frame_value_std = contexts_np.std(axis=(2, 3), dtype=np.float64)
    per_joint_temporal_std_mean = temporal_std_per_joint_dim.mean(axis=-1)
    mean_joint_std = per_joint_temporal_std_mean.mean(axis=0)
    top_joint_indices = np.argsort(mean_joint_std)[::-1][:5]

    value_std = float(contexts_np.std(dtype=np.float64))
    temporal_std_mean = float(temporal_std_per_joint_dim.mean())

    arrays = {
        "history_contexts": (
            np.squeeze(contexts_np, axis=0)
            if contexts_np.shape[0] == 1
            else contexts_np
        ),
        "temporal_std_per_joint_dim": (
            np.squeeze(temporal_std_per_joint_dim, axis=0)
            if temporal_std_per_joint_dim.shape[0] == 1
            else temporal_std_per_joint_dim
        ),
        "per_frame_value_std": (
            np.squeeze(per_frame_value_std, axis=0)
            if per_frame_value_std.shape[0] == 1
            else per_frame_value_std
        ),
        "per_joint_temporal_std_mean": (
            np.squeeze(per_joint_temporal_std_mean, axis=0)
            if per_joint_temporal_std_mean.shape[0] == 1
            else per_joint_temporal_std_mean
        ),
    }

    summary: dict[str, object] = {
        "shape": list(history_contexts.shape),
        "value_distribution": _distribution_stats(contexts_np),
        "per_frame_value_std_distribution": _distribution_stats(per_frame_value_std),
        "temporal_std_per_joint_dim_distribution": _distribution_stats(
            temporal_std_per_joint_dim
        ),
        "per_joint_temporal_std_mean_distribution": _distribution_stats(
            per_joint_temporal_std_mean
        ),
        "per_joint_temporal_std_mean": {
            joint_name(int(joint_idx)): float(joint_std)
            for joint_idx, joint_std in enumerate(mean_joint_std.tolist())
        },
        "top_temporal_std_joints": [
            {
                "joint_index": int(joint_idx),
                "joint_name": joint_name(int(joint_idx)),
                "mean_temporal_std": float(mean_joint_std[joint_idx]),
            }
            for joint_idx in top_joint_indices.tolist()
        ],
        "small_gaussian_noise_heuristics": {
            "based_on_value_std": _noise_candidates(value_std),
            "based_on_temporal_std_mean": _noise_candidates(temporal_std_mean),
        },
    }
    return summary, arrays


def _save_motion_video(
    motion: torch.Tensor,
    title: str,
    save_path: Path,
    fps: float,
) -> None:
    plot_3d_motion(
        motion=motion.detach().cpu().numpy(),
        fps=fps,
        title=title,
        save_path=save_path,
    )


def _mixed_teacher_force_rollout(
    generator: HumanMotionGenerator,
    joints: torch.Tensor,
    motion_norm: torch.Tensor,
    text_clip: torch.Tensor,
    num_steps: int,
    teacher_force_prob: float,
) -> tuple[torch.Tensor, list[bool], list[dict[str, float]]]:
    mixed_positions = [joints[:, 0]]
    mixed_metrics: list[dict[str, float]] = []
    teacher_force_mask: list[bool] = []

    with torch.no_grad():
        frame_buffer = motion_norm[:, :0]
        cache_state = None
        current_positions = joints[:, 0]
        current_frame = motion_norm[:, 0]

        for frame_idx in range(joints.shape[1] - 1):
            context, frame_buffer, cache_state = generator.encoder.step(
                current_frame,
                text_clip[:, 0, :],
                frame_buffer=frame_buffer,
                cache_state=cache_state,
            )
            pred_positions, _ = _predict_next_positions(
                generator=generator,
                context=context,
                current_positions=current_positions,
                current_frame_norm=current_frame,
                text_embedding=text_clip[:, 0, :],
                num_steps=num_steps,
            )
            gt_next_positions = joints[:, frame_idx + 1]

            use_teacher_force = bool(torch.rand(1).item() < teacher_force_prob)
            teacher_force_mask.append(use_teacher_force)
            next_positions = gt_next_positions if use_teacher_force else pred_positions
            mixed_positions.append(next_positions)
            mixed_metrics.append(_joint_metrics(next_positions, gt_next_positions))

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

    return torch.cat(mixed_positions, dim=0), teacher_force_mask, mixed_metrics


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalizer = _load_feature_normalizer(DATASET_PATH, device="cpu")
    generator = _load_generator_from_checkpoint(
        CHECKPOINT_PATH,
        dataset_path=DATASET_PATH,
        device="cpu",
        normalizer=normalizer,
    )
    config = generator.config

    batch = _load_long_sample(config, normalizer)
    sample_name = str(batch["sample_name"])
    sample_len = int(batch["sample_length"])
    caption = str(batch["captions"][0])

    motion_raw = batch["motion"].to(config.device)
    joints = batch["joints"].to(config.device)
    text_clip = batch["text_clip"].to(config.device)
    motion_norm = normalizer.normalize(motion_raw)

    with torch.no_grad():
        history_contexts = generator.encoder(
            motion_norm[:, :-1],
            text_clip[:, 0, :],
            return_all=True,
        )
    history_context_summary, history_context_arrays = _summarize_history_contexts(
        history_contexts
    )

    horizon = int(config.horizon)
    inference_steps = max(
        1,
        int(getattr(config, "num_inference_steps", PREDICTOR_STEPS)),
    )
    inference_steps = PREDICTOR_STEPS

    teacher_forced_positions = [joints[:, 0]]
    teacher_forced_metrics: list[dict[str, float]] = []
    teacher_forced_flows = []
    with torch.no_grad():
        frame_buffer = motion_norm[:, :0]
        cache_state = None
        for frame_idx in range(sample_len - 1):
            current_frame_norm = motion_norm[:, frame_idx]
            current_positions = joints[:, frame_idx]
            context, frame_buffer, cache_state = generator.encoder.step(
                current_frame_norm,
                text_clip[:, 0, :],
                frame_buffer=frame_buffer,
                cache_state=cache_state,
            )
            pred_positions, pred_flow_raw = _predict_next_positions(
                generator=generator,
                context=context,
                current_positions=current_positions,
                current_frame_norm=current_frame_norm,
                text_embedding=text_clip[:, 0, :],
                num_steps=PREDICTOR_STEPS,
            )
            teacher_forced_positions.append(pred_positions)
            teacher_forced_metrics.append(
                _joint_metrics(pred_positions, joints[:, frame_idx + 1])
            )
            teacher_forced_flows.append(pred_flow_raw.squeeze(0).cpu().numpy())

    teacher_forced_sequence = torch.cat(teacher_forced_positions, dim=0)

    if sample_len <= horizon:
        raise RuntimeError(
            f"Sample '{sample_name}' length {sample_len} is not longer than horizon {horizon}."
        )

    seed_positions = joints[:, :10]
    rollout_frames = sample_len - 10
    with torch.no_grad():
        rollout_positions, _, _ = generator.generate_sequence(
            text=text_clip,
            num_frames=rollout_frames,
            num_steps=PREDICTOR_STEPS,
            horizon=horizon,
            input_positions=seed_positions,
            guidance_scale=1.0,
            dataset_type="t2m",
            use_fk=False,
        )

    rollout_future = rollout_positions[:, horizon:]
    rollout_gt_future = joints[:, horizon:]
    rollout_metrics = _joint_metrics(rollout_future, rollout_gt_future)
    rollout_step_metrics = [
        _joint_metrics(rollout_future[:, i], rollout_gt_future[:, i])
        for i in range(rollout_future.shape[1])
    ]

    masked_sequence, masked_teacher_force_mask, masked_metrics = (
        _mixed_teacher_force_rollout(
            generator=generator,
            joints=joints,
            motion_norm=motion_norm,
            text_clip=text_clip,
            num_steps=PREDICTOR_STEPS,
            teacher_force_prob=MASKED_TEACHER_FORCE_PROB,
        )
    )

    np.save(OUTPUT_DIR / "ground_truth_full.npy", joints.squeeze(0).cpu().numpy())
    np.save(
        OUTPUT_DIR / "teacher_forced_one_step_full.npy",
        teacher_forced_sequence.cpu().numpy(),
    )
    np.save(
        OUTPUT_DIR / "teacher_forced_one_step_flows_68d.npy",
        np.stack(teacher_forced_flows, axis=0),
    )
    np.save(
        OUTPUT_DIR / "rollout_full_sequence.npy",
        rollout_positions.squeeze(0).cpu().numpy(),
    )
    np.save(
        OUTPUT_DIR / "masked_teacher_force_rollout_full.npy",
        masked_sequence.cpu().numpy(),
    )
    np.save(
        OUTPUT_DIR / "masked_teacher_force_rollout_mask.npy",
        np.asarray(masked_teacher_force_mask, dtype=np.bool_),
    )
    np.save(
        OUTPUT_DIR / "teacher_forced_encoder_history_contexts.npy",
        history_context_arrays["history_contexts"],
    )
    np.save(
        OUTPUT_DIR
        / "teacher_forced_encoder_history_context_temporal_std_per_joint_dim.npy",
        history_context_arrays["temporal_std_per_joint_dim"],
    )
    np.save(
        OUTPUT_DIR / "teacher_forced_encoder_history_context_per_frame_value_std.npy",
        history_context_arrays["per_frame_value_std"],
    )
    np.save(
        OUTPUT_DIR
        / "teacher_forced_encoder_history_context_per_joint_temporal_std_mean.npy",
        history_context_arrays["per_joint_temporal_std_mean"],
    )

    fps = float(getattr(config, "fps", 20))
    _save_motion_video(
        motion=joints.squeeze(0),
        title=f"Ground Truth | {sample_name}",
        save_path=OUTPUT_DIR / "ground_truth_full.mp4",
        fps=fps,
    )
    _save_motion_video(
        motion=teacher_forced_sequence,
        title=f"Teacher-Forced One-Step | {sample_name}",
        save_path=OUTPUT_DIR / "teacher_forced_one_step_full.mp4",
        fps=fps,
    )
    _save_motion_video(
        motion=rollout_positions.squeeze(0),
        title=f"Autoregressive Rollout | {sample_name}",
        save_path=OUTPUT_DIR / "rollout_full.mp4",
        fps=fps,
    )
    _save_motion_video(
        motion=masked_sequence,
        title=f"Random Teacher-Force + Rollout | {sample_name}",
        save_path=OUTPUT_DIR / "masked_teacher_force_rollout_full.mp4",
        fps=fps,
    )

    metrics = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "dataset_path": str(DATASET_PATH),
        "sample_name": sample_name,
        "sample_length": sample_len,
        "caption": caption,
        "horizon": horizon,
        "rollout_seed_length": horizon,
        "teacher_forced_predictor_steps": PREDICTOR_STEPS,
        "rollout_inference_steps": inference_steps,
        "teacher_forced_encoder_history_context": history_context_summary,
        "teacher_forced_avg": {
            key: float(np.mean([m[key] for m in teacher_forced_metrics]))
            for key in teacher_forced_metrics[0]
        },
        "teacher_forced_last": teacher_forced_metrics[-1],
        "rollout_avg": rollout_metrics,
        "rollout_last": rollout_step_metrics[-1],
        "masked_teacher_force_probability": MASKED_TEACHER_FORCE_PROB,
        "masked_teacher_force_count": int(sum(masked_teacher_force_mask)),
        "masked_teacher_force_ratio": float(
            sum(masked_teacher_force_mask) / max(1, len(masked_teacher_force_mask))
        ),
        "masked_teacher_force_rollout_avg": {
            key: float(np.mean([m[key] for m in masked_metrics]))
            for key in masked_metrics[0]
        },
        "masked_teacher_force_rollout_last": masked_metrics[-1],
        "teacher_forced_per_step": teacher_forced_metrics,
        "rollout_per_step": rollout_step_metrics,
        "masked_teacher_force_rollout_per_step": masked_metrics,
        "masked_teacher_force_rollout_mask": masked_teacher_force_mask,
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved visualizations to {OUTPUT_DIR}")
    print(json.dumps(metrics["teacher_forced_encoder_history_context"], indent=2))
    print(json.dumps(metrics["teacher_forced_avg"], indent=2))
    print(json.dumps(metrics["rollout_avg"], indent=2))


if __name__ == "__main__":
    main()
