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
    load_feature_normalizer as shared_load_feature_normalizer,
    load_generator_from_checkpoint as shared_load_generator_from_checkpoint,
    predict_next_positions as shared_predict_next_positions,
)
from utils.dataset import Text2MotionDataset, text2motion_collate_fn
from utils.motion_utils import FeatureNormalizer, generated_positions_to_271d
from utils.visualization import plot_3d_motion


CHECKPOINT_PATH = PROJECT_ROOT / "tests" / "checkpoints" / "latest5.pt"
DATASET_PATH = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "checkpoint_visualizations" / CHECKPOINT_PATH.stem
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

    sample_index = int(candidate_indices[-1])
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
        h_state = None
        current_positions = joints[:, 0]
        current_frame = motion_norm[:, 0]

        for frame_idx in range(joints.shape[1] - 1):
            context, h_state = generator.encoder.gru_step(
                current_frame,
                text_clip[:, 0, :],
                h_state,
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

    horizon = int(config.horizon)
    predictor_steps = 50
    inference_steps = max(
        1,
        int(getattr(config, "num_inference_steps", predictor_steps)),
    )

    teacher_forced_positions = [joints[:, 0]]
    teacher_forced_metrics: list[dict[str, float]] = []
    teacher_forced_flows = []
    with torch.no_grad():
        h_state = None
        for frame_idx in range(sample_len - 1):
            current_frame_norm = motion_norm[:, frame_idx]
            current_positions = joints[:, frame_idx]
            context, h_state = generator.encoder.gru_step(
                current_frame_norm,
                text_clip[:, 0, :],
                h_state,
            )
            pred_positions, pred_flow_raw = _predict_next_positions(
                generator=generator,
                context=context,
                current_positions=current_positions,
                current_frame_norm=current_frame_norm,
                text_embedding=text_clip[:, 0, :],
                num_steps=predictor_steps,
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

    seed_positions = joints[:, :horizon]
    rollout_frames = sample_len - horizon
    with torch.no_grad():
        rollout_positions, _, _ = generator.generate_sequence(
            text=text_clip,
            num_frames=rollout_frames,
            num_steps=inference_steps,
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
            num_steps=predictor_steps,
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
        "teacher_forced_predictor_steps": predictor_steps,
        "rollout_inference_steps": inference_steps,
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
    print(json.dumps(metrics["teacher_forced_avg"], indent=2))
    print(json.dumps(metrics["rollout_avg"], indent=2))


if __name__ == "__main__":
    main()
