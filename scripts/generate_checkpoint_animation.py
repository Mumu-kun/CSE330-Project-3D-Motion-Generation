from __future__ import annotations

"""Generate and save a motion animation from a checkpoint and random dataset seed."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.checkpoint_eval import (  # noqa: E402
    load_feature_normalizer,
    load_generator_from_checkpoint,
    resolve_dataset_path,
)
from utils.visualization import plot_3d_motion  # noqa: E402

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "tests" / "checkpoints" / "best_val_noss_fullset.pt"
)
DEFAULT_DATASET_PATH = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
DEFAULT_SEED_SOURCE_SPLIT = "all"
DEFAULT_PROMPT = "a person walks one way then backtracks"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _slugify(text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        slug = "generation"
    return slug[:max_length].rstrip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a motion animation from a saved checkpoint using a custom "
            "text prompt and the first frame from a random dataset animation."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=f"Path to the model checkpoint (.pt). Defaults to {DEFAULT_CHECKPOINT_PATH}.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Custom text prompt to condition generation on. Defaults to '{DEFAULT_PROMPT}'.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help=(
            "Dataset root containing Mean.npy, Std.npy, all.txt, and new_joints/. "
            f"Defaults to {DEFAULT_DATASET_PATH}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save outputs in. Defaults to "
            "'output/custom_checkpoint_generations/<checkpoint-stem>'."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional base filename for saved outputs. Defaults to a slugified prompt.",
    )
    parser.add_argument(
        "--num-generated-frames",
        type=int,
        default=None,
        help=(
            "Number of new frames to generate after the seed frame. "
            "Defaults to checkpoint config max_motion_length - 1."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of flow ODE inference steps. Defaults to checkpoint config value.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale. Defaults to 1.0.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Context horizon to use during autoregressive generation. Defaults to checkpoint config value.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output animation frame rate. Defaults to checkpoint config value.",
    )
    parser.add_argument(
        "--dataset-type",
        default="t2m",
        help="Dataset type for motion feature conversion. Defaults to 't2m'.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        choices=["cpu", "cuda"],
        help="Device to run generation on.",
    )
    parser.add_argument(
        "--use-fk",
        action="store_true",
        help="Enable FK-corrected positions during generation.",
    )
    return parser.parse_args()


def _extract_first_seed_frame(
    joints_array: np.ndarray,
    *,
    expected_joint_count: int,
) -> np.ndarray:
    if joints_array.ndim != 3:
        raise ValueError(
            "Joint file must have shape (T, J, 3); " f"got {tuple(joints_array.shape)}."
        )
    if joints_array.shape[0] < 1:
        raise ValueError("Joint file must contain at least one frame.")

    seed_frame = joints_array[0]
    if seed_frame.shape != (expected_joint_count, 3):
        raise ValueError(
            f"Expected first frame shape ({expected_joint_count}, 3), "
            f"got {tuple(seed_frame.shape)}."
        )
    return seed_frame.astype(np.float32, copy=False)


def _read_optional_seed_caption(dataset_path: Path, sample_id: str) -> str | None:
    text_path = dataset_path / "texts" / f"{sample_id}.txt"
    if not text_path.exists():
        return None

    for line in text_path.read_text(encoding="utf-8").splitlines():
        caption = line.strip().split("#")[0].strip()
        if caption:
            return caption
    return None


def _sample_random_seed_from_dataset(
    dataset_path: str | Path,
    *,
    expected_joint_count: int,
    device: torch.device,
    split_name: str = DEFAULT_SEED_SOURCE_SPLIT,
) -> tuple[torch.Tensor, dict[str, object]]:
    dataset_path = Path(dataset_path)
    split_path = dataset_path / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Seed split file not found: {split_path}")

    sample_ids = [
        line.strip() for line in split_path.read_text().splitlines() if line.strip()
    ]
    if not sample_ids:
        raise RuntimeError(f"No sample ids found in seed split file: {split_path}")

    eligible_samples: list[tuple[str, Path, np.ndarray]] = []
    for sample_id in sample_ids:
        joints_path = dataset_path / "new_joints" / f"{sample_id}.npy"
        if not joints_path.exists():
            continue
        try:
            joints_array = np.load(joints_path)
            seed_frame = _extract_first_seed_frame(
                joints_array,
                expected_joint_count=expected_joint_count,
            )
        except (OSError, ValueError):
            continue
        eligible_samples.append((sample_id, joints_path, seed_frame))

    if not eligible_samples:
        raise RuntimeError(
            "No eligible seed animations found in "
            f"{split_path}. Expected at least one valid new_joints/<sample_id>.npy "
            "with shape (T, J, 3) and at least one frame."
        )

    sample_id, joints_path, seed_frame = random.choice(eligible_samples)
    seed_positions = torch.from_numpy(seed_frame).float().unsqueeze(0).to(device)

    seed_metadata: dict[str, object] = {
        "seed_sample_id": sample_id,
        "seed_joints_path": str(joints_path.resolve()),
        "seed_frame_index": 0,
        "seed_source_split": split_name,
    }
    seed_caption = _read_optional_seed_caption(dataset_path, sample_id)
    if seed_caption is not None:
        seed_metadata["seed_caption"] = seed_caption

    return seed_positions, seed_metadata


def _build_metadata(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    prompt: str,
    device: torch.device,
    dataset_type: str,
    num_generated_frames: int,
    generated_motion: np.ndarray,
    num_steps: int,
    guidance_scale: float,
    horizon: int,
    fps: float,
    use_fk: bool,
    positions_path: Path,
    video_path: Path,
    seed_metadata: dict[str, object],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "prompt": prompt,
        "device": str(device),
        "dataset_type": dataset_type,
        "seed_frames_used": 1,
        "num_generated_frames": int(num_generated_frames),
        "total_output_frames": int(generated_motion.shape[0]),
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "horizon": int(horizon),
        "fps": float(fps),
        "use_fk": bool(use_fk),
        "positions_path": str(positions_path.resolve()),
        "video_path": str(video_path.resolve()),
    }
    metadata.update(seed_metadata)
    return metadata


def main() -> None:
    args = _parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    dataset_path = resolve_dataset_path(
        checkpoint_path=checkpoint_path,
        dataset_path=args.dataset_path,
        map_location=device,
    )
    normalizer = load_feature_normalizer(dataset_path=dataset_path, device=device)
    generator = load_generator_from_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        device=device,
        normalizer=normalizer,
    )
    config = generator.config

    num_generated_frames = (
        int(args.num_generated_frames)
        if args.num_generated_frames is not None
        else max(1, int(config.max_motion_length) - 1)
    )
    num_steps = (
        int(args.num_steps)
        if args.num_steps is not None
        else max(1, int(config.num_inference_steps))
    )
    horizon = (
        int(args.horizon) if args.horizon is not None else max(1, int(config.horizon))
    )
    fps = float(args.fps) if args.fps is not None else float(getattr(config, "fps", 20))

    seed_positions, seed_metadata = _sample_random_seed_from_dataset(
        dataset_path,
        expected_joint_count=int(config.num_joints),
        device=device,
    )

    if num_generated_frames < 1:
        raise ValueError("--num-generated-frames must be >= 1.")
    if num_steps < 1:
        raise ValueError("--num-steps must be >= 1.")
    if args.guidance_scale < 0.0:
        raise ValueError("--guidance-scale must be >= 0.")
    if horizon < 1:
        raise ValueError("--horizon must be >= 1.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT
        / "output"
        / "custom_checkpoint_generations"
        / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name or _slugify(args.prompt)
    positions_path = output_dir / f"{output_name}.npy"
    video_path = output_dir / f"{output_name}.mp4"
    metadata_path = output_dir / f"{output_name}.json"

    with torch.no_grad():
        position_history, _, _ = generator.generate_sequence(
            text=args.prompt,
            num_frames=num_generated_frames,
            num_steps=num_steps,
            horizon=horizon,
            input_positions=seed_positions,
            guidance_scale=float(args.guidance_scale),
            dataset_type=args.dataset_type,
            use_fk=args.use_fk,
        )

    generated_motion = position_history.squeeze(0).detach().cpu().numpy()
    np.save(positions_path, generated_motion)

    plot_3d_motion(
        motion=generated_motion,
        fps=fps,
        title=args.prompt,
        save_path=video_path,
    )

    metadata = _build_metadata(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        prompt=args.prompt,
        device=device,
        dataset_type=args.dataset_type,
        num_generated_frames=num_generated_frames,
        generated_motion=generated_motion,
        num_steps=num_steps,
        guidance_scale=float(args.guidance_scale),
        horizon=horizon,
        fps=fps,
        use_fk=bool(args.use_fk),
        positions_path=positions_path,
        video_path=video_path,
        seed_metadata=seed_metadata,
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        "Selected random dataset seed "
        f"{seed_metadata['seed_sample_id']} from split '{seed_metadata['seed_source_split']}'."
    )
    print(f"Saved generated positions to {positions_path}")
    print(f"Saved animation to {video_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()

"""
Minimal bash command using defaults for omitted flags:
python scripts/generate_checkpoint_animation.py \
  --checkpoint "best_val_noss_fullset.pt" \
  --prompt "a person is walking forward straight in a straight path" \
  --dataset-path "humanml3d-subset-mini" \
  --num-generated-frames 100 \
  --num-steps 20 \
  --guidance-scale 1 \
  --horizon 40

Full example bash command:
python scripts/generate_checkpoint_animation.py \
  --checkpoint "tests/checkpoints/best_val_noss_fullset.pt" \
  --prompt "a person is walking forward straight in a straight path" \
  --dataset-path "tests/dataset/humanml3d-subset-mini" \
  --output-dir "output/custom_checkpoint_generations/best_val_noss_fullset" \
  --output-name "walking_forward_straight" \
  --num-generated-frames 100 \
  --num-steps 20 \
  --guidance-scale 1.5 \
  --horizon 40 \
  --fps 20 \
  --dataset-type "t2m" \
  --device "cpu" \
  --use-fk
"""
