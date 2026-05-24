from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluator"))

from config import Config
from evaluator.eval_t2m import (  # type: ignore[import-not-found]
    build_human_motion_eval_resources,
    evaluate_human_motion_generation,
)
from models import HumanMotionGenerator, _windows_checkpoint_path_compat
from utils.dataset import create_dataloader

DEFAULT_DATASET_PATH = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluation"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path_value: str | Path) -> Path:
    return Path(path_value).expanduser().resolve()


def _training_short_string(now: datetime | None = None) -> str:
    current = datetime.now() if now is None else now
    return f"{current.strftime('%H%M')}{current.day}{current.strftime('%m')}"


def _checkpoint_identifier(checkpoint_path: Path) -> str:
    with _windows_checkpoint_path_compat():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_id = checkpoint.get("checkpoint_id")
    if checkpoint_id:
        return str(checkpoint_id)

    wandb_run_name = checkpoint.get("wandb_run_name")
    if wandb_run_name:
        return str(wandb_run_name)

    return f"{_training_short_string()}unk"


def _jsonable_summary(
    summary,
    *,
    checkpoint_id: str,
    checkpoint_path: Path,
    dataset_path: Path,
    split: str,
) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "split": split,
        "fid": float(summary.fid),
        "diversity": float(summary.diversity),
        "r_precision": [float(value) for value in summary.r_precision.tolist()],
        "matching_score": float(summary.matching_score),
        "multimodality": float(summary.multimodality),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a native Human Motion checkpoint on a HumanML3D split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the HumanML3D-style dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for metric sampling.",
    )
    parser.add_argument(
        "--seed-frames",
        type=int,
        default=10,
        help="Number of seed frames used for rollout generation.",
    )
    parser.add_argument(
        "--multimodality-batches",
        type=int,
        default=3,
        help="Number of batches used when computing multimodality.",
    )
    parser.add_argument(
        "--multimodality-repeats",
        type=int,
        default=30,
        help="Number of repeated generations per multimodality batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional maximum number of evaluation batches to consume.",
    )
    parser.add_argument(
        "--evaluator-checkpoint",
        type=str,
        default=None,
        help="Optional path to the evaluator text-motion matching checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved JSON summaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_path = _resolve_path(args.checkpoint)
    dataset_path = _resolve_path(args.dataset_path)
    output_dir = _resolve_path(args.output_dir)
    device = torch.device(args.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    _seed_everything(args.seed)

    config = Config(device=device, dataset_path=dataset_path)
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.pin_memory = device.type == "cuda"

    val_loader, normalizer = create_dataloader(
        config,
        split=args.split,
        shuffle=False,
    )
    if args.max_batches is not None:
        val_loader = itertools.islice(val_loader, max(0, args.max_batches))

    generator = HumanMotionGenerator.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=str(device),
        normalizer=normalizer,
    )

    resources = build_human_motion_eval_resources(
        dataset_path,
        device=device,
        checkpoint_path=args.evaluator_checkpoint,
    )

    summary = evaluate_human_motion_generation(
        val_loader,
        generator,
        resources,
        epoch=0,
        seed_frames=args.seed_frames,
        multimodality_batches=args.multimodality_batches,
        multimodality_repeats=args.multimodality_repeats,
    )

    print()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Split: {args.split}")
    print(f"FID: {summary.fid:.4f}")
    print(f"Diversity: {summary.diversity:.4f}")
    print(
        "R-precision: "
        f"top1={summary.r_precision[0]:.4f}, "
        f"top2={summary.r_precision[1]:.4f}, "
        f"top3={summary.r_precision[2]:.4f}"
    )
    print(f"Matching score: {summary.matching_score:.4f}")
    print(f"Multimodality: {summary.multimodality:.4f}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = _checkpoint_identifier(checkpoint_path)
        summary_path = output_dir / checkpoint_id / f"{args.split}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                _jsonable_summary(
                    summary,
                    checkpoint_id=checkpoint_id,
                    checkpoint_path=checkpoint_path,
                    dataset_path=dataset_path,
                    split=args.split,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Summary written to: {summary_path}")
        print(f"Checkpoint ID: {checkpoint_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
