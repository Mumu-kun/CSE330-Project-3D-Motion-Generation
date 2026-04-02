from __future__ import annotations

import copy
import os
import pathlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import Config
from models import FlowMatchingPredictor, HumanMotionGenerator, integrate_flow_ode
from utils.motion_utils import (
    FeatureNormalizer,
    extract_prev_frame_features,
    flow_output_to_positions,
)
from utils.pose_validation import get_joint_names


_FALLBACK_CLIP_ENCODER = None


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    mode: str
    seed_length: int
    num_steps: int
    teacher_force_prob: float | None = None


@contextmanager
def windows_checkpoint_path_compat():
    """Allow checkpoints pickled with PosixPath to load on Windows."""
    original_posix_path = pathlib.PosixPath
    should_patch_posix = os.name == "nt"
    if should_patch_posix:
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        if should_patch_posix:
            pathlib.PosixPath = original_posix_path


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    with windows_checkpoint_path_compat():
        return torch.load(
            Path(checkpoint_path),
            map_location=map_location,
            weights_only=False,
        )


def load_checkpoint_config(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Config:
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing required key 'config'.")
    config = checkpoint["config"]
    if not isinstance(config, Config):
        raise TypeError(
            f"Expected checkpoint config to be Config, got {type(config).__name__}."
        )
    return copy.deepcopy(config)


def load_feature_normalizer(
    dataset_path: str | Path,
    device: str | torch.device = "cpu",
) -> FeatureNormalizer:
    dataset_path = Path(dataset_path)
    return FeatureNormalizer.load_from_files(
        dataset_path / "Mean.npy",
        dataset_path / "Std.npy",
        device=device,
    )


def resolve_dataset_path(
    checkpoint_path: str | Path,
    dataset_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> Path:
    if dataset_path is not None:
        resolved = Path(dataset_path)
    else:
        config = load_checkpoint_config(checkpoint_path, map_location=map_location)
        resolved = Path(config.dataset_path)

    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return resolved


def load_generator_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_path: str | Path,
    device: str | torch.device = "cpu",
    normalizer: FeatureNormalizer | None = None,
) -> HumanMotionGenerator:
    device_str = str(device)
    with windows_checkpoint_path_compat():
        generator = HumanMotionGenerator.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=Config(device=device_str),
            device=device_str,
            normalizer=normalizer,
        )

    generator.config.device = torch.device(device)
    generator.config.pin_memory = False
    generator.config.num_workers = 0
    generator.config.dataset_path = Path(dataset_path)
    return generator.eval()


def config_to_jsonable(config: Config) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, slice):
            return {
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }
        if is_dataclass(value):
            return {k: _convert(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    return {key: _convert(value) for key, value in vars(config).items()}


def read_split_ids(dataset_path: str | Path, split: str) -> list[str]:
    split_path = Path(dataset_path) / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def sample_length(dataset_path: str | Path, sample_id: str) -> int:
    sample_path = Path(dataset_path) / "new_joint_vecs" / f"{sample_id}.npy"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing motion file for sample '{sample_id}': {sample_path}")
    return int(np.load(sample_path, mmap_mode="r").shape[0])


def select_sample_ids_by_length_quantiles(
    dataset_path: str | Path,
    split: str,
    num_samples: int = 3,
) -> list[str]:
    ids = read_split_ids(dataset_path, split)
    if not ids:
        raise RuntimeError(f"No sample ids found for split '{split}'.")

    records = sorted((sample_length(dataset_path, sample_id), sample_id) for sample_id in ids)
    if num_samples <= 1:
        quantiles = [0.0]
    else:
        quantiles = np.linspace(0.0, 1.0, num_samples).tolist()

    selected: list[str] = []
    for quantile in quantiles:
        index = int(round((len(records) - 1) * quantile))
        sample_id = records[index][1]
        if sample_id not in selected:
            selected.append(sample_id)
    return selected


def load_text_embedding_cache(dataset_path: str | Path) -> dict[str, torch.Tensor]:
    cache_path = Path(dataset_path) / "text_embeddings_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Text embedding cache not found: {cache_path}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise TypeError(
            f"Expected text embedding cache dict, got {type(cache).__name__}."
        )
    return cache


def read_sample_caption(
    dataset_path: str | Path,
    sample_id: str,
    text_cache: dict[str, torch.Tensor] | None = None,
) -> str:
    text_path = Path(dataset_path) / "texts" / f"{sample_id}.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found for sample '{sample_id}': {text_path}")

    fallback_caption: str | None = None
    for line in text_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            caption = line.split("#")[0]
            if fallback_caption is None:
                fallback_caption = caption
            if text_cache is None or caption in text_cache:
                return caption
    if fallback_caption is not None:
        return fallback_caption
    raise RuntimeError(f"No caption found in text file: {text_path}")


def load_sample(
    dataset_path: str | Path,
    sample_id: str,
    *,
    device: str | torch.device = "cpu",
    text_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    motion_path = dataset_path / "new_joint_vecs" / f"{sample_id}.npy"
    joints_path = dataset_path / "new_joints" / f"{sample_id}.npy"

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    if not joints_path.exists():
        raise FileNotFoundError(f"Joint file not found: {joints_path}")

    if text_cache is None:
        text_cache = load_text_embedding_cache(dataset_path)
    caption = read_sample_caption(dataset_path, sample_id, text_cache=text_cache)
    if caption not in text_cache:
        global _FALLBACK_CLIP_ENCODER
        from utils.text_encoder import CLIPEncoder

        if _FALLBACK_CLIP_ENCODER is None:
            _FALLBACK_CLIP_ENCODER = CLIPEncoder(
                model_name="openai/clip-vit-base-patch32"
            )
        clip_encoder = _FALLBACK_CLIP_ENCODER.to(device)
        with torch.no_grad():
            encoded = clip_encoder([caption]).detach().cpu().float()
        text_cache[caption] = encoded.squeeze(0).squeeze(0)

    text_embedding = text_cache[caption].detach().cpu().float()
    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)
    elif text_embedding.ndim != 2 or text_embedding.shape[0] != 1:
        raise ValueError(
            f"Expected text embedding shape (1, 512), got {tuple(text_embedding.shape)}."
        )

    motion = torch.from_numpy(np.load(motion_path)).float().unsqueeze(0)
    joints = torch.from_numpy(np.load(joints_path)).float().unsqueeze(0)
    return {
        "sample_id": sample_id,
        "caption": caption,
        "sample_length": int(motion.shape[1]),
        "motion": motion.to(device),
        "joints": joints.to(device),
        "text_clip": text_embedding.unsqueeze(0).to(device),
    }


def build_default_scenarios(config: Config) -> list[ScenarioSpec]:
    default_horizon = max(1, int(config.horizon))
    default_steps = max(1, int(config.num_inference_steps))
    short_horizon = max(1, default_horizon // 2)
    high_steps = max(default_steps * 2, default_steps + 10)
    return [
        ScenarioSpec(
            name="teacher_forced_one_step",
            mode="teacher_forced_one_step",
            seed_length=1,
            num_steps=default_steps,
        ),
        ScenarioSpec(
            name="mixed_teacher_force_rollout",
            mode="mixed_teacher_force_rollout",
            seed_length=1,
            num_steps=default_steps,
            teacher_force_prob=0.4,
        ),
        ScenarioSpec(
            name="rollout_default",
            mode="rollout",
            seed_length=default_horizon,
            num_steps=default_steps,
        ),
        ScenarioSpec(
            name="rollout_short_context",
            mode="rollout",
            seed_length=short_horizon,
            num_steps=default_steps,
        ),
        ScenarioSpec(
            name="rollout_high_steps",
            mode="rollout",
            seed_length=default_horizon,
            num_steps=high_steps,
        ),
    ]


def predict_next_positions(
    *,
    generator: HumanMotionGenerator,
    context: torch.Tensor,
    current_positions: torch.Tensor,
    current_frame_norm: torch.Tensor,
    text_embedding: torch.Tensor,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictor: FlowMatchingPredictor = generator.predictor
    current_frame_features = extract_prev_frame_features(
        current_frame_norm,
        normalizer=generator.normalizer,
    )

    x_t = integrate_flow_ode(
        predictor=predictor,
        track_features=context,
        current_frame_features=current_frame_features,
        text_embedding=text_embedding,
        num_steps=num_steps,
        time_schedule_power=generator.config.inference_t_schedule_power,
    )

    pred_flow_raw = (
        generator.normalizer.denormalize_flow_output(x_t)
        if generator.normalizer is not None
        else x_t
    )
    current_frame_raw = (
        generator.normalizer.denormalize(current_frame_norm)
        if generator.normalizer is not None
        else current_frame_norm
    )
    pred_positions = flow_output_to_positions(
        pred_flow_raw,
        prev_root_pos=current_positions[:, 0],
        prev_root_rot_6d=current_frame_raw[:, 69:75],
    )
    return pred_positions, pred_flow_raw


def per_joint_l2(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vector_norm(pred_positions - target_positions, ord=2, dim=-1)


def aggregate_position_metrics(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> dict[str, float]:
    diff = pred_positions - target_positions
    per_joint = per_joint_l2(pred_positions, target_positions)
    return {
        "joint_mse": float(diff.pow(2).mean().item()),
        "joint_mae": float(diff.abs().mean().item()),
        "joint_l2_mean": float(per_joint.mean().item()),
        "root_l2_mean": float(per_joint[..., 0].mean().item()),
        "nonroot_joint_l2_mean": float(per_joint[..., 1:].mean().item()),
    }


def compute_drift_metrics(frame_joint_l2_mean: torch.Tensor) -> dict[str, float]:
    if frame_joint_l2_mean.ndim != 1:
        raise ValueError(
            f"Expected 1D frame error tensor, got {tuple(frame_joint_l2_mean.shape)}."
        )
    if frame_joint_l2_mean.numel() == 0:
        raise ValueError("Drift metrics require at least one evaluated frame.")

    chunk = max(1, frame_joint_l2_mean.numel() // 3)
    first = frame_joint_l2_mean[:chunk]
    last = frame_joint_l2_mean[-chunk:]
    return {
        "first_third_joint_l2_mean": float(first.mean().item()),
        "last_third_joint_l2_mean": float(last.mean().item()),
        "drift_delta": float((last.mean() - first.mean()).item()),
        "final_joint_l2_mean": float(frame_joint_l2_mean[-1].item()),
        "worst_frame_joint_l2_mean": float(frame_joint_l2_mean.max().item()),
    }


def mean_per_joint_l2(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    return per_joint_l2(pred_positions, target_positions).mean(dim=0)


def joint_name(index: int) -> str:
    return get_joint_names().get(index, f"joint_{index}")
