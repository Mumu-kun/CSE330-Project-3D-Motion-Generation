from __future__ import annotations

"""Visualize one-step and rollout predictions from a saved checkpoint."""

import copy
import json
import os
import pathlib
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import FlowMatchingPredictorConfig
from models import FlowMatchingPredictor, HumanMotionGenerator, MotionHistoryEncoder
from utils.dataset import Text2MotionDataset, text2motion_collate_fn
from utils.motion_utils import FeatureNormalizer, generated_positions_to_271d
from utils.visualization import plot_3d_motion


CHECKPOINT_PATH = PROJECT_ROOT / "tests" / "checkpoints" / "latest4.pt"
DATASET_PATH = PROJECT_ROOT / "tests" / "dataset" / "humanml3d-subset-mini"
OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "checkpoint_visualizations" / CHECKPOINT_PATH.stem
)
MASKED_TEACHER_FORCE_PROB = 0.4


class _LegacyMotionHistoryEncoder(MotionHistoryEncoder):
    """Compatibility encoder for checkpoints without joint projection heads."""

    def _gru_block(
        self,
        motion_in: torch.Tensor,
        text_emb: torch.Tensor,
        h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = motion_in.shape

        if h is None:
            h = self.init_hidden(text_emb)

        text_proj = self.text_proj(text_emb)
        text_rep = self.text_scale * text_proj
        text_rep = text_rep.unsqueeze(1).expand(batch_size, time_steps, -1)
        gru_in = torch.cat([motion_in, text_rep], dim=-1)

        h_seq, h_next = self.gru(gru_in, h)
        h_t = h_seq[:, -1, :]
        history_features = h_t.unsqueeze(1).expand(
            batch_size, self.joint_count, self.model_dim
        )
        return history_features, h_next


def _load_feature_normalizer(
    dataset_path: str | Path,
    device: str | torch.device = "cpu",
) -> FeatureNormalizer:
    dataset_path = Path(dataset_path)
    return FeatureNormalizer.load_from_files(
        dataset_path / "Mean.npy",
        dataset_path / "Std.npy",
        device=device,
    )


def _load_checkpoint_payload(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    original_posix_path = pathlib.PosixPath
    should_patch_posix = os.name == "nt"
    if should_patch_posix:
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    finally:
        if should_patch_posix:
            pathlib.PosixPath = original_posix_path


def _select_model_state(
    checkpoint: dict[str, Any],
    key: str,
    prefer_ema: bool = True,
) -> dict[str, torch.Tensor]:
    ema_key = f"{key}_ema"
    if prefer_ema and ema_key in checkpoint:
        return checkpoint[ema_key]
    if key not in checkpoint:
        raise KeyError(f"Checkpoint is missing required key '{key}'.")
    return checkpoint[key]


def _checkpoint_config(checkpoint: dict[str, Any]) -> Any | None:
    config = checkpoint.get("config")
    return copy.deepcopy(config) if config is not None else None


def _extract_component_metadata(
    checkpoint: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    for container_key in ("model_metadata", "model_architecture"):
        container = checkpoint.get(container_key)
        if isinstance(container, dict):
            component = container.get(key)
            if isinstance(component, dict):
                return component

    direct_keys = {
        "encoder": ("encoder_metadata", "encoder_arch"),
        "predictor": ("predictor_metadata", "predictor_kwargs"),
    }
    for metadata_key in direct_keys[key]:
        component = checkpoint.get(metadata_key)
        if isinstance(component, dict):
            return component
    return None


def _metadata_kwargs(component_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not component_metadata:
        return {}

    for key in ("kwargs", "init_kwargs"):
        payload = component_metadata.get(key)
        if isinstance(payload, dict):
            return dict(payload)

    return {
        key: value
        for key, value in component_metadata.items()
        if key not in {"name", "class_name", "type"}
    }


def _encoder_has_joint_projection(
    encoder_state: dict[str, torch.Tensor],
) -> bool:
    return any(key.startswith("global_to_joints") for key in encoder_state.keys())


def _infer_encoder_kwargs(
    checkpoint: dict[str, Any],
    prefer_ema: bool = True,
) -> dict[str, Any]:
    config = _checkpoint_config(checkpoint)
    encoder_state = _select_model_state(checkpoint, "encoder", prefer_ema=prefer_ema)
    model_dim = int(encoder_state["text_to_hidden.weight"].shape[0])
    text_embedding_dim = int(encoder_state["text_to_hidden.weight"].shape[1])
    text_proj_dim = int(encoder_state["text_proj.weight"].shape[0])
    num_layers = len(
        [key for key in encoder_state.keys() if key.startswith("gru.weight_ih_l")]
    )
    frame_feature_dim = int(encoder_state["gru.weight_ih_l0"].shape[1] - text_proj_dim)
    has_joint_projection = _encoder_has_joint_projection(encoder_state)
    joint_count = getattr(config, "encoder_num_joints", 22)
    if has_joint_projection:
        per_joint_out_dim = int(
            encoder_state["global_to_joints.2.weight"].shape[0] // joint_count
        )
    else:
        per_joint_out_dim = model_dim

    return {
        "frame_feature_dim": frame_feature_dim,
        "text_embedding_dim": text_embedding_dim,
        "text_proj_dim": text_proj_dim,
        "model_dim": model_dim,
        "per_joint_out_dim": per_joint_out_dim,
        "num_layers": num_layers,
        "joint_count": joint_count,
        "text_scale": getattr(config, "encoder_text_scale", 1.0),
        "dropout": getattr(config, "encoder_dropout", 0.0),
    }


def _coerce_predictor_config(
    predictor_config: Any,
    fallback_config: FlowMatchingPredictorConfig,
) -> FlowMatchingPredictorConfig:
    if isinstance(predictor_config, FlowMatchingPredictorConfig):
        return copy.deepcopy(predictor_config)

    if isinstance(predictor_config, dict):
        valid_fields = {field.name for field in fields(FlowMatchingPredictorConfig)}
        merged = {
            field.name: copy.deepcopy(getattr(fallback_config, field.name))
            for field in fields(FlowMatchingPredictorConfig)
        }
        merged.update(
            {
                key: copy.deepcopy(value)
                for key, value in predictor_config.items()
                if key in valid_fields
            }
        )
        return FlowMatchingPredictorConfig(**merged)

    return copy.deepcopy(fallback_config)


def _infer_predictor_kwargs(
    checkpoint: dict[str, Any],
    prefer_ema: bool = True,
) -> dict[str, Any]:
    config = _checkpoint_config(checkpoint)
    if config is None:
        raise KeyError(
            "Checkpoint is missing required key 'config' needed for predictor loading."
        )

    predictor_state = _select_model_state(
        checkpoint,
        "predictor",
        prefer_ema=prefer_ema,
    )
    track_dim = int(predictor_state["output_projection.weight"].shape[0])
    input_dim = int(predictor_state["input_projection.weight"].shape[1])
    expected_feature_size = int(config.get_predictor_feature_size())

    candidates: list[tuple[bool, int]] = []
    if input_dim >= 2 * track_dim:
        candidates.append((True, input_dim - (2 * track_dim)))
    if input_dim >= track_dim:
        candidates.append((False, input_dim - track_dim))

    use_relative_shift = True
    feature_size = expected_feature_size
    for candidate_use_relative_shift, candidate_feature_size in candidates:
        if candidate_feature_size == expected_feature_size:
            use_relative_shift = candidate_use_relative_shift
            feature_size = candidate_feature_size
            break
    else:
        valid_candidates = [candidate for candidate in candidates if candidate[1] >= 0]
        if not valid_candidates:
            raise ValueError(
                f"Could not infer predictor feature size from input_dim={input_dim} "
                f"and track_dim={track_dim}."
            )
        use_relative_shift, feature_size = valid_candidates[0]

    return {
        "feature_size": feature_size,
        "config": copy.deepcopy(config.predictor_config),
        "out_channels": track_dim,
        "use_relative_shift": use_relative_shift,
    }


def _resolve_encoder_spec(
    checkpoint: dict[str, Any],
    prefer_ema: bool = True,
) -> tuple[type[MotionHistoryEncoder], dict[str, Any], bool]:
    encoder_state = _select_model_state(checkpoint, "encoder", prefer_ema=prefer_ema)
    inferred_kwargs = _infer_encoder_kwargs(checkpoint, prefer_ema=prefer_ema)
    metadata = _extract_component_metadata(checkpoint, "encoder")
    metadata_kwargs = _metadata_kwargs(metadata)
    encoder_kwargs = {**inferred_kwargs, **metadata_kwargs}

    arch_name = ""
    if metadata is not None:
        for key in ("name", "class_name", "type"):
            value = metadata.get(key)
            if isinstance(value, str):
                arch_name = value.lower()
                break

    has_joint_projection = _encoder_has_joint_projection(encoder_state)
    if arch_name and "legacy" in arch_name:
        return _LegacyMotionHistoryEncoder, encoder_kwargs, False
    if arch_name and "motionhistoryencoder" in arch_name.replace("_", ""):
        return MotionHistoryEncoder, encoder_kwargs, True
    if arch_name and "joint_projection" in arch_name:
        return MotionHistoryEncoder, encoder_kwargs, True
    if has_joint_projection:
        return MotionHistoryEncoder, encoder_kwargs, True
    return _LegacyMotionHistoryEncoder, encoder_kwargs, False


def _resolve_predictor_kwargs(
    checkpoint: dict[str, Any],
    prefer_ema: bool = True,
) -> dict[str, Any]:
    config = _checkpoint_config(checkpoint)
    if config is None:
        raise KeyError(
            "Checkpoint is missing required key 'config' needed for predictor loading."
        )

    inferred_kwargs = _infer_predictor_kwargs(checkpoint, prefer_ema=prefer_ema)
    metadata = _extract_component_metadata(checkpoint, "predictor")
    metadata_kwargs = _metadata_kwargs(metadata)
    predictor_kwargs = {**inferred_kwargs, **metadata_kwargs}
    predictor_kwargs["config"] = _coerce_predictor_config(
        predictor_kwargs.get("config"),
        fallback_config=config.predictor_config,
    )
    return predictor_kwargs


def _load_generator_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    normalizer: FeatureNormalizer | None = None,
    prefer_ema: bool = True,
) -> HumanMotionGenerator:
    checkpoint = _load_checkpoint_payload(checkpoint_path, map_location=device)
    if checkpoint.get("format") == "human_motion_inference_bundle":
        raise ValueError(
            "Inference bundle checkpoints are not supported by this script. "
            "Load a raw training checkpoint instead."
        )
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing required key 'config'.")

    config = copy.deepcopy(checkpoint["config"])
    config.device = torch.device(device)
    config.pin_memory = False
    config.num_workers = 0
    if dataset_path is not None:
        config.dataset_path = Path(dataset_path)

    if normalizer is None and getattr(config, "dataset_path", None) is not None:
        dataset_root = Path(config.dataset_path)
        mean_path = dataset_root / "Mean.npy"
        std_path = dataset_root / "Std.npy"
        if mean_path.exists() and std_path.exists():
            normalizer = _load_feature_normalizer(dataset_root, device=config.device)

    encoder_cls, encoder_kwargs, strict_encoder = _resolve_encoder_spec(
        checkpoint,
        prefer_ema=prefer_ema,
    )
    predictor_kwargs = _resolve_predictor_kwargs(checkpoint, prefer_ema=prefer_ema)

    encoder = encoder_cls(
        **encoder_kwargs,
        normalizer=normalizer,
    ).to(config.device)
    predictor = FlowMatchingPredictor(
        **predictor_kwargs,
    ).to(config.device)

    encoder.load_state_dict(
        _select_model_state(checkpoint, "encoder", prefer_ema=prefer_ema),
        strict=strict_encoder,
    )
    predictor.load_state_dict(
        _select_model_state(checkpoint, "predictor", prefer_ema=prefer_ema)
    )
    encoder.eval()
    predictor.eval()

    return HumanMotionGenerator(
        encoder=encoder,
        predictor=predictor,
        config=config,
    ).eval()


def _load_long_sample(config, normalizer):
    dataset = Text2MotionDataset(
        config=config,
        mean=normalizer.mean.cpu().numpy(),
        std=normalizer.std.cpu().numpy(),
        split="train",
    )
    min_required_len = config.horizon + 8
    candidate_indices = np.where(dataset.length_arr >= min_required_len)[0]
    if len(candidate_indices) == 0:
        raise RuntimeError(
            f"No sample in {DATASET_PATH} is long enough for horizon={config.horizon}."
        )
    sample_index = int(candidate_indices[-1])
    sample_name = dataset.name_list[sample_index]
    sample_len = int(dataset.data_dict[sample_name]["length"])

    dataset.set_horizon(sample_len)
    batch = text2motion_collate_fn([dataset[0]])
    batch["sample_name"] = sample_name
    batch["sample_length"] = sample_len
    return batch


def _ode_predict_shift(
    predictor: FlowMatchingPredictor,
    context: torch.Tensor,
    text_embedding: torch.Tensor,
    prev_relative_shift: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    batch_size = context.shape[0]
    x_t = torch.randn(
        (batch_size, context.shape[1], predictor.out_channels),
        device=context.device,
        dtype=context.dtype,
    )
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.full(
            (batch_size,),
            float(step) * dt,
            device=context.device,
            dtype=context.dtype,
        )
        pred = predictor(
            track_features=context,
            noised_tracks=x_t,
            timesteps=t,
            prev_relative_shifts=prev_relative_shift,
            text_embedding=text_embedding,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        x_t = x_t + pred * dt
    return x_t


def _joint_metrics(
    pred_positions: torch.Tensor, target_positions: torch.Tensor
) -> dict:
    diff = pred_positions - target_positions
    return {
        "joint_mse": float(diff.pow(2).mean().item()),
        "joint_mae": float(diff.abs().mean().item()),
        "joint_l2_mean": float(torch.norm(diff, dim=-1).mean().item()),
    }


def _save_motion_video(
    motion: torch.Tensor, title: str, save_path: Path, fps: float
) -> None:
    plot_3d_motion(
        motion=motion.detach().cpu().numpy(),
        fps=fps,
        title=title,
        save_path=save_path,
    )


def _mixed_teacher_force_rollout(
    generator,
    joints: torch.Tensor,
    motion_norm: torch.Tensor,
    text_clip: torch.Tensor,
    num_steps: int,
    teacher_force_prob: float,
    normalizer,
):
    mixed_positions = [joints[:, 0]]
    mixed_metrics = []
    teacher_force_mask = []

    with torch.no_grad():
        h_state = None
        current_positions = joints[:, 0]
        current_frame = motion_norm[:, 0]
        prev_positions = None

        for frame_idx in range(joints.shape[1] - 1):
            context, h_state = generator.encoder.gru_step(
                current_frame,
                text_clip[:, 0, :],
                h_state,
            )
            prev_relative_shift = (
                current_positions - prev_positions
                if prev_positions is not None
                else torch.zeros_like(current_positions)
            )
            pred_shift = _ode_predict_shift(
                predictor=generator.predictor,
                context=context,
                text_embedding=text_clip[:, 0, :],
                prev_relative_shift=prev_relative_shift,
                num_steps=num_steps,
            )
            pred_positions = current_positions + pred_shift
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
                    normalizer=normalizer,
                )

            prev_positions = current_positions
            current_positions = next_positions
            current_frame = next_frame

    return (
        torch.cat(mixed_positions, dim=0),
        teacher_force_mask,
        mixed_metrics,
    )


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
        prefer_ema=True,
    )
    config = generator.config

    batch = _load_long_sample(config, normalizer)
    sample_name = batch["sample_name"]
    sample_len = int(batch["sample_length"])
    caption = batch["captions"][0]

    motion_raw = batch["motion"].to(config.device)
    joints = batch["joints"].to(config.device)
    text_clip = batch["text_clip"].to(config.device)
    motion_norm = normalizer.normalize(motion_raw)

    batch_size = 1
    horizon = int(config.horizon)
    one_step_steps = max(1, sample_len - 1)
    predictor_steps = 50
    inference_steps = max(
        1, int(getattr(config, "num_inference_steps", predictor_steps))
    )

    # Teacher-forced one-step predictions across the sample.
    teacher_forced_positions = [joints[:, 0]]
    teacher_forced_metrics = []
    with torch.no_grad():
        h_state = None
        for frame_idx in range(sample_len - 1):
            context, h_state = generator.encoder.gru_step(
                motion_norm[:, frame_idx],
                text_clip[:, 0, :],
                h_state,
            )
            prev_relative_shift = (
                joints[:, frame_idx] - joints[:, frame_idx - 1]
                if frame_idx > 0
                else torch.zeros_like(joints[:, frame_idx])
            )
            pred_shift = _ode_predict_shift(
                predictor=generator.predictor,
                context=context,
                text_embedding=text_clip[:, 0, :],
                prev_relative_shift=prev_relative_shift,
                num_steps=predictor_steps,
            )
            pred_positions = joints[:, frame_idx] + pred_shift
            teacher_forced_positions.append(pred_positions)
            teacher_forced_metrics.append(
                _joint_metrics(pred_positions, joints[:, frame_idx + 1])
            )

    teacher_forced_sequence = torch.cat(teacher_forced_positions, dim=0)

    # Full rollout using the first horizon frames as seed.
    if sample_len <= horizon:
        raise RuntimeError(
            f"Sample '{sample_name}' length {sample_len} is not longer than horizon {horizon}."
        )
    seed_positions = joints[:, :1]
    rollout_frames = sample_len - 1
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
            normalizer=normalizer,
        )
    )

    # Save raw arrays for later inspection.
    np.save(OUTPUT_DIR / "ground_truth_full.npy", joints.squeeze(0).cpu().numpy())
    np.save(
        OUTPUT_DIR / "teacher_forced_one_step_full.npy",
        teacher_forced_sequence.cpu().numpy(),
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

    # Save videos.
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

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved visualizations to {OUTPUT_DIR}")
    print(json.dumps(metrics["teacher_forced_avg"], indent=2))
    print(json.dumps(metrics["rollout_avg"], indent=2))


if __name__ == "__main__":
    main()
