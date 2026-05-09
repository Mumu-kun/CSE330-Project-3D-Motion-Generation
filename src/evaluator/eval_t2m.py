from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metrics import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
    calculate_multimodality,
)
from t2m_eval_wrapper import EvaluatorWrapper
from utils.motion_utils import sequence_271d_to_263d
from word_vectorizer import WordVectorizer


DEFAULT_EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_GLOVE_ROOT = DEFAULT_EVAL_ROOT / "glove"
DEFAULT_GLOVE_PREFIX = "our_vab"
DEFAULT_META_ROOT = DEFAULT_EVAL_ROOT / "Comp_v6_KLD005" / "meta"
DEFAULT_MAX_TEXT_LEN = 20
DEFAULT_MULTIMODALITY_BATCHES = 3
DEFAULT_MULTIMODALITY_REPEATS = 30


@dataclass(frozen=True)
class HumanMotionEvalResources:
    wrapper: EvaluatorWrapper
    word_vectorizer: WordVectorizer
    motion_mean: torch.Tensor
    motion_std: torch.Tensor
    dataset_path: Path
    device: torch.device
    max_text_len: int = DEFAULT_MAX_TEXT_LEN


@dataclass(frozen=True)
class HumanMotionEvalSummary:
    fid: float
    diversity: float
    r_precision: np.ndarray
    matching_score: float
    multimodality: float

    def as_tuple(self) -> tuple[float, float, np.ndarray, float, float]:
        return (
            self.fid,
            self.diversity,
            self.r_precision,
            self.matching_score,
            self.multimodality,
        )


def _as_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _load_eval_stats() -> tuple[torch.Tensor, torch.Tensor]:
    mean_path = DEFAULT_META_ROOT / "mean.npy"
    std_path = DEFAULT_META_ROOT / "std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Evaluator motion stats not found in {DEFAULT_META_ROOT}."
        )

    motion_mean = torch.from_numpy(np.load(mean_path)).float()
    motion_std = torch.from_numpy(np.load(std_path)).float()
    return motion_mean, motion_std


def build_human_motion_eval_resources(
    dataset_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
) -> HumanMotionEvalResources:
    device_obj = _as_device(device)
    motion_mean, motion_std = _load_eval_stats()
    wrapper = EvaluatorWrapper(
        dataset_name="humanml",
        device=device_obj,
        checkpoint_path=checkpoint_path,
    )
    word_vectorizer = WordVectorizer(str(DEFAULT_GLOVE_ROOT), DEFAULT_GLOVE_PREFIX)
    return HumanMotionEvalResources(
        wrapper=wrapper,
        word_vectorizer=word_vectorizer,
        motion_mean=motion_mean,
        motion_std=motion_std,
        dataset_path=Path(dataset_path),
        device=device_obj,
    )


def _read_caption_tokens(text_path: Path, caption: str) -> list[str]:
    fallback_tokens: list[str] | None = None
    if not text_path.exists():
        return _fallback_tokenize_caption(caption)
    for line in text_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 2:
            continue
        line_caption = parts[0].strip()
        tokens = [token for token in parts[1].split(" ") if token]
        if fallback_tokens is None:
            fallback_tokens = tokens
        if line_caption == caption.strip():
            return tokens
    if fallback_tokens is not None:
        return fallback_tokens
    return _fallback_tokenize_caption(caption)


def _fallback_tokenize_caption(caption: str) -> list[str]:
    tokens = [token for token in caption.replace(",", " ").replace(".", " ").split() if token]
    if not tokens:
        tokens = ["motion"]
    return [f"{token}/OTHER" for token in tokens]


def _pad_tokens(tokens: list[str], max_text_len: int) -> tuple[list[str], int]:
    if len(tokens) < max_text_len:
        padded_tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(padded_tokens)
        padded_tokens += ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        return padded_tokens, sent_len

    cropped_tokens = tokens[:max_text_len]
    padded_tokens = ["sos/OTHER"] + cropped_tokens + ["eos/OTHER"]
    return padded_tokens, len(padded_tokens)


def _build_text_batch(
    *,
    resources: HumanMotionEvalResources,
    captions: list[str],
    sample_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    word_embeddings: list[np.ndarray] = []
    pos_one_hots: list[np.ndarray] = []
    cap_lens: list[int] = []

    for caption, sample_id in zip(captions, sample_ids):
        text_path = resources.dataset_path / "texts" / f"{sample_id}.txt"
        tokens = _read_caption_tokens(text_path, caption)
        tokens, sent_len = _pad_tokens(tokens, resources.max_text_len)
        cap_lens.append(sent_len)

        sample_word_embeddings: list[np.ndarray] = []
        sample_pos_one_hots: list[np.ndarray] = []
        for token in tokens:
            word_emb, pos_oh = resources.word_vectorizer[token]
            sample_word_embeddings.append(np.asarray(word_emb, dtype=np.float32)[None, :])
            sample_pos_one_hots.append(np.asarray(pos_oh, dtype=np.float32)[None, :])

        word_embeddings.append(np.concatenate(sample_word_embeddings, axis=0))
        pos_one_hots.append(np.concatenate(sample_pos_one_hots, axis=0))

    word_tensor = torch.from_numpy(np.stack(word_embeddings, axis=0)).float()
    pos_tensor = torch.from_numpy(np.stack(pos_one_hots, axis=0)).float()
    lens_tensor = torch.tensor(cap_lens, dtype=torch.long)
    return word_tensor, pos_tensor, lens_tensor


def _standardize_motion_263d(
    motion_263d: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
) -> torch.Tensor:
    motion_mean = motion_mean.to(device=motion_263d.device, dtype=motion_263d.dtype)
    motion_std = motion_std.to(device=motion_263d.device, dtype=motion_263d.dtype)
    return (motion_263d - motion_mean) / motion_std


def _prepare_motion_batch(
    motion_271d: torch.Tensor,
    *,
    generator_normalizer: Any,
    resources: HumanMotionEvalResources,
) -> torch.Tensor:
    motion_263d = sequence_271d_to_263d(motion_271d, normalizer=generator_normalizer)
    return _standardize_motion_263d(
        motion_263d,
        resources.motion_mean,
        resources.motion_std,
    )


def _build_rollout_inputs(
    joints: torch.Tensor,
    lengths: torch.Tensor,
    seed_frames: int,
) -> tuple[torch.Tensor, int]:
    max_seed_frames = int(lengths.min().item())
    seed_frames = max(1, min(seed_frames, max_seed_frames, joints.shape[1]))
    return joints[:, :seed_frames], seed_frames


def _generate_motion_sequence(
    generator: Any,
    text_clip: torch.Tensor,
    seed_positions: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    rollout_steps = int(getattr(generator.config, "num_inference_steps", 20))
    horizon = int(getattr(generator.config, "horizon", seed_positions.shape[1]))
    use_fk = bool(getattr(generator.config, "use_fk", False))

    _, pred_features, _ = generator.generate_sequence(
        text=text_clip,
        num_frames=num_frames,
        num_steps=rollout_steps,
        horizon=horizon,
        input_positions=seed_positions,
        guidance_scale=1.0,
        dataset_type="t2m",
        use_fk=use_fk,
    )
    return pred_features


@torch.no_grad()
def evaluate_human_motion_generation(
    val_loader,
    generator,
    resources: HumanMotionEvalResources,
    *,
    writer=None,
    epoch: int = 0,
    seed_frames: int = 10,
    multimodality_batches: int = DEFAULT_MULTIMODALITY_BATCHES,
    multimodality_repeats: int = DEFAULT_MULTIMODALITY_REPEATS,
) -> HumanMotionEvalSummary:
    generator.eval()

    motion_annotation_list: list[np.ndarray] = []
    motion_pred_list: list[np.ndarray] = []
    multimodality_batches_list: list[np.ndarray] = []
    r_precision_total = np.zeros(3, dtype=np.float64)
    matching_score_total = 0.0
    sample_count = 0

    for batch_index, batch in enumerate(val_loader):
        captions = batch["captions"]
        sample_ids = batch.get("sample_ids")
        if sample_ids is None:
            raise KeyError(
                "The native dataloader must include 'sample_ids' for evaluation."
            )

        motion_271d = batch["motion"].to(resources.device)
        joints = batch["joints"].to(resources.device)
        lengths = batch["lengths"].to(resources.device)
        text_clip = batch["text_clip"].to(resources.device)

        text_embs, pos_ohot, cap_lens = _build_text_batch(
            resources=resources,
            captions=captions,
            sample_ids=sample_ids,
        )
        text_embs = text_embs.to(resources.device)
        pos_ohot = pos_ohot.to(resources.device)
        cap_lens = cap_lens.to(resources.device)

        seed_positions, effective_seed_frames = _build_rollout_inputs(
            joints=joints,
            lengths=lengths,
            seed_frames=seed_frames,
        )
        num_frames = max(0, motion_271d.shape[1] - effective_seed_frames)

        pred_features = _generate_motion_sequence(
            generator,
            text_clip=text_clip,
            seed_positions=seed_positions,
            num_frames=num_frames,
        )

        gt_motion_263d = _prepare_motion_batch(
            motion_271d,
            generator_normalizer=generator.normalizer,
            resources=resources,
        )
        pred_motion_263d = _prepare_motion_batch(
            pred_features,
            generator_normalizer=generator.normalizer,
            resources=resources,
        )

        gt_text_emb, gt_motion_emb = resources.wrapper.get_co_embeddings(
            text_embs,
            pos_ohot,
            cap_lens,
            gt_motion_263d,
            lengths,
        )
        _, pred_motion_emb = resources.wrapper.get_co_embeddings(
            text_embs,
            pos_ohot,
            cap_lens,
            pred_motion_263d,
            lengths,
        )

        motion_annotation_list.append(gt_motion_emb.detach().cpu().numpy())
        motion_pred_list.append(pred_motion_emb.detach().cpu().numpy())

        r_precision_total += calculate_R_precision(
            gt_text_emb.detach().cpu().numpy(),
            pred_motion_emb.detach().cpu().numpy(),
            top_k=3,
            sum_all=True,
        )
        matching_score_total += float(
            calculate_matching_score(
                gt_text_emb.detach().cpu().numpy(),
                pred_motion_emb.detach().cpu().numpy(),
                sum_all=True,
            )
        )
        sample_count += int(lengths.shape[0])

        if batch_index < multimodality_batches:
            multimodality_rollouts: list[torch.Tensor] = []
            for _ in range(multimodality_repeats):
                repeated_features = _generate_motion_sequence(
                    generator,
                    text_clip=text_clip,
                    seed_positions=seed_positions,
                    num_frames=num_frames,
                )
                repeated_motion_263d = _prepare_motion_batch(
                    repeated_features,
                    generator_normalizer=generator.normalizer,
                    resources=resources,
                )
                repeated_motion_emb = resources.wrapper.get_motion_embeddings(
                    repeated_motion_263d,
                    lengths,
                )
                multimodality_rollouts.append(repeated_motion_emb.unsqueeze(1))
            multimodality_batches_list.append(
                torch.cat(multimodality_rollouts, dim=1).detach().cpu().numpy()
            )

    if not motion_annotation_list or not motion_pred_list:
        raise RuntimeError("Evaluation loader produced no batches.")

    motion_annotation_np = np.concatenate(motion_annotation_list, axis=0)
    motion_pred_np = np.concatenate(motion_pred_list, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    pred_mu, pred_cov = calculate_activation_statistics(motion_pred_np)

    if motion_pred_np.shape[0] > 1:
        diversity_times = min(300, motion_pred_np.shape[0] - 1)
        diversity = float(calculate_diversity(motion_pred_np, diversity_times))
    else:
        diversity = 0.0

    fid = float(calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov))
    r_precision = r_precision_total / max(sample_count, 1)
    matching_score = matching_score_total / max(sample_count, 1)

    if multimodality_batches_list:
        multimodality_np = np.concatenate(multimodality_batches_list, axis=0)
        multimodality_times = min(10, multimodality_repeats - 1)
        multimodality = float(
            calculate_multimodality(multimodality_np, max(1, multimodality_times))
        )
    else:
        multimodality = 0.0

    summary = HumanMotionEvalSummary(
        fid=fid,
        diversity=diversity,
        r_precision=r_precision,
        matching_score=matching_score,
        multimodality=multimodality,
    )

    print(
        f"--> Eva. Ep {epoch}: FID {summary.fid:.4f}, Diversity {summary.diversity:.4f}, "
        f"R_precision ({summary.r_precision[0]:.4f}, {summary.r_precision[1]:.4f}, {summary.r_precision[2]:.4f}), "
        f"matching_score {summary.matching_score:.4f}, multimodality {summary.multimodality:.4f}"
    )

    if writer is not None:
        writer.add_scalar("./Test/FID", summary.fid, epoch)
        writer.add_scalar("./Test/Diversity", summary.diversity, epoch)
        writer.add_scalar("./Test/R_precision_top1", float(summary.r_precision[0]), epoch)
        writer.add_scalar("./Test/R_precision_top2", float(summary.r_precision[1]), epoch)
        writer.add_scalar("./Test/R_precision_top3", float(summary.r_precision[2]), epoch)
        writer.add_scalar("./Test/matching_score", summary.matching_score, epoch)
        writer.add_scalar("./Test/multimodality", summary.multimodality, epoch)

    return summary


evaluation_human_motion_generation = evaluate_human_motion_generation