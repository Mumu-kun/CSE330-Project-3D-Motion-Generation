from __future__ import annotations

from pathlib import Path

import torch

from t2m_eval_modules import (
    MovementConvEncoder,
    MotionEncoderBiGRUCo,
    TextEncoderBiGRUCo,
)
from word_vectorizer import POS_enumerator


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "text_mot_match" / "model" / "finest.tar"
)


def _resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path:
    resolved = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {resolved}")
    return resolved


def build_models(opt):
    checkpoint_path = _resolve_checkpoint_path(opt.get("checkpoint_path"))

    movement_enc = MovementConvEncoder(
        opt["dim_pose"] - 4,
        opt["dim_movement_enc_hidden"],
        opt["dim_movement_latent"],
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=opt["dim_word"],
        pos_size=opt["dim_pos_ohot"],
        hidden_size=opt["dim_text_hidden"],
        output_size=opt["dim_coemb_hidden"],
        device=opt["device"],
    )
    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt["dim_movement_latent"],
        hidden_size=opt["dim_motion_hidden"],
        output_size=opt["dim_coemb_hidden"],
        device=opt["device"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=opt["device"],
        weights_only=False,
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(f"Loading Evaluation Model Wrapper (Epoch {checkpoint['epoch']}) Completed!!")
    return text_enc, motion_enc, movement_enc


class EvaluatorWrapper(object):
    def __init__(self, dataset_name, device, checkpoint_path: str | Path | None = None):
        if dataset_name in {"humanml", "t2m"}:
            dim_pose = 263
        elif dataset_name == "kit":
            dim_pose = 251
        else:
            raise KeyError("Dataset not Recognized!!!")

        self.opt = {
            "dataset_name": dataset_name,
            "device": device,
            "dim_word": 300,
            "max_motion_length": 196,
            "dim_pos_ohot": len(POS_enumerator),
            "dim_motion_hidden": 1024,
            "max_text_len": 20,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": dim_pose,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "unit_length": 4,
            "checkpoint_path": checkpoint_path,
        }
        self.device = device

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(self.opt)

        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    @staticmethod
    def _restore_order(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        restore_idx = torch.argsort(order)
        return tensor[restore_idx]

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            cap_lens = cap_lens.detach().to(self.device)
            m_lens = m_lens.detach().to(self.device)

            align_idx = torch.argsort(m_lens, descending=True)
            sorted_word_embs = word_embs[align_idx]
            sorted_pos_ohot = pos_ohot[align_idx]
            sorted_cap_lens = cap_lens[align_idx]
            sorted_motions = motions[align_idx]
            sorted_m_lens = m_lens[align_idx]

            movements = self.movement_encoder(sorted_motions[..., :-4]).detach()
            sorted_m_lens = sorted_m_lens // self.opt["unit_length"]
            motion_embedding = self.motion_encoder(movements, sorted_m_lens)

            text_embedding = self.text_encoder(
                sorted_word_embs,
                sorted_pos_ohot,
                sorted_cap_lens,
            )

            return (
                self._restore_order(text_embedding, align_idx),
                self._restore_order(motion_embedding, align_idx),
            )

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device)

            align_idx = torch.argsort(m_lens, descending=True)
            sorted_motions = motions[align_idx]
            sorted_m_lens = m_lens[align_idx]

            movements = self.movement_encoder(sorted_motions[..., :-4]).detach()
            sorted_m_lens = sorted_m_lens // self.opt["unit_length"]
            motion_embedding = self.motion_encoder(movements, sorted_m_lens)

            return self._restore_order(motion_embedding, align_idx)


def build_evaluators(opt):
    return build_models(opt)