import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluator.t2m_eval_wrapper import EvaluatorWrapper
from utils.motion_process import recover_from_ric


def test_evaluator_wrapper_summarizes_native_tensors() -> None:
    wrapper = EvaluatorWrapper(dataset_name="t2m", device="cpu")

    word_embs = torch.zeros(2, 4, 300)
    pos_ohot = torch.zeros(2, 4, 15)
    cap_lens = torch.tensor([4, 3])
    motions = torch.zeros(2, 5, 271)
    m_lens = torch.tensor([5, 4])

    text_embedding, motion_embedding = wrapper.get_co_embeddings(
        word_embs,
        pos_ohot,
        cap_lens,
        motions,
        m_lens,
    )

    assert text_embedding.shape == (2, 512)
    assert motion_embedding.shape == (2, 512)


def test_recover_from_ric_handles_263d_compatibility_layout() -> None:
    features = torch.zeros(3, 263)
    recovered = recover_from_ric(features, num_joint=22)

    assert recovered.shape == (3, 22, 3)
    assert torch.isfinite(recovered).all()