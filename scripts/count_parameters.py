from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import Config
from utils.models.motion_history_encoder import JepaPredictor, LinearProbe, MotionHistoryEncoder
from utils.models.flow_matching_predictor import LatentDecoder


def _count_parameters(module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def _load_config(checkpoint: str | None) -> Config:
    return Config()


def _categorize_parameters(model: nn.Module) -> Dict[str, int]:
    """Categorize model parameters by component type."""
    categories: Dict[str, int] = {}
    for name, param in model.named_parameters():
        cat = _assign_category(name)
        categories[cat] = categories.get(cat, 0) + param.numel()
    return categories


def _assign_category(name: str) -> str:
    """Assign a category label to a parameter name."""
    name_lower = name.lower()
    if "cross_attn" in name_lower:
        return "cross_attn"
    if "self_attn" in name_lower or "attn" in name_lower:
        return "self_attn"
    if "mlp" in name_lower:
        return "mlp"
    if "adaln" in name_lower:
        return "adaln"
    if "layer_norm" in name_lower or "norm" in name_lower:
        return "layer_norm"
    if "register" in name_lower or "mask_token" in name_lower:
        return "learnable_tokens"
    if "linear" in name_lower or "proj" in name_lower:
        return "linear_proj"
    return "other"


def _print_model_summary(name: str, model: nn.Module) -> int:
    """Print detailed parameter summary for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}")
    print("-" * 60)
    print(f"{'Category':<30} {'Params':>12} {'%':>7}")
    print("-" * 60)
    categories = _categorize_parameters(model)
    for cat, count in categories.items():
        pct = count / total * 100 if total > 0 else 0.0
        print(f"{cat:<30} {count:>12,} {pct:>6.2f}%")
    print("-" * 60)
    return trainable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count model parameters for the current config or a checkpoint config."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If provided, uses the config stored in that checkpoint.",
    )
    parser.add_argument(
        "--include-ema",
        action="store_true",
        help="Also report total parameters if EMA copies are kept during training.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    args = parser.parse_args()

    config = _load_config(args.checkpoint)

    # Instantiate models directly without full trainer overhead (avoid EMA deepcopy OOM)
    encoder = MotionHistoryEncoder(config.encoder_config)
    jepa_predictor = JepaPredictor(config.encoder_config)
    linear_probe = LinearProbe(
        hidden_size=config.encoder_config.hidden_size,
        text_embedding_dim=config.text_embedding_dim,
    )
    decoder = LatentDecoder(config)

    encoder_counts = _count_parameters(encoder)
    jepa_predictor_counts = _count_parameters(jepa_predictor)
    probe_counts = _count_parameters(linear_probe)
    decoder_counts = _count_parameters(decoder)

    combined_total = encoder_counts["total"] + jepa_predictor_counts["total"]
    combined_trainable = encoder_counts["trainable"] + jepa_predictor_counts["trainable"]

    result = {
        "config_source": (str(Path(args.checkpoint).resolve()) if args.checkpoint else "Config() defaults"),
        "encoder": encoder_counts,
        "predictor": jepa_predictor_counts,
        "probe": probe_counts,
        "decoder": decoder_counts,
        "combined": {
            "total": int(combined_total),
            "trainable": int(combined_trainable),
            "non_trainable": int(combined_total - combined_trainable),
        },
    }

    if args.include_ema:
        result["combined_with_ema_copies"] = {
            "total": int(combined_total * 2),
            "trainable": int(combined_trainable),
            "non_trainable": int(combined_total * 2 - combined_trainable),
        }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Config source: {result['config_source']}")
    print()
    _print_model_summary("MotionHistoryEncoder", encoder)
    _print_model_summary("JepaPredictor", jepa_predictor)
    _print_model_summary("LinearProbe", linear_probe)
    _print_model_summary("LatentDecoder", decoder)
    print()
    print(f"Combined: total={combined_total:,} trainable={combined_trainable:,}")
    if args.include_ema:
        print(f"Combined with EMA copies: total={combined_total * 2:,} trainable={combined_trainable:,}")


if __name__ == "__main__":
    main()
