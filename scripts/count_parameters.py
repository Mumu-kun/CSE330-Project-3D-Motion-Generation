from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Config
from utils.checkpoint_eval import load_checkpoint_config
from utils.train_utils import Trainer


def _count_parameters(module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def _load_config(checkpoint: str | None) -> Config:
    if checkpoint is None:
        return Config()
    return load_checkpoint_config(checkpoint_path=checkpoint, map_location="cpu")


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
    encoder, predictor = Trainer._build_models_from_config(config, normalizer=None)

    encoder_counts = _count_parameters(encoder)
    predictor_counts = _count_parameters(predictor)

    combined_total = encoder_counts["total"] + predictor_counts["total"]
    combined_trainable = (
        encoder_counts["trainable"] + predictor_counts["trainable"]
    )

    result = {
        "config_source": (
            str(Path(args.checkpoint).resolve()) if args.checkpoint else "Config() defaults"
        ),
        "encoder": encoder_counts,
        "predictor": predictor_counts,
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
    print(
        "MotionHistoryEncoder: "
        f"total={encoder_counts['total']:,} "
        f"trainable={encoder_counts['trainable']:,}"
    )
    print(
        "FlowMatchingPredictor: "
        f"total={predictor_counts['total']:,} "
        f"trainable={predictor_counts['trainable']:,}"
    )
    print(
        "Combined: "
        f"total={combined_total:,} "
        f"trainable={combined_trainable:,}"
    )
    if args.include_ema:
        print(
            "Combined with EMA copies: "
            f"total={combined_total * 2:,} "
            f"trainable={combined_trainable:,}"
        )


if __name__ == "__main__":
    main()
