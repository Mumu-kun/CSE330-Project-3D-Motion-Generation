from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from utils.config import Config
from utils.models.pretrain_trainer import PretrainTrainer


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

    config.dataset_path = Path("tests/dataset/humanml3d-subset-mini")
    config.device = "cpu"

    trainer = PretrainTrainer(config)

    encoder_counts = _count_parameters(trainer.encoder)
    jepa_predictor_counts = _count_parameters(trainer.jepa_predictor)
    probe_counts = _count_parameters(trainer.linear_probe)

    combined_total = encoder_counts["total"] + jepa_predictor_counts["total"]
    combined_trainable = encoder_counts["trainable"] + jepa_predictor_counts["trainable"]

    result = {
        "config_source": (str(Path(args.checkpoint).resolve()) if args.checkpoint else "Config() defaults"),
        "encoder": encoder_counts,
        "predictor": jepa_predictor_counts,
        "probe": probe_counts,
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
    print(f"MotionHistoryEncoder: total={encoder_counts['total']:,} trainable={encoder_counts['trainable']:,}")
    print(f"JepaPredictor: total={jepa_predictor_counts['total']:,} trainable={jepa_predictor_counts['trainable']:,}")
    print(f"LinearProbe: total={probe_counts['total']:,} trainable={probe_counts['trainable']:,}")
    print(f"Combined: total={combined_total:,} trainable={combined_trainable:,}")
    if args.include_ema:
        print(f"Combined with EMA copies: total={combined_total * 2:,} trainable={combined_trainable:,}")


if __name__ == "__main__":
    main()
