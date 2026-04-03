from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
COLOR_PALETTE = [
    "#0B6E4F",
    "#C84C09",
    "#1D4E89",
    "#9A031E",
    "#6C757D",
]


@dataclass
class SamplingPlotConfig:
    t_sampling_mode: str = "power"
    t_sampling_power: float = 4.0
    t_sampling_power_warmup_fraction: float = 0.1
    num_epochs: int = 800


def _load_config_defaults() -> tuple[SamplingPlotConfig, str | None]:
    """Load plotting defaults from Config when possible, otherwise fall back."""
    try:
        from config import Config

        cfg = Config()
        return (
            SamplingPlotConfig(
                t_sampling_mode=str(cfg.t_sampling_mode).lower(),
                t_sampling_power=float(cfg.t_sampling_power),
                t_sampling_power_warmup_fraction=float(
                    cfg.t_sampling_power_warmup_fraction
                ),
                num_epochs=int(cfg.num_epochs),
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return SamplingPlotConfig(), f"{type(exc).__name__}: {exc}"


def _compute_t_sampling_power(
    epoch: int,
    *,
    t_sampling_mode: str,
    t_sampling_power: float,
    t_sampling_power_warmup_fraction: float,
    num_epochs: int,
) -> float:
    """Mirror TrainUtils._compute_t_sampling_power."""
    if t_sampling_mode != "power":
        return 0.0

    target_power = max(0.0, float(t_sampling_power))
    if target_power == 0.0 or num_epochs <= 1:
        return target_power

    warmup_fraction = min(max(float(t_sampling_power_warmup_fraction), 0.0), 1.0)
    if warmup_fraction <= 0.0:
        return target_power

    progress = float(epoch) / float(num_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    if progress >= warmup_fraction:
        return target_power

    return target_power * (progress / warmup_fraction)


def _power_pdf(t: np.ndarray, power: float) -> np.ndarray:
    """Return p(t) = (k + 1) t^k over [0, 1]."""
    return (power + 1.0) * np.power(t, power)


def _sample_training_timesteps(
    *,
    num_samples: int,
    epoch: int,
    cfg: SamplingPlotConfig,
    seed: int,
) -> np.ndarray:
    """Mirror Trainer._sample_training_timesteps with torch CPU sampling."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(epoch) * 9973)

    u = torch.rand(
        max(1, int(num_samples)),
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    if cfg.t_sampling_mode == "uniform":
        return u.numpy()

    power = _compute_t_sampling_power(
        epoch,
        t_sampling_mode=cfg.t_sampling_mode,
        t_sampling_power=cfg.t_sampling_power,
        t_sampling_power_warmup_fraction=cfg.t_sampling_power_warmup_fraction,
        num_epochs=cfg.num_epochs,
    )
    if power <= 0.0:
        return u.numpy()
    return u.pow(1.0 / (power + 1.0)).numpy()


def _build_epoch_markers(num_epochs: int, warmup_fraction: float) -> list[int]:
    last_epoch = max(0, num_epochs - 1)
    candidates = [
        0,
        int(round(last_epoch * 0.125)),
        int(round(last_epoch * warmup_fraction)),
        last_epoch,
    ]
    deduped: list[int] = []
    for epoch in candidates:
        epoch = max(0, min(last_epoch, epoch))
        if epoch not in deduped:
            deduped.append(epoch)
    return sorted(deduped)


def plot_distribution(
    cfg: SamplingPlotConfig,
    *,
    output_path: Path,
    num_samples: int,
    num_bins: int,
    epochs: list[int] | None = None,
    seed: int = 42,
    fallback_reason: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(1e-4, 1.0, 1000)
    epochs = (
        sorted(
            {max(0, min(max(0, cfg.num_epochs - 1), int(epoch))) for epoch in epochs}
        )
        if epochs
        else _build_epoch_markers(
            cfg.num_epochs,
            cfg.t_sampling_power_warmup_fraction,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_hist, ax_schedule = axes
    fig.patch.set_facecolor("#F7F7F5")
    for ax in axes:
        ax.set_facecolor("#FCFCFA")

    bin_edges = np.linspace(0.0, 1.0, num=max(2, int(num_bins)) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for index, epoch in enumerate(epochs):
        power = _compute_t_sampling_power(
            epoch,
            t_sampling_mode=cfg.t_sampling_mode,
            t_sampling_power=cfg.t_sampling_power,
            t_sampling_power_warmup_fraction=cfg.t_sampling_power_warmup_fraction,
            num_epochs=cfg.num_epochs,
        )
        samples = _sample_training_timesteps(
            num_samples=num_samples,
            epoch=epoch,
            cfg=cfg,
            seed=seed,
        )
        progress = 1.0 if cfg.num_epochs <= 1 else epoch / (cfg.num_epochs - 1)
        hist, _ = np.histogram(samples, bins=bin_edges, density=True)
        color = COLOR_PALETTE[index % len(COLOR_PALETTE)]

        ax_hist.plot(
            bin_centers,
            hist,
            color=color,
            linewidth=2,
            label=(
                f"sampled epoch {epoch} | progress={progress:.3f} | "
                f"k={power:.2f} | mean={samples.mean():.3f}"
            ),
        )
        ax_hist.plot(
            t,
            _power_pdf(t, power),
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.75,
        )

    all_epochs = np.arange(max(1, cfg.num_epochs))
    all_powers = np.array(
        [
            _compute_t_sampling_power(
                int(epoch),
                t_sampling_mode=cfg.t_sampling_mode,
                t_sampling_power=cfg.t_sampling_power,
                t_sampling_power_warmup_fraction=cfg.t_sampling_power_warmup_fraction,
                num_epochs=cfg.num_epochs,
            )
            for epoch in all_epochs
        ]
    )
    warmup_end_epoch = int(
        round(max(0, cfg.num_epochs - 1) * cfg.t_sampling_power_warmup_fraction)
    )

    ax_hist.set_title("Empirical Distribution of Sampled Training Timesteps")
    ax_hist.set_xlabel("t")
    ax_hist.set_ylabel("density")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_ylim(bottom=0.0)
    ax_hist.grid(True, color="#D8D8D3", alpha=0.65, linewidth=0.8)
    ax_hist.legend(fontsize=8)
    ax_hist.text(
        0.02,
        0.98,
        "Solid = sampled histogram density\nDashed = analytic PDF",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    ax_schedule.plot(all_epochs, all_powers, color="tab:blue", linewidth=2)
    ax_schedule.axvline(
        warmup_end_epoch,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"warmup end ~ epoch {warmup_end_epoch}",
    )
    ax_schedule.set_title("Effective Power Schedule")
    ax_schedule.set_xlabel("epoch")
    ax_schedule.set_ylabel("k")
    ax_schedule.set_xlim(0, max(0, cfg.num_epochs - 1))
    ax_schedule.set_ylim(bottom=0.0)
    ax_schedule.grid(True, color="#D8D8D3", alpha=0.65, linewidth=0.8)
    ax_schedule.legend(fontsize=8)

    title = (
        "Training timestep sampling diagnostics "
        f"(mode={cfg.t_sampling_mode}, target_power={cfg.t_sampling_power}, "
        f"warmup_fraction={cfg.t_sampling_power_warmup_fraction}, "
        f"num_epochs={cfg.num_epochs}, samples/epoch={num_samples})"
    )
    if fallback_reason is not None:
        title += "\nusing fallback defaults because Config import failed"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    # default_cfg, fallback_reason = _load_config_defaults()
    default_cfg = SamplingPlotConfig()
    fallback_reason = "Config import disabled in current code version"
    parser = argparse.ArgumentParser(
        description="Plot the training timestep sampling distribution.",
    )
    parser.set_defaults(_fallback_reason=fallback_reason)
    parser.add_argument(
        "--mode",
        default=default_cfg.t_sampling_mode,
        choices=["uniform", "power"],
        help="Sampling mode used during training.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=default_cfg.t_sampling_power,
        help="Target power-law exponent k in p(t)=(k+1)t^k.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=default_cfg.t_sampling_power_warmup_fraction,
        help="Fraction of training spent linearly ramping k from 0 to target.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=default_cfg.num_epochs,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "diagnostics" / "t_sampling_distribution.png",
        help="Destination image path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Number of timesteps to sample per selected epoch.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins for the empirical distribution.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit epoch indices to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SamplingPlotConfig(
        t_sampling_mode=args.mode,
        t_sampling_power=max(0.0, float(args.power)),
        t_sampling_power_warmup_fraction=min(max(args.warmup_fraction, 0.0), 1.0),
        num_epochs=max(1, int(args.num_epochs)),
    )
    plot_distribution(
        cfg,
        output_path=args.output,
        num_samples=max(1, int(args.num_samples)),
        num_bins=max(2, int(args.bins)),
        epochs=args.epochs,
        seed=int(args.seed),
        fallback_reason=args._fallback_reason,
    )
    print(f"Saved plot to {args.output}")
    selected_epochs = (
        sorted(
            {
                max(0, min(max(0, cfg.num_epochs - 1), int(epoch)))
                for epoch in args.epochs
            }
        )
        if args.epochs
        else _build_epoch_markers(
            cfg.num_epochs,
            cfg.t_sampling_power_warmup_fraction,
        )
    )
    for epoch in selected_epochs:
        samples = _sample_training_timesteps(
            num_samples=max(1, int(args.num_samples)),
            epoch=epoch,
            cfg=cfg,
            seed=int(args.seed),
        )
        power = _compute_t_sampling_power(
            epoch,
            t_sampling_mode=cfg.t_sampling_mode,
            t_sampling_power=cfg.t_sampling_power,
            t_sampling_power_warmup_fraction=cfg.t_sampling_power_warmup_fraction,
            num_epochs=cfg.num_epochs,
        )
        print(
            "epoch="
            f"{epoch}, k={power:.4f}, mean={samples.mean():.6f}, "
            f"std={samples.std():.6f}, p50={np.quantile(samples, 0.5):.6f}, "
            f"p90={np.quantile(samples, 0.9):.6f}, p99={np.quantile(samples, 0.99):.6f}"
        )
    if args._fallback_reason is not None:
        print(f"Config import fallback: {args._fallback_reason}")


if __name__ == "__main__":
    main()
