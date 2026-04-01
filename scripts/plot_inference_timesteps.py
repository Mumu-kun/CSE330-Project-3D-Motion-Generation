from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLOR_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
LINE_STYLES = ["-", "--", ":", "-."]


def build_inference_time_boundaries(num_steps: int, power: float) -> np.ndarray:
    """Mirror models._build_inference_time_boundaries on CPU with NumPy."""
    steps = max(1, int(num_steps))
    s = np.linspace(0.0, 1.0, num=steps + 1, dtype=np.float64)
    return 1.0 - np.power(1.0 - s, power)


def _style_for_series(
    *,
    num_steps_index: int,
    power_index: int,
    num_steps_count: int,
    power_count: int,
) -> tuple[str, str, str]:
    """Pick readable, consistent color/marker/linestyle combinations."""
    if power_count > 1 and num_steps_count > 1:
        color = COLOR_PALETTE[power_index % len(COLOR_PALETTE)]
        marker = MARKERS[num_steps_index % len(MARKERS)]
        line_style = LINE_STYLES[power_index % len(LINE_STYLES)]
        return color, marker, line_style

    if power_count > 1:
        color = COLOR_PALETTE[power_index % len(COLOR_PALETTE)]
        return color, "o", LINE_STYLES[power_index % len(LINE_STYLES)]

    color = COLOR_PALETTE[num_steps_index % len(COLOR_PALETTE)]
    marker = MARKERS[num_steps_index % len(MARKERS)]
    return color, marker, "-"


def plot_inference_timesteps(
    num_steps_values: list[int],
    powers: list[float],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_points, ax_dt = axes
    fig.patch.set_facecolor("#F7F7F5")
    for ax in axes:
        ax.set_facecolor("#FCFCFA")

    for row_index, num_steps in enumerate(num_steps_values):
        for power_index, power in enumerate(powers):
            tau = build_inference_time_boundaries(num_steps, power)
            ode_progress = np.linspace(0.0, 1.0, num=tau.shape[0], dtype=np.float64)
            dt = np.diff(tau)
            t_mid = 0.5 * (tau[:-1] + tau[1:])
            color, marker, line_style = _style_for_series(
                num_steps_index=row_index,
                power_index=power_index,
                num_steps_count=len(num_steps_values),
                power_count=len(powers),
            )
            label = f"steps={num_steps}, p={power:g}"

            ax_points.plot(
                tau,
                ode_progress,
                color=color,
                linestyle=line_style,
                linewidth=1.9,
                alpha=0.82,
            )
            ax_points.scatter(
                tau,
                ode_progress,
                marker=marker,
                s=34,
                color=color,
                edgecolors="white",
                linewidths=0.7,
                label=label,
            )

            ax_dt.plot(
                t_mid,
                dt,
                marker=marker,
                linestyle=line_style,
                linewidth=1.8,
                markersize=4,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=label,
            )

    ax_points.set_title("Selected Inference t Values")
    ax_points.set_xlabel("t")
    ax_points.set_ylabel("ODE integration progress")
    ax_points.set_xlim(0.0, 1.0)
    ax_points.set_ylim(0.0, 1.0)
    ax_points.grid(True, color="#D8D8D3", alpha=0.65, linewidth=0.8)
    ax_points.legend(fontsize=8)

    ax_dt.set_title("Interval Widths Across t")
    ax_dt.set_xlabel("interval midpoint t")
    ax_dt.set_ylabel("dt")
    ax_dt.set_xlim(0.0, 1.0)
    ax_dt.set_ylim(bottom=0.0)
    ax_dt.grid(True, color="#D8D8D3", alpha=0.65, linewidth=0.8)
    ax_dt.legend(fontsize=8)

    fig.suptitle(
        "Inference uses an end-biased power grid: "
        r"$\tau_i = 1 - (1 - i/N)^p$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected inference t values for several num_steps settings.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40],
        help="One or more num_steps values to visualize.",
    )
    parser.add_argument(
        "--powers",
        type=float,
        nargs="+",
        default=[2.0],
        help="One or more inference time-schedule powers p in t=1-(1-s)^p.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "diagnostics" / "inference_t_values.png",
        help="Destination image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_steps_values = sorted({max(1, int(value)) for value in args.steps})
    powers = [float(value) for value in args.powers]
    if any(value <= 0.0 for value in powers):
        raise ValueError("All --powers values must be positive.")
    plot_inference_timesteps(num_steps_values, powers, output_path=args.output)
    print(f"Saved plot to {args.output}")
    for num_steps in num_steps_values:
        for power in powers:
            tau = build_inference_time_boundaries(num_steps, power)
            tau_str = ", ".join(f"{value:.6f}" for value in tau)
            print(f"num_steps={num_steps}, power={power:g}: [{tau_str}]")


if __name__ == "__main__":
    main()
