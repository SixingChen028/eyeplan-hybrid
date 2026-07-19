#!/usr/bin/env python3
"""Plot active working-memory items from a detailed simulation trace.

Example:
    .venv/bin/python simulate.py \
        results/runs/0712_main_alt/seed1_20260712_190737_h1gc \
        --detailed --num_trials=10000
    .venv/bin/python scripts/plot_effective_wm_capacity.py \
        results/runs/0712_main_alt/seed1_20260712_190737_h1gc/data_simulation_detailed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STEPS = (5, 10, 15)
DEFAULT_OUTPUT_DIR = Path("results/diagnostics/effective_wm_capacity")
COLOR = "#276FBF"


def _load_active_counts(path: Path) -> list[np.ndarray]:
    with path.open() as file:
        data = json.load(file)

    activation_trials = data.get("activations")
    if not isinstance(activation_trials, list) or not activation_trials:
        raise ValueError(f"{path} does not contain detailed activation traces")

    max_steps = max(len(trial) for trial in activation_trials)
    counts_by_step: list[list[int]] = [[] for _ in range(max_steps)]
    for trial in activation_trials:
        for step_index, activations in enumerate(trial):
            counts_by_step[step_index].append(int(np.count_nonzero(np.asarray(activations) > 0.0)))

    return [np.asarray(counts, dtype=np.int16) for counts in counts_by_step]


def _summarize(counts_by_step: list[np.ndarray], source: Path) -> dict[str, Any]:
    time_series = []
    for step, counts in enumerate(counts_by_step, start=1):
        mean = float(np.mean(counts))
        sd = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
        half_width = 1.96 * sd / np.sqrt(len(counts))
        time_series.append(
            {
                "time_step": step,
                "num_trials": int(len(counts)),
                "mean": mean,
                "sd": sd,
                "ci95_lower": mean - half_width,
                "ci95_upper": mean + half_width,
            }
        )

    return {
        "source": str(source.resolve()),
        "active_item_definition": "activation > 0",
        "time_step_definition": "one-based decision state before the action at that step",
        "conditioning": "trials with an activation state recorded at the time step",
        "num_trials": int(len(counts_by_step[0])),
        "time_series": time_series,
    }


def _plot_mean(summary: dict[str, Any], output_path: Path) -> None:
    rows = summary["time_series"]
    steps = np.asarray([row["time_step"] for row in rows])
    means = np.asarray([row["mean"] for row in rows])
    lowers = np.asarray([row["ci95_lower"] for row in rows])
    uppers = np.asarray([row["ci95_upper"] for row in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.fill_between(steps, lowers, uppers, color=COLOR, alpha=0.18, linewidth=0)
    ax.plot(steps, means, color=COLOR, linewidth=2.2)
    ax.set(xlabel="Time step", ylabel="Active items", title="Effective working-memory capacity over time")
    ax.set_xlim(1, int(steps[-1]))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distributions(
    counts_by_step: list[np.ndarray],
    steps: tuple[int, ...],
    output_path: Path,
) -> None:
    if any(step < 1 or step > len(counts_by_step) for step in steps):
        raise ValueError(f"distribution steps must be between 1 and {len(counts_by_step)}")

    selected = [counts_by_step[step - 1] for step in steps]
    max_count = max(int(np.max(counts)) for counts in selected)
    x = np.arange(max_count + 1)
    fig, axes = plt.subplots(1, len(steps), figsize=(10.0, 3.3), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, step, counts in zip(axes, steps, selected):
        frequencies = np.bincount(counts, minlength=max_count + 1) / len(counts)
        ax.bar(x, frequencies, width=0.82, color=COLOR, alpha=0.9)
        ax.set_title(f"Step {step}")
        ax.text(
            0.97,
            0.95,
            f"mean = {np.mean(counts):.2f}\nn = {len(counts):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.set_xticks(x)
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Proportion of trials")
    fig.supxlabel("Active items")
    fig.suptitle("Distribution of effective working-memory capacity")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detailed_simulation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    args = parser.parse_args()

    counts_by_step = _load_active_counts(args.detailed_simulation)
    summary = _summarize(counts_by_step, args.detailed_simulation)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "effective_wm_capacity_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    steps = tuple(args.steps)
    mean_path = output_dir / "active_items_over_time.png"
    distributions_path = output_dir / "active_items_distributions.png"
    _plot_mean(summary, mean_path)
    _plot_distributions(counts_by_step, steps, distributions_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {mean_path}")
    print(f"Wrote {distributions_path}")


if __name__ == "__main__":
    main()
