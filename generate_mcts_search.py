#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from modules.config import DEFAULT_META, SHAPE_KEYS, expand_config_runs, is_list, load_config
from modules.mcts_search import (
    DEFAULT_MCTS_C_GRID,
    DEFAULT_MCTS_EVALS_PER_POINT,
    DEFAULT_MCTS_ROLLOUTS,
    MCTSSimulator,
)
from modules.train_results import env_cache_key, env_from_args, env_params_from_args, prepare_run_dirs


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _parse_c_grid(raw: str | None) -> list[float]:
    if raw is None:
        return list(DEFAULT_MCTS_C_GRID)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--c_grid must contain at least one value")
    if any(value < 0.0 for value in values):
        raise ValueError("--c_grid values must be non-negative")
    return values


def _write_evaluations(run_dir: str, evaluations: list[dict]) -> None:
    json_path = os.path.join(run_dir, "mcts_grid_search.json")
    with open(json_path, "w") as file:
        json.dump(evaluations, file, indent=2)
        file.write("\n")

    csv_path = os.path.join(run_dir, "mcts_grid_search.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(evaluations[0]))
        writer.writeheader()
        writer.writerows(evaluations)


def _write_plot(run_dir: str, evaluations: list[dict]) -> str:
    c_raw = np.asarray([item["c_raw"] for item in evaluations], dtype=float)
    reward = np.asarray([item["mean_episode_reward"] for item in evaluations], dtype=float)
    lengths = np.asarray([item["mean_episode_length"] for item in evaluations], dtype=float)

    fig, reward_ax = plt.subplots(figsize=(7.0, 4.5))
    reward_ax.plot(c_raw, reward, marker="o", color="#2563eb", label="episode reward")
    reward_ax.set_xlabel("c (raw reward units)")
    reward_ax.set_ylabel("Mean episode reward")
    reward_ax.grid(alpha=0.3)

    length_ax = reward_ax.twinx()
    length_ax.plot(c_raw, lengths, marker="s", color="#dc2626", label="episode length")
    length_ax.set_ylabel("Mean episode length")

    lines, labels = reward_ax.get_legend_handles_labels()
    more_lines, more_labels = length_ax.get_legend_handles_labels()
    reward_ax.legend(lines + more_lines, labels + more_labels, loc="best")
    fig.tight_layout()

    plot_path = os.path.join(run_dir, "mcts_grid_search.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _write_training_log(
    run_dir: str,
    *,
    run_index: int,
    seed: int,
    elapsed_seconds: float,
    num_trials: int,
    c_raw: float,
    c_scaled: float,
    num_rollouts: int,
    evals_per_point: int,
) -> None:
    log_path = os.path.join(run_dir, "training.log")
    with open(log_path, "a") as file:
        file.write("\n")
        file.write(
            "mcts_search_summary "
            f"run_index={run_index} "
            f"seed={seed} "
            f"elapsed_seconds={elapsed_seconds:.3f} "
            f"num_trials={num_trials} "
            f"c_raw={c_raw:.6g} "
            f"c_scaled={c_scaled:.6g} "
            f"num_rollouts={num_rollouts} "
            f"evals_per_point={evals_per_point}\n"
        )
        file.write(f"training_log={log_path}\n")


def _with_mcts_metadata(run: dict, *, c_raw: float, c_scaled: float, num_rollouts: int) -> dict:
    out = dict(run)
    out["label"] = "mcts_search"
    out["lesion_policy"] = "mcts_search"
    out["mcts_c_raw"] = float(c_raw)
    out["mcts_c_scaled"] = float(c_scaled)
    out["mcts_num_rollouts"] = int(num_rollouts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCTS baseline simulations for a TOML sweep.")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument("--experiment", help="Override experiment name. Defaults to <config stem>_mcts_search.")
    parser.add_argument("--condition", type=int, help="0-based [[conditions]] table index to generate.")
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--skip_timeout_trials", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=DEFAULT_MCTS_ROLLOUTS)
    parser.add_argument("--evals_per_point", type=int, default=DEFAULT_MCTS_EVALS_PER_POINT)
    parser.add_argument("--c_grid", help="Comma-separated c values in raw reward units.")
    parser.add_argument("--plot", action="store_true", help="Write mcts_grid_search.png for each run.")
    args, override_tokens = parser.parse_known_args()

    c_grid = _parse_c_grid(args.c_grid)
    config_path, config = load_config(args.config)
    meta = dict(DEFAULT_META)
    meta.update(config.get("meta", {}))

    # Flatten shape-key arrays to their first value; this script doesn't sweep them.
    params = config.get("params", {})
    for key in list(params):
        if key in SHAPE_KEYS and is_list(params[key]):
            params[key] = params[key][0]

    _, raw_runs, varied_keys, _, condition_index = expand_config_runs(
        config,
        condition_index=args.condition,
        override_tokens=override_tokens,
    )

    output_path = args.path or str(meta["result_path"])
    experiment = args.experiment or f"{config_path.stem}_mcts_search"
    start = time.time()
    simulators = {}

    for run_index, run in enumerate(raw_runs):
        env_key = env_cache_key(run)
        if env_key not in simulators:
            env = env_from_args(run)
            simulators[env_key] = MCTSSimulator(
                env,
                env_params_from_args(env, run),
                num_rollouts=int(args.num_rollouts),
            )
        simulator = simulators[env_key]

        evaluations = [
            asdict(
                simulator.evaluate(
                    c_raw=c_raw,
                    seed=int(run["seed"]) + 100_000 + c_idx,
                    num_trials=int(args.evals_per_point),
                )
            )
            for c_idx, c_raw in enumerate(c_grid)
        ]
        best_eval = max(evaluations, key=lambda item: item["mean_episode_reward"])
        run_with_metadata = _with_mcts_metadata(
            run,
            c_raw=best_eval["c_raw"],
            c_scaled=best_eval["c_scaled"],
            num_rollouts=int(args.num_rollouts),
        )
        run_dirs = prepare_run_dirs(
            [run_with_metadata],
            path=output_path,
            experiment=experiment,
            config_path=config_path,
            varied_keys=varied_keys,
            label=None,
            condition_index=condition_index,
            run_eval=False,
            eval_episodes=None,
        )
        run_dir = run_dirs[0]
        _write_evaluations(run_dir, evaluations)
        plot_path = _write_plot(run_dir, evaluations) if args.plot else None

        data = simulator.simulate(
            c_raw=float(best_eval["c_raw"]),
            seed=int(run["seed"]),
            num_trials=int(args.num_trials),
            skip_timeout_trials=bool(args.skip_timeout_trials),
        )
        output_file = os.path.join(run_dir, "data_simulation.json")
        with open(output_file, "w") as file:
            json.dump(_round_floats(data), file)
            file.write("\n")

        _write_training_log(
            run_dir,
            run_index=run_index,
            seed=int(run["seed"]),
            elapsed_seconds=time.time() - start,
            num_trials=len(data["actions"]),
            c_raw=float(best_eval["c_raw"]),
            c_scaled=float(best_eval["c_scaled"]),
            num_rollouts=int(args.num_rollouts),
            evals_per_point=int(args.evals_per_point),
        )
        message = (
            f"{run_index + 1}/{len(raw_runs)} {output_file} trials={len(data['actions'])} "
            f"best_c_raw={best_eval['c_raw']:.6g}"
        )
        if plot_path is not None:
            message += f" plot={plot_path}"
        print(message, flush=True)


if __name__ == "__main__":
    main()
