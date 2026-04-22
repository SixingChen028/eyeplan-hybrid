import argparse
import csv
import json
import os
import pickle
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling_mean_last(values: list[float], window: int) -> float:
    if not values:
        return float("nan")
    window = max(1, min(window, len(values)))
    return float(sum(values[-window:]) / window)


def _rolling_mean_best(values: list[float], window: int) -> float:
    if not values:
        return float("nan")
    window = max(1, min(window, len(values)))
    best = float("-inf")
    running = sum(values[:window])
    best = max(best, running / window)
    for idx in range(window, len(values)):
        running += values[idx] - values[idx - window]
        best = max(best, running / window)
    return float(best)


def _find_run_dirs(root: str) -> list[str]:
    run_dirs: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        if "metadata.json" in filenames:
            run_dirs.append(dirpath)
    run_dirs.sort()
    return run_dirs


def _read_json(path: str) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _read_pickle(path: str) -> dict:
    with open(path, "rb") as file:
        return pickle.load(file)


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _safe_round(value: float, digits: int = 6):
    if value != value:
        return value
    return round(float(value), digits)


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: str, rows: list[dict], columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def _safe_float(value):
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _plot_top_runs(rows: list[dict], score_key: str, output_path: str, top_k: int) -> None:
    top_rows = rows[: max(1, min(top_k, len(rows)))]
    labels = [os.path.basename(row["run_dir"]) for row in top_rows]
    scores = [float(row[score_key]) for row in top_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_rows)), scores)
    plt.xticks(range(len(top_rows)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel(score_key)
    plt.title(f"Top {len(top_rows)} Runs by {score_key}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_score_vs_runtime(rows: list[dict], score_key: str, output_path: str) -> None:
    x = []
    y = []
    for row in rows:
        runtime = _safe_float(row.get("total_train_time_s"))
        score = _safe_float(row.get(score_key))
        if runtime is None or score is None:
            continue
        x.append(runtime)
        y.append(score)

    if not x:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("Total Train Time (s)")
    plt.ylabel(score_key)
    plt.title(f"{score_key} vs Runtime")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_param_effects(rows: list[dict], score_key: str, params: list[str], output_path: str) -> None:
    params = [param for param in params if param]
    params = [param for param in params if any(param in row for row in rows)]
    if not params:
        return

    n = len(params)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, param in enumerate(params):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            score = _safe_float(row.get(score_key))
            if score is None:
                continue
            if param not in row:
                continue
            groups[str(row[param])].append(score)

        keys = sorted(groups.keys(), key=lambda x: _safe_float(x) if _safe_float(x) is not None else x)
        means = [statistics.mean(groups[key]) for key in keys]
        stdevs = [statistics.stdev(groups[key]) if len(groups[key]) > 1 else 0.0 for key in keys]

        ax = axes[idx]
        ax.errorbar(range(len(keys)), means, yerr=stdevs, marker="o", capsize=3)
        ax.set_title(param)
        ax.set_xlabel("value")
        ax.set_ylabel(score_key)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.grid(alpha=0.25)

    for idx in range(len(params), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Hyperparameter Effects on {score_key}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_record(
    run_dir: str,
    metadata: dict,
    training: dict,
    ma_window: int,
    score_key: str,
) -> dict:
    args = metadata.get("args", {})
    rewards = _to_float_list(training.get("episode_reward", []))
    losses = _to_float_list(training.get("loss", []))
    step_time_s = _to_float_list(training.get("step_time_s", []))
    cumulative_time_s = _to_float_list(training.get("cumulative_time_s", []))

    num_updates = len(rewards)
    total_train_time_s = cumulative_time_s[-1] if cumulative_time_s else sum(step_time_s)
    mean_step_time_ms = (sum(step_time_s) / len(step_time_s) * 1000.0) if step_time_s else float("nan")

    record = {
        "run_dir": run_dir,
        "started_at_utc": metadata.get("started_at_utc"),
        "git_sha": metadata.get("git_sha"),
        "num_updates": num_updates,
        "total_train_time_s": _safe_round(total_train_time_s),
        "updates_per_s": _safe_round(num_updates / max(total_train_time_s, 1e-9)) if num_updates > 0 else float("nan"),
        "mean_step_time_ms": _safe_round(mean_step_time_ms),
        "final_episode_reward": _safe_round(rewards[-1]) if rewards else float("nan"),
        "best_episode_reward": _safe_round(max(rewards)) if rewards else float("nan"),
        "final_reward_ma": _safe_round(_rolling_mean_last(rewards, ma_window)),
        "best_reward_ma": _safe_round(_rolling_mean_best(rewards, ma_window)),
        "final_loss": _safe_round(losses[-1]) if losses else float("nan"),
    }

    for key, value in args.items():
        record[f"arg_{key}"] = value

    if score_key not in record:
        raise ValueError(f"score_key='{score_key}' is not available in run record.")

    return record


def _aggregate_by_group(rows: list[dict], group_keys: list[str], score_key: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        group_id = tuple(row.get(key) for key in group_keys)
        groups[group_id].append(row)

    summary_rows: list[dict] = []
    for group_id, group_rows in groups.items():
        score_values = [float(row[score_key]) for row in group_rows]
        runtime_values = [float(row["total_train_time_s"]) for row in group_rows if row["total_train_time_s"] == row["total_train_time_s"]]
        updates_per_s_values = [float(row["updates_per_s"]) for row in group_rows if row["updates_per_s"] == row["updates_per_s"]]

        summary = {key: value for key, value in zip(group_keys, group_id)}
        summary["n_runs"] = len(group_rows)
        summary[f"{score_key}_mean"] = _safe_round(statistics.mean(score_values))
        summary[f"{score_key}_stdev"] = _safe_round(statistics.stdev(score_values)) if len(score_values) > 1 else 0.0
        summary["runtime_mean_s"] = _safe_round(statistics.mean(runtime_values)) if runtime_values else float("nan")
        summary["updates_per_s_mean"] = _safe_round(statistics.mean(updates_per_s_values)) if updates_per_s_values else float("nan")
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: float(row[f"{score_key}_mean"]), reverse=True)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JAX grid-search runs.")
    parser.add_argument("--results_root", type=str, default=os.path.join("results", "jax-grid-cpu"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_file", type=str, default="data_training_jax.p")
    parser.add_argument("--ma_window", type=int, default=100)
    parser.add_argument("--score_key", type=str, default="final_reward_ma")
    parser.add_argument(
        "--group_by",
        type=str,
        default="arg_batch_size,arg_lr,arg_lamda,arg_beta_e_init",
        help="Comma-separated columns for aggregation.",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--plot_params", type=str, default=None, help="Comma-separated arg columns to plot.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_root
    os.makedirs(output_dir, exist_ok=True)

    run_dirs = _find_run_dirs(args.results_root)
    rows: list[dict] = []
    skipped_no_training = 0

    for run_dir in run_dirs:
        metadata_path = os.path.join(run_dir, "metadata.json")
        training_path = os.path.join(run_dir, args.data_file)
        if not os.path.exists(training_path):
            skipped_no_training += 1
            continue

        metadata = _read_json(metadata_path)
        training = _read_pickle(training_path)
        row = _run_record(
            run_dir=run_dir,
            metadata=metadata,
            training=training,
            ma_window=args.ma_window,
            score_key=args.score_key,
        )
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No runs with metadata.json and {args.data_file} found under {args.results_root}"
        )

    rows.sort(key=lambda row: float(row[args.score_key]), reverse=True)
    group_keys = [part.strip() for part in args.group_by.split(",") if part.strip()]
    summary_rows = _aggregate_by_group(rows, group_keys=group_keys, score_key=args.score_key)

    runs_csv = os.path.join(output_dir, "grid_analysis_runs.csv")
    summary_csv = os.path.join(output_dir, "grid_analysis_summary.csv")
    top_md = os.path.join(output_dir, "grid_analysis_top.md")

    run_fields = sorted({key for row in rows for key in row.keys()})
    summary_fields = list(dict.fromkeys(group_keys + [
        "n_runs",
        f"{args.score_key}_mean",
        f"{args.score_key}_stdev",
        "runtime_mean_s",
        "updates_per_s_mean",
    ]))

    _write_csv(runs_csv, rows, run_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)

    top_rows = rows[: max(1, args.top_k)]
    top_columns = [
        "run_dir",
        args.score_key,
        "final_episode_reward",
        "best_episode_reward",
        "num_updates",
        "total_train_time_s",
        "arg_batch_size",
        "arg_lr",
        "arg_lamda",
        "arg_beta_e_init",
        "arg_seed",
    ]
    top_columns = [col for col in top_columns if col in run_fields]
    _write_markdown(top_md, top_rows, top_columns)

    default_plot_params = [key for key in group_keys if key.startswith("arg_")]
    plot_params = (
        [part.strip() for part in args.plot_params.split(",") if part.strip()]
        if args.plot_params
        else default_plot_params
    )
    top_plot = os.path.join(output_dir, "grid_plot_top_runs.png")
    runtime_plot = os.path.join(output_dir, "grid_plot_score_vs_runtime.png")
    params_plot = os.path.join(output_dir, "grid_plot_param_effects.png")
    _plot_top_runs(rows=rows, score_key=args.score_key, output_path=top_plot, top_k=args.top_k)
    _plot_score_vs_runtime(rows=rows, score_key=args.score_key, output_path=runtime_plot)
    _plot_param_effects(rows=rows, score_key=args.score_key, params=plot_params, output_path=params_plot)

    print(f"Analyzed runs: {len(rows)}")
    print(f"Skipped (missing {args.data_file}): {skipped_no_training}")
    print(f"Wrote: {runs_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {top_md}")
    print(f"Wrote: {top_plot}")
    print(f"Wrote: {runtime_plot}")
    print(f"Wrote: {params_plot}")


if __name__ == "__main__":
    main()
