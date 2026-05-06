import argparse
import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.results_layout import (
    get_run_analysis_dir,
    get_summary_analysis_dir,
    resolve_analysis_target,
)


def _value_key(value) -> str:
    return json.dumps(value, sort_keys=True)


def _run_args_from_metadata(run_dir: str) -> dict:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file for run: {metadata_path}")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    args = metadata.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"metadata.json must contain an object at key 'args': {metadata_path}")
    return args


def _varying_params_from_metadata(run_dirs: list[str]) -> list[str]:
    ignored_params = {
        "experiment",
        "parallel_config",
        "parallel_hyper_index",
        "parallel_seed_index",
        "parallel_varied_keys",
        "path",
        "resume",
    }
    run_args = [_run_args_from_metadata(run_dir) for run_dir in sorted(run_dirs)]
    common_params = set(run_args[0])
    for args in run_args[1:]:
        common_params &= set(args)

    varying_params: list[str] = []
    for param in sorted(common_params):
        if param in ignored_params:
            continue

        seen_values = set()
        for args in run_args:
            seen_values.add(_value_key(args[param]))

        if len(seen_values) > 1:
            varying_params.append(param)

    return varying_params


def _parse_training_log(run_dir: str, data_file: str) -> pd.DataFrame:
    data_path = os.path.join(run_dir, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    rows: list[list[str]] = []
    with open(data_path, "r") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("run_dir="):
                continue
            if stripped.startswith("update") or stripped.startswith("-"):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue
            rows.append(parts)

    if not rows:
        raise ValueError(f"No training rows found in: {data_path}")

    df = pd.DataFrame(rows)
    if df.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in: {data_path}")

    df = df.iloc[:, :4].copy()
    df.columns = ["update", "ep_num", "ep_rew", "ep_len"]
    for column in ["update", "ep_num", "ep_rew", "ep_len"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["update", "ep_rew", "ep_len"]).sort_values("update")
    if df.empty:
        raise ValueError(f"No valid update/ep_rew/ep_len rows found in: {data_path}")

    return df


def _format_group_label(group_params: list[str], run_args: dict) -> str:
    if not group_params:
        return "default"
    return ", ".join(f"{param}={run_args[param]}" for param in group_params)


def _plot_target(
    experiment: str,
    run_dirs: list[str],
    target_kind: str,
    target_name: str,
    results_root: str,
    data_file: str,
    output_prefix: str,
    facet: str | None,
) -> dict:
    run_dirs = sorted(run_dirs)
    varying_params = _varying_params_from_metadata(run_dirs) if len(run_dirs) > 1 else []
    group_params = [param for param in varying_params if param != "seed"]

    run_frames: list[tuple[str, pd.DataFrame, dict, str, str, str]] = []
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        df = _parse_training_log(run_dir=run_dir, data_file=data_file)
        run_args = _run_args_from_metadata(run_dir)
        facet_value = "all"
        if facet is not None:
            if facet not in run_args:
                raise ValueError(f"Facet '{facet}' not found in metadata args for run {run_id}")
            facet_value = str(run_args[facet])
        group_label = _format_group_label(group_params=group_params, run_args=run_args)
        run_frames.append((run_id, df, run_args, group_label, run_dir, facet_value))

    unique_groups = sorted({group_label for _, _, _, group_label, _, _ in run_frames})
    cmap = plt.get_cmap("tab10")
    color_by_group = {
        group_label: cmap(index % cmap.N)
        for index, group_label in enumerate(unique_groups)
    }

    facet_values = sorted({facet_value for _, _, _, _, _, facet_value in run_frames})
    faceted = facet is not None

    if faceted:
        n_cols = max(1, len(facet_values))
        fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 8), sharex=True)
        if n_cols == 1:
            axes = np.array(axes).reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    legend_seen: set[str] = set()
    multi_run = len(run_frames) > 1

    facet_col_idx = {value: index for index, value in enumerate(facet_values)}

    for run_id, df, run_args, group_label, run_dir, facet_value in run_frames:
        color = color_by_group[group_label]
        if multi_run:
            if group_label not in legend_seen:
                legend_label = group_label
                legend_seen.add(group_label)
            else:
                legend_label = None
        else:
            legend_label = run_id

        if faceted:
            col_idx = facet_col_idx[facet_value]
            reward_ax = axes[0, col_idx]
            length_ax = axes[1, col_idx]
            reward_ax.plot(df["update"], df["ep_rew"], color=color, alpha=0.75, linewidth=1.5, label=legend_label)
            length_ax.plot(df["update"], df["ep_len"], color=color, alpha=0.75, linewidth=1.5, label=legend_label)
        else:
            axes[0].plot(df["update"], df["ep_rew"], color=color, alpha=0.75, linewidth=1.5, label=legend_label)
            axes[1].plot(df["update"], df["ep_len"], color=color, alpha=0.75, linewidth=1.5, label=legend_label)

    if faceted:
        for col_idx, facet_value in enumerate(facet_values):
            reward_ax = axes[0, col_idx]
            length_ax = axes[1, col_idx]

            reward_ax.set_title(f"{facet}={facet_value}")
            reward_ax.set_ylabel("ep_rew")
            reward_ax.set_ylim(0, 1.3)
            reward_ax.grid(alpha=0.3)

            length_ax.set_xlabel("Update")
            length_ax.set_ylabel("ep_len")
            length_ax.set_ylim(0, 25)
            length_ax.grid(alpha=0.3)

        if legend_seen or (not multi_run and run_frames):
            axes[1, -1].legend(fontsize=8)
    else:
        axes[0].set_title("Episode Reward")
        axes[0].set_xlabel("Update")
        axes[0].set_ylabel("ep_rew")
        axes[0].set_ylim(0, 1.3)
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Episode Length")
        axes[1].set_xlabel("Update")
        axes[1].set_ylabel("ep_len")
        axes[1].set_ylim(0, 25)
        axes[1].grid(alpha=0.3)

        if legend_seen or (not multi_run and run_frames):
            axes[1].legend(fontsize=8)

    fig.suptitle(target_name)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if len(run_dirs) == 1 and target_kind == "run":
        run_id = os.path.basename(run_dirs[0])
        output_dir = get_run_analysis_dir(results_root=results_root, experiment=experiment, run_id=run_id)
    else:
        output_dir = get_summary_analysis_dir(results_root=results_root, experiment=experiment)

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{output_prefix}_curves.png")
    csv_path = os.path.join(output_dir, f"{output_prefix}_curves.csv")

    combined_rows = []
    for run_id, df, _, group_label, _, facet_value in run_frames:
        run_df = df[["update", "ep_rew", "ep_len"]].copy()
        run_df.insert(0, "facet", facet_value)
        run_df.insert(0, "group", group_label)
        run_df.insert(0, "run_id", run_id)
        combined_rows.append(run_df)

    pd.concat(combined_rows, ignore_index=True).to_csv(csv_path, index=False)
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    return {
        "experiment": experiment,
        "target_kind": target_kind,
        "run_count": len(run_dirs),
        "varying_params": varying_params,
        "group_params": group_params,
        "facet": facet,
        "facet_values": facet_values if faceted else [],
        "csv_path": csv_path,
        "fig_path": fig_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="+",
        type=str,
        help="One or more targets: <experiment>, results/runs/<experiment>, or results/runs/<experiment>/<run_id>.",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--data_file", type=str, default="training.log")
    parser.add_argument("--output_prefix", type=str, default="training")
    parser.add_argument("--facet", type=str, default=None)
    args = parser.parse_args()

    had_error = False
    for target_arg in args.targets:
        try:
            target = resolve_analysis_target(target_arg, results_root=args.results_root)
            print(
                f"target={target_arg} target_kind={target.kind} experiment={target.experiment} runs={len(target.run_dirs)}"
            )
            result = _plot_target(
                experiment=target.experiment,
                run_dirs=target.run_dirs,
                target_kind=target.kind,
                target_name=(os.path.basename(target.run_dirs[0]) if target.kind == "run" else target.experiment),
                results_root=args.results_root,
                data_file=args.data_file,
                output_prefix=args.output_prefix,
                facet=args.facet,
            )
            print(f"Runs plotted: {result['run_count']}")
            print(
                "Varying parameters: "
                + (", ".join(result["varying_params"]) if result["varying_params"] else "(none)")
            )
            print("Legend parameters: " + (", ".join(result["group_params"]) if result["group_params"] else "(none)"))
            print("Facet: " + (result["facet"] if result["facet"] else "(none)"))
            if result["facet_values"]:
                print("Facet values: " + ", ".join(result["facet_values"]))
            print("Wrote:")
            print(" ", result["csv_path"])
            print(" ", result["fig_path"])
        except Exception:
            had_error = True
            print(f"Error plotting target: {target_arg}", file=sys.stderr)
            traceback.print_exc()

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
