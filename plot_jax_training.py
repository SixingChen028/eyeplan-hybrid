import argparse
import json
import os
import pickle
import sys
import traceback

from modules.analysis_targets import (
    get_run_analysis_dir,
    resolve_analysis_target,
    select_most_recent_run,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_batch_size(run_dir: str) -> int:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    metadata_args = metadata.get("args", {})
    if "batch_size" not in metadata_args:
        raise KeyError(f"batch_size not found in metadata args: {metadata_path}")

    batch_size = int(metadata_args["batch_size"])
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive in metadata: {metadata_path}")
    return batch_size


def _analyze_run(run_dir: str, output_dir: str, ma_window: int, data_file: str, output_prefix: str) -> dict:
    batch_size = _read_batch_size(run_dir)
    data_path = os.path.join(run_dir, data_file)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    with open(data_path, "rb") as file:
        data = pickle.load(file)

    num_updates = len(data["episode_reward"])
    updates = np.arange(1, num_updates + 1)
    episodes = updates * batch_size

    df = pd.DataFrame(
        {
            "update": updates,
            "episodes": episodes,
            "episode_reward": np.asarray(data["episode_reward"], dtype=np.float64),
            "episode_length": np.asarray(data["episode_length"], dtype=np.float64),
            "loss": np.asarray(data["loss"], dtype=np.float64),
            "policy_loss": np.asarray(data["policy_loss"], dtype=np.float64),
            "value_loss": np.asarray(data["value_loss"], dtype=np.float64),
            "entropy_loss": np.asarray(data["entropy_loss"], dtype=np.float64),
            "step_time_s": np.asarray(data["step_time_s"], dtype=np.float64),
            "cumulative_time_s": np.asarray(data["cumulative_time_s"], dtype=np.float64),
        }
    )

    df["episode_reward_ma"] = df["episode_reward"].rolling(ma_window, min_periods=1).mean()
    df["episode_length_ma"] = df["episode_length"].rolling(ma_window, min_periods=1).mean()
    df["loss_ma"] = df["loss"].rolling(ma_window, min_periods=1).mean()

    summary = {
        "run_dir": run_dir,
        "num_updates": int(num_updates),
        "num_episodes": int(df["episodes"].iloc[-1]) if num_updates > 0 else 0,
        "final_episode_reward": round(float(df["episode_reward"].iloc[-1]), 6) if num_updates > 0 else np.nan,
        "final_episode_reward_ma": round(float(df["episode_reward_ma"].iloc[-1]), 6) if num_updates > 0 else np.nan,
        "final_episode_length": round(float(df["episode_length"].iloc[-1]), 6) if num_updates > 0 else np.nan,
        "final_loss": round(float(df["loss"].iloc[-1]), 6) if num_updates > 0 else np.nan,
        "mean_step_time_ms": round(float(df["step_time_s"].mean() * 1000.0), 6) if num_updates > 0 else np.nan,
        "total_train_time_s": round(float(df["cumulative_time_s"].iloc[-1]), 6) if num_updates > 0 else np.nan,
        "updates_per_s": round(float(num_updates / max(df["cumulative_time_s"].iloc[-1], 1e-9)), 6) if num_updates > 0 else np.nan,
    }

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{output_prefix}_curves.csv")
    summary_json_path = os.path.join(output_dir, f"{output_prefix}_summary.json")
    fig_path = os.path.join(output_dir, f"{output_prefix}_curves.png")

    df.to_csv(csv_path, index=False)
    with open(summary_json_path, "w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(2, 2, 1)
    ax.plot(df["episodes"], df["episode_reward"], alpha=0.35, linewidth=1, label="reward")
    ax.plot(df["episodes"], df["episode_reward_ma"], linewidth=2, label=f"reward ma({ma_window})")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 2)
    ax.plot(df["episodes"], df["episode_length"], alpha=0.35, linewidth=1, label="length")
    ax.plot(df["episodes"], df["episode_length_ma"], linewidth=2, label=f"length ma({ma_window})")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 3)
    ax.plot(df["episodes"], df["loss"], alpha=0.35, linewidth=1, label="loss")
    ax.plot(df["episodes"], df["loss_ma"], linewidth=2, label=f"loss ma({ma_window})")
    ax.set_title("Training Loss")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 4)
    ax.plot(df["episodes"], df["step_time_s"] * 1000.0, linewidth=1.5, label="step ms")
    ax.set_title("Per-Update Runtime")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Milliseconds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    result = {
        "run_dir": run_dir,
        "analysis_dir": output_dir,
        "csv_path": csv_path,
        "summary_json_path": summary_json_path,
        "fig_path": fig_path,
        "summary_json": json.dumps(summary, indent=2),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="+",
        type=str,
        help="One or more targets: <experiment>, <experiment>/<run_id>, <experiment>/*, or full path in runs/analysis.",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--ma_window", type=int, default=100)
    parser.add_argument("--data_file", type=str, default="data_training_jax.p")
    parser.add_argument("--output_prefix", type=str, default="training_jax")
    args = parser.parse_args()

    runs_to_analyze: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    had_error = False
    for target_arg in args.targets:
        try:
            target = resolve_analysis_target(target_arg, results_root=args.results_root)
            if target.kind == "experiment":
                run_dirs = [select_most_recent_run(target.run_dirs)]
            else:
                run_dirs = target.run_dirs

            print(f"target={target_arg} target_kind={target.kind} experiment={target.experiment} runs={len(run_dirs)}")
            for run_dir in run_dirs:
                run_key = (target.experiment, run_dir)
                if run_key in seen:
                    continue
                seen.add(run_key)
                runs_to_analyze.append(run_key)
        except Exception:
            had_error = True
            print(f"Error resolving target: {target_arg}", file=sys.stderr)
            traceback.print_exc()
            continue

    for experiment, run_dir in runs_to_analyze:
        try:
            run_id = os.path.basename(run_dir)
            output_dir = get_run_analysis_dir(
                results_root=args.results_root,
                experiment=experiment,
                run_id=run_id,
            )
            result = _analyze_run(
                run_dir=run_dir,
                output_dir=output_dir,
                ma_window=args.ma_window,
                data_file=args.data_file,
                output_prefix=args.output_prefix,
            )
            print("Saved:")
            print(" ", result["csv_path"])
            print(" ", result["summary_json_path"])
            print(" ", result["fig_path"])
            print(result["summary_json"])
        except Exception:
            had_error = True
            print(f"Error analyzing run: {run_dir}", file=sys.stderr)
            traceback.print_exc()
            continue

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
