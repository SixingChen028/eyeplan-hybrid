import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def resolve_run_dir(args: argparse.Namespace) -> str:
    if args.run_dir is not None:
        return args.run_dir

    if args.jobid is None:
        raise ValueError("Either --run_dir or --jobid must be provided.")

    return os.path.join(args.path, f"exp_{args.learning_rate}_{args.wm_decay}_{args.jobid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--jobid", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.2)
    parser.add_argument("--wm_decay", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--ma_window", type=int, default=100)
    parser.add_argument("--data_file", type=str, default="data_training_jax.p")
    parser.add_argument("--output_prefix", type=str, default="training_jax")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args)
    data_path = os.path.join(run_dir, args.data_file)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    with open(data_path, "rb") as file:
        data = pickle.load(file)

    num_updates = len(data["episode_reward"])
    updates = np.arange(1, num_updates + 1)
    episodes = updates * args.batch_size

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

    df["episode_reward_ma"] = df["episode_reward"].rolling(args.ma_window, min_periods=1).mean()
    df["episode_length_ma"] = df["episode_length"].rolling(args.ma_window, min_periods=1).mean()
    df["loss_ma"] = df["loss"].rolling(args.ma_window, min_periods=1).mean()

    summary = pd.DataFrame(
        [
            {
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
        ]
    )

    csv_path = os.path.join(run_dir, f"{args.output_prefix}_curves.csv")
    summary_csv_path = os.path.join(run_dir, f"{args.output_prefix}_summary.csv")
    summary_md_path = os.path.join(run_dir, f"{args.output_prefix}_summary.md")
    fig_path = os.path.join(run_dir, f"{args.output_prefix}_curves.png")

    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)

    with open(summary_md_path, "w") as file:
        file.write(_to_markdown_table(summary) + "\n")

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(2, 2, 1)
    ax.plot(df["episodes"], df["episode_reward"], alpha=0.35, linewidth=1, label="reward")
    ax.plot(df["episodes"], df["episode_reward_ma"], linewidth=2, label=f"reward ma({args.ma_window})")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 2)
    ax.plot(df["episodes"], df["episode_length"], alpha=0.35, linewidth=1, label="length")
    ax.plot(df["episodes"], df["episode_length_ma"], linewidth=2, label=f"length ma({args.ma_window})")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 2, 3)
    ax.plot(df["episodes"], df["loss"], alpha=0.35, linewidth=1, label="loss")
    ax.plot(df["episodes"], df["loss_ma"], linewidth=2, label=f"loss ma({args.ma_window})")
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

    print("Saved:")
    print(" ", csv_path)
    print(" ", summary_csv_path)
    print(" ", summary_md_path)
    print(" ", fig_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
