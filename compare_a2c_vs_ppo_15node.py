import argparse
import os
import pickle

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.jax_a2c import JaxBatchMaskA2C, save_jax_params
from modules.jax_baselines import evaluate_baseline_policies, evaluate_network_greedy
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_ppo import JaxBatchMaskPPO


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _make_env(args: argparse.Namespace) -> JaxDecisionTreeEnv:
    return JaxDecisionTreeEnv(
        num_nodes=args.num_nodes,
        beta_move=args.beta_move,
        eps_move=args.eps_move,
        learning_rate=args.learning_rate,
        wm_decay=args.wm_decay,
        t_max=args.t_max,
        cost=args.cost,
        scale_factor=args.scale_factor,
        shuffle_nodes=True,
    )


def _logs_to_df(logs: dict, algo: str, batch_size: int) -> pd.DataFrame:
    n = len(logs["episode_reward"])
    df = pd.DataFrame(
        {
            "algo": [algo] * n,
            "update": np.arange(1, n + 1),
            "episodes": np.arange(1, n + 1) * batch_size,
            "episode_reward": np.asarray(logs["episode_reward"], dtype=np.float64),
            "episode_length": np.asarray(logs["episode_length"], dtype=np.float64),
            "loss": np.asarray(logs["loss"], dtype=np.float64),
            "step_time_s": np.asarray(logs["step_time_s"], dtype=np.float64),
            "cumulative_time_s": np.asarray(logs["cumulative_time_s"], dtype=np.float64),
        }
    )
    return df


def _plot_curves(df: pd.DataFrame, out_path: str, ma_window: int):
    plt.figure(figsize=(12, 5))

    ax = plt.subplot(1, 2, 1)
    for algo, group in df.groupby("algo"):
        g = group.sort_values("episodes")
        ax.plot(g["episodes"], g["episode_reward"], alpha=0.2)
        ax.plot(g["episodes"], g["episode_reward_ma"], linewidth=2, label=f"{algo} reward_ma")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = plt.subplot(1, 2, 2)
    for algo, group in df.groupby("algo"):
        g = group.sort_values("episodes")
        ax.plot(g["episodes"], g["cumulative_time_s"], linewidth=2, label=f"{algo}")
    ax.set_title("Cumulative Runtime")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Seconds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(f"A2C vs PPO (ma_window={ma_window})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "a2c_vs_ppo_15node"))
    parser.add_argument("--seed", type=int, default=15)

    parser.add_argument("--num_nodes", type=int, default=15)
    parser.add_argument("--t_max", type=int, default=61)
    parser.add_argument("--cost", type=float, default=0.0)
    parser.add_argument("--beta_move", type=float, default=100.0)
    parser.add_argument("--eps_move", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--wm_decay", type=float, default=1.0)
    parser.add_argument("--scale_factor", type=float, default=1 / 8)

    parser.add_argument("--num_episodes", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--beta_v", type=float, default=0.05)
    parser.add_argument("--beta_e", type=float, default=0.02)
    parser.add_argument("--beta_e_init", type=float, default=0.02)
    parser.add_argument("--beta_e_final", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)

    parser.add_argument("--ma_window", type=int, default=100)
    parser.add_argument("--eval_trials", type=int, default=3000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_updates = int(args.num_episodes / args.batch_size)
    entropy_schedule = np.linspace(args.beta_e_init, args.beta_e_final, num_updates, dtype=np.float32)

    env_a2c = _make_env(args)
    a2c = JaxBatchMaskA2C(
        env=env_a2c,
        feature_size=env_a2c.observation_shape[0],
        action_size=env_a2c.action_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        lamda=args.lamda,
        beta_v=args.beta_v,
        beta_e=args.beta_e,
        max_grad_norm=args.max_grad_norm,
    )

    a2c_state = a2c.init_state(seed=args.seed)
    a2c_state, a2c_logs = a2c.train(a2c_state, num_updates=num_updates, entropy_schedule=entropy_schedule)

    env_ppo = _make_env(args)
    ppo = JaxBatchMaskPPO(
        env=env_ppo,
        feature_size=env_ppo.observation_shape[0],
        action_size=env_ppo.action_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        lamda=args.lamda,
        beta_v=args.beta_v,
        beta_e=args.beta_e,
        max_grad_norm=args.max_grad_norm,
        clip_eps=args.ppo_clip_eps,
        ppo_epochs=args.ppo_epochs,
        normalize_advantages=True,
    )

    ppo_state = ppo.init_state(seed=args.seed)
    ppo_state, ppo_logs = ppo.train(ppo_state, num_updates=num_updates, entropy_schedule=entropy_schedule)

    save_jax_params(a2c_state.params, os.path.join(args.output_dir, "a2c_params.p"))
    save_jax_params(ppo_state.params, os.path.join(args.output_dir, "ppo_params.p"))
    with open(os.path.join(args.output_dir, "a2c_training.pkl"), "wb") as file:
        pickle.dump(a2c_logs, file)
    with open(os.path.join(args.output_dir, "ppo_training.pkl"), "wb") as file:
        pickle.dump(ppo_logs, file)

    df = pd.concat(
        [_logs_to_df(a2c_logs, "a2c", args.batch_size), _logs_to_df(ppo_logs, "ppo", args.batch_size)],
        ignore_index=True,
    )
    df["episode_reward_ma"] = df.groupby("algo")["episode_reward"].transform(
        lambda s: s.rolling(args.ma_window, min_periods=1).mean()
    )

    eval_env = _make_env(args)
    eval_keys = jax.random.split(jax.random.PRNGKey(args.seed + 101), args.eval_trials)

    a2c_eval = evaluate_network_greedy(eval_env, a2c_state.params, eval_keys)
    ppo_eval = evaluate_network_greedy(eval_env, ppo_state.params, eval_keys)
    _, optimal_scaled, optimal_raw = evaluate_baseline_policies(eval_env, [], eval_keys)

    summary = pd.DataFrame(
        [
            {
                "algo": "a2c",
                "updates": num_updates,
                "episodes": args.num_episodes,
                "final_reward_ma": round(float(df[df["algo"] == "a2c"]["episode_reward_ma"].iloc[-1]), 6),
                "total_train_time_s": round(float(df[df["algo"] == "a2c"]["cumulative_time_s"].iloc[-1]), 6),
                "mean_step_time_ms": round(float(df[df["algo"] == "a2c"]["step_time_s"].mean() * 1000.0), 6),
                "greedy_eval_reward": round(a2c_eval.mean_episode_reward, 6),
                "greedy_eval_no_cost_scaled": round(a2c_eval.mean_no_cost_reward_scaled, 6),
            },
            {
                "algo": "ppo",
                "updates": num_updates,
                "episodes": args.num_episodes,
                "final_reward_ma": round(float(df[df["algo"] == "ppo"]["episode_reward_ma"].iloc[-1]), 6),
                "total_train_time_s": round(float(df[df["algo"] == "ppo"]["cumulative_time_s"].iloc[-1]), 6),
                "mean_step_time_ms": round(float(df[df["algo"] == "ppo"]["step_time_s"].mean() * 1000.0), 6),
                "greedy_eval_reward": round(ppo_eval.mean_episode_reward, 6),
                "greedy_eval_no_cost_scaled": round(ppo_eval.mean_no_cost_reward_scaled, 6),
            },
            {
                "algo": "optimal_no_cost_expected",
                "updates": np.nan,
                "episodes": np.nan,
                "final_reward_ma": np.nan,
                "total_train_time_s": np.nan,
                "mean_step_time_ms": np.nan,
                "greedy_eval_reward": np.nan,
                "greedy_eval_no_cost_scaled": round(optimal_scaled, 6),
            },
        ]
    )

    curves_csv = os.path.join(args.output_dir, "a2c_vs_ppo_curves.csv")
    summary_csv = os.path.join(args.output_dir, "a2c_vs_ppo_summary.csv")
    summary_md = os.path.join(args.output_dir, "a2c_vs_ppo_summary.md")
    fig_path = os.path.join(args.output_dir, "a2c_vs_ppo_curves.png")

    df.to_csv(curves_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    with open(summary_md, "w") as file:
        file.write(_to_markdown_table(summary) + "\n")

    _plot_curves(df, fig_path, ma_window=args.ma_window)

    print("Saved outputs to:", args.output_dir)
    print(summary.to_string(index=False))
    print("optimal_no_cost_raw_expected", round(optimal_raw, 6))


if __name__ == "__main__":
    main()
