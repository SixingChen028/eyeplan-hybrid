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


def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")

    return "\n".join([header, divider] + rows)


def _plot_learning_curves(
    training_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    optimal_scaled: float,
    output_path: str,
):
    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(
        training_df["episodes"],
        training_df["episode_reward_ma"],
        color="#1f77b4",
        linewidth=2,
        label="network train reward (MA)",
    )

    for _, row in baseline_df.iterrows():
        if pd.isna(row["mean_episode_reward"]):
            continue
        ax1.axhline(
            row["mean_episode_reward"],
            linestyle="--",
            linewidth=1.4,
            label=row["policy"],
        )

    ax1.set_title("Reward With Cost")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Episode reward")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        training_df["episodes"],
        training_df["episode_no_cost_proxy_ma"],
        color="#2ca02c",
        linewidth=2,
        label="network no-cost proxy (MA)",
    )

    for _, row in baseline_df.iterrows():
        if pd.isna(row["mean_no_cost_reward_scaled"]):
            continue
        ax2.axhline(
            row["mean_no_cost_reward_scaled"],
            linestyle="--",
            linewidth=1.4,
            label=row["policy"],
        )

    ax2.axhline(
        optimal_scaled,
        color="black",
        linestyle="-",
        linewidth=2,
        label="optimal path expected (no-cost)",
    )

    ax2.set_title("No-Cost Path Reward")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Scaled reward")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "jax_policy_compare_7node_100k"))

    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=15)

    parser.add_argument("--beta_move", type=float, default=4.0)
    parser.add_argument("--eps_move", type=float, default=0.02)
    parser.add_argument("--learning_rate", type=float, default=0.2)
    parser.add_argument("--wm_decay", type=float, default=0.8)
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--cost", type=float, default=0.01)
    parser.add_argument("--scale_factor", type=float, default=1 / 8)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--beta_v", type=float, default=0.05)
    parser.add_argument("--beta_e", type=float, default=0.05)
    parser.add_argument("--beta_e_init", type=float, default=0.05)
    parser.add_argument("--beta_e_final", type=float, default=0.015)

    parser.add_argument("--eval_trials", type=int, default=12000)
    parser.add_argument("--ma_window", type=int, default=40)

    config = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)

    env = JaxDecisionTreeEnv(
        num_nodes=config.num_nodes,
        beta_move=config.beta_move,
        eps_move=config.eps_move,
        learning_rate=config.learning_rate,
        wm_decay=config.wm_decay,
        t_max=config.t_max,
        cost=config.cost,
        scale_factor=config.scale_factor,
        shuffle_nodes=True,
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=config.hidden_size,
        batch_size=config.batch_size,
        lr=config.lr,
        max_grad_norm=config.max_grad_norm,
        gamma=config.gamma,
        lamda=config.lamda,
        beta_v=config.beta_v,
        beta_e=config.beta_e,
    )

    state = trainer.init_state(seed=config.seed)

    num_updates = int(config.num_episodes / config.batch_size)
    entropy_schedule = np.linspace(
        config.beta_e_init,
        config.beta_e_final,
        num_updates,
        dtype=np.float32,
    )

    state, training_data = trainer.train(
        state=state,
        num_updates=num_updates,
        entropy_schedule=entropy_schedule,
    )

    save_jax_params(state.params, os.path.join(config.output_dir, "jax_params_100k.p"))
    with open(os.path.join(config.output_dir, "training_data_100k.pkl"), "wb") as file:
        pickle.dump(training_data, file)

    training_df = pd.DataFrame(
        {
            "update": np.arange(1, num_updates + 1),
            "episodes": np.arange(1, num_updates + 1) * config.batch_size,
            "episode_reward": np.asarray(training_data["episode_reward"], dtype=np.float64),
            "episode_length": np.asarray(training_data["episode_length"], dtype=np.float64),
            "loss": np.asarray(training_data["loss"], dtype=np.float64),
            "step_time_s": np.asarray(training_data["step_time_s"], dtype=np.float64),
            "cumulative_time_s": np.asarray(training_data["cumulative_time_s"], dtype=np.float64),
        }
    )

    training_df["episode_reward_ma"] = (
        training_df["episode_reward"].rolling(config.ma_window, min_periods=1).mean()
    )

    no_cost_proxy = training_df["episode_reward"] + config.cost * np.maximum(training_df["episode_length"] - 1.0, 0.0)
    training_df["episode_no_cost_proxy"] = no_cost_proxy
    training_df["episode_no_cost_proxy_ma"] = (
        training_df["episode_no_cost_proxy"].rolling(config.ma_window, min_periods=1).mean()
    )

    training_df.to_csv(os.path.join(config.output_dir, "network_learning_curve_100k.csv"), index=False)

    key = jax.random.PRNGKey(config.seed + 100)
    reset_keys = jax.random.split(key, config.eval_trials)

    baseline_names = [
        "depth1_then_terminate",
        "visit_all_then_bestg_then_parent_chain",
        "immediate_move",
        "best_depth1_then_move",
    ]

    baseline_stats, optimal_scaled, optimal_raw = evaluate_baseline_policies(
        env=env,
        policy_names=baseline_names,
        reset_keys=reset_keys,
    )

    network_stats = evaluate_network_greedy(
        env=env,
        params=state.params,
        reset_keys=reset_keys,
    )

    rows = []
    for stats in baseline_stats:
        rows.append(
            {
                "policy": stats.name,
                "mean_episode_reward": round(stats.mean_episode_reward, 6),
                "mean_no_cost_reward_scaled": round(stats.mean_no_cost_reward_scaled, 6),
                "mean_no_cost_reward_raw": round(stats.mean_no_cost_reward_raw, 6),
                "mean_episode_length": round(stats.mean_episode_length, 4),
            }
        )

    rows.append(
        {
            "policy": network_stats.name,
            "mean_episode_reward": round(network_stats.mean_episode_reward, 6),
            "mean_no_cost_reward_scaled": round(network_stats.mean_no_cost_reward_scaled, 6),
            "mean_no_cost_reward_raw": round(network_stats.mean_no_cost_reward_raw, 6),
            "mean_episode_length": round(network_stats.mean_episode_length, 4),
        }
    )

    rows.append(
        {
            "policy": "optimal_path_expected_no_cost",
            "mean_episode_reward": np.nan,
            "mean_no_cost_reward_scaled": round(optimal_scaled, 6),
            "mean_no_cost_reward_raw": round(optimal_raw, 6),
            "mean_episode_length": np.nan,
        }
    )

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(os.path.join(config.output_dir, "policy_comparison_table.csv"), index=False)

    markdown = _to_markdown_table(baseline_df)
    with open(os.path.join(config.output_dir, "policy_comparison_table.md"), "w") as file:
        file.write(markdown + "\n")

    _plot_learning_curves(
        training_df=training_df,
        baseline_df=baseline_df,
        optimal_scaled=optimal_scaled,
        output_path=os.path.join(config.output_dir, "policy_vs_learning_curves.png"),
    )

    print("Saved outputs to:", config.output_dir)
    print(baseline_df)


if __name__ == "__main__":
    main()
