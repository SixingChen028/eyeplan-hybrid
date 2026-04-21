import argparse
import os
import pickle
import random
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from modules.a2c import FeedForwardBatchMaskA2C
from modules.environment import DecisionTreeEnv
from modules.jax_a2c import JaxBatchMaskA2C, save_jax_params
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_simulation import JaxSimulator
from modules.network import SharedFeedForwardActorCriticPolicy
from modules.simulation import simulate


def _make_legacy_env_factory(config, seed):
    def _factory():
        return DecisionTreeEnv(
            num_nodes=config.num_nodes,
            beta_move=config.beta_move,
            eps_move=config.eps_move,
            learning_rate=config.learning_rate,
            wm_decay=config.wm_decay,
            t_max=config.t_max,
            cost=config.cost,
            scale_factor=config.scale_factor,
            shuffle_nodes=config.shuffle_nodes,
            mask_fixation=config.mask_fixation,
            seed=seed,
        )

    return _factory


def _run_legacy_training(config, output_dir):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    num_updates = int(config.num_episodes / config.batch_size)
    entropy_schedule = np.linspace(
        config.beta_e_init,
        config.beta_e_final,
        num_updates,
        dtype=np.float32,
    )

    seed_rng = np.random.default_rng(config.seed)
    env_seeds = seed_rng.integers(0, 10_000_000, size=config.batch_size).tolist()
    env = gym.vector.SyncVectorEnv(
        [_make_legacy_env_factory(config, seed=s) for s in env_seeds]
    )

    net = SharedFeedForwardActorCriticPolicy(
        feature_size=env.single_observation_space.shape[0],
        action_size=env.single_action_space.n,
        hidden_size=config.hidden_size,
    )

    trainer = FeedForwardBatchMaskA2C(
        net=net,
        env=env,
        lr=config.lr,
        batch_size=config.batch_size,
        max_grad_norm=config.max_grad_norm,
        gamma=config.gamma,
        lamda=config.lamda,
        beta_v=config.beta_v,
        beta_e=config.beta_e,
        entropy_schedule=entropy_schedule,
    )

    logs = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy_loss": [],
        "episode_length": [],
        "episode_reward": [],
        "step_time_s": [],
        "cumulative_time_s": [],
    }

    start_time = time.perf_counter()
    for step in range(num_updates):
        step_start = time.perf_counter()
        data_step = trainer.train_one_episode()
        step_time = time.perf_counter() - step_start

        logs["loss"].append(float(data_step["loss"]))
        logs["policy_loss"].append(float(data_step["policy_loss"]))
        logs["value_loss"].append(float(data_step["value_loss"]))
        logs["entropy_loss"].append(float(data_step["entropy_loss"]))
        logs["episode_length"].append(float(data_step["episode_length"]))
        logs["episode_reward"].append(float(data_step["episode_reward"]))
        logs["step_time_s"].append(step_time)
        logs["cumulative_time_s"].append(time.perf_counter() - start_time)

        trainer.update_entropy_coef(step)

    with open(os.path.join(output_dir, "legacy_training.pkl"), "wb") as file:
        pickle.dump(logs, file)
    torch.save(net, os.path.join(output_dir, "legacy_net.pth"))

    env.close()

    return net, logs


def _run_jax_training(config, output_dir):
    num_updates = int(config.num_episodes / config.batch_size)
    entropy_schedule = np.linspace(
        config.beta_e_init,
        config.beta_e_final,
        num_updates,
        dtype=np.float32,
    )

    env = JaxDecisionTreeEnv(
        num_nodes=config.num_nodes,
        beta_move=config.beta_move,
        eps_move=config.eps_move,
        learning_rate=config.learning_rate,
        wm_decay=config.wm_decay,
        t_max=config.t_max,
        cost=config.cost,
        scale_factor=config.scale_factor,
        shuffle_nodes=config.shuffle_nodes,
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
    state, logs = trainer.train(
        state=state,
        num_updates=num_updates,
        entropy_schedule=entropy_schedule,
    )

    with open(os.path.join(output_dir, "jax_training.pkl"), "wb") as file:
        pickle.dump(logs, file)
    save_jax_params(state.params, os.path.join(output_dir, "jax_params.p"))

    return state.params, logs


def _mean_choice_reward(sim_data):
    rewards = []
    for points, choice_seq in zip(sim_data["points"], sim_data["choice_seqs"]):
        rewards.append(sum(points[node] for node in choice_seq))
    if not rewards:
        return 0.0
    return float(np.mean(rewards))


def _mean_action_length(sim_data):
    lengths = [len(seq) for seq in sim_data["action_seqs"]]
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def _run_legacy_simulation_benchmark(config, net):
    env = DecisionTreeEnv(
        num_nodes=config.num_nodes,
        beta_move=config.beta_move,
        eps_move=config.eps_move,
        learning_rate=config.learning_rate,
        wm_decay=config.wm_decay,
        t_max=config.t_max,
        cost=config.cost,
        scale_factor=config.scale_factor,
        shuffle_nodes=config.shuffle_nodes,
        mask_fixation=config.mask_fixation,
        seed=config.seed,
    )

    start = time.perf_counter()
    sim_data = simulate(
        net=net,
        env=env,
        num_trials=config.sim_trials,
        greedy=False,
    )
    duration = time.perf_counter() - start

    return {
        "simulation_time_s": duration,
        "simulation_trials_per_s": config.sim_trials / max(duration, 1e-9),
        "simulation_mean_choice_reward": _mean_choice_reward(sim_data),
        "simulation_mean_action_length": _mean_action_length(sim_data),
    }


def _run_jax_simulation_benchmark(config, params):
    env = JaxDecisionTreeEnv(
        num_nodes=config.num_nodes,
        beta_move=config.beta_move,
        eps_move=config.eps_move,
        learning_rate=config.learning_rate,
        wm_decay=config.wm_decay,
        t_max=config.t_max,
        cost=config.cost,
        scale_factor=config.scale_factor,
        shuffle_nodes=config.shuffle_nodes,
    )

    simulator = JaxSimulator(env)

    start = time.perf_counter()
    sim_data = simulator.simulate(
        params=params,
        seed=config.seed,
        num_trials=config.sim_trials,
        greedy=False,
    )
    duration = time.perf_counter() - start

    return {
        "simulation_time_s": duration,
        "simulation_trials_per_s": config.sim_trials / max(duration, 1e-9),
        "simulation_mean_choice_reward": _mean_choice_reward(sim_data),
        "simulation_mean_action_length": _mean_action_length(sim_data),
    }


def _logs_to_dataframe(logs, backend, batch_size):
    num_updates = len(logs["loss"])
    rows = []
    for idx in range(num_updates):
        rows.append(
            {
                "backend": backend,
                "update": idx + 1,
                "episodes": (idx + 1) * batch_size,
                "loss": logs["loss"][idx],
                "policy_loss": logs["policy_loss"][idx],
                "value_loss": logs["value_loss"][idx],
                "entropy_loss": logs["entropy_loss"][idx],
                "episode_length": logs["episode_length"][idx],
                "episode_reward": logs["episode_reward"][idx],
                "step_time_s": logs["step_time_s"][idx],
                "cumulative_time_s": logs["cumulative_time_s"][idx],
            }
        )
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, divider] + rows)


def _plot_curves(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    for backend, backend_df in df.groupby("backend"):
        plt.plot(
            backend_df["episodes"],
            backend_df["episode_reward_ma"],
            label=backend,
            linewidth=2,
        )
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward (MA)")
    plt.title("Performance vs Training Steps")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    for backend, backend_df in df.groupby("backend"):
        plt.plot(
            backend_df["episodes"],
            backend_df["cumulative_time_s"],
            label=backend,
            linewidth=2,
        )
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Runtime (s)")
    plt.title("Runtime vs Training Steps")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_curves.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "benchmark_7node"))
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--num_episodes", type=int, default=20000)
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
    parser.add_argument("--shuffle_nodes", dest="shuffle_nodes", action="store_true")
    parser.add_argument("--no-shuffle_nodes", dest="shuffle_nodes", action="store_false")
    parser.set_defaults(shuffle_nodes=True)
    parser.add_argument("--mask_fixation", dest="mask_fixation", action="store_true")
    parser.add_argument("--no-mask_fixation", dest="mask_fixation", action="store_false")
    parser.set_defaults(mask_fixation=True)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--beta_v", type=float, default=0.05)
    parser.add_argument("--beta_e", type=float, default=0.05)
    parser.add_argument("--beta_e_init", type=float, default=0.05)
    parser.add_argument("--beta_e_final", type=float, default=0.015)

    parser.add_argument("--sim_trials", type=int, default=2000)
    parser.add_argument("--ma_window", type=int, default=20)

    config = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)

    legacy_net, legacy_logs = _run_legacy_training(config, config.output_dir)
    jax_params, jax_logs = _run_jax_training(config, config.output_dir)

    legacy_sim_stats = _run_legacy_simulation_benchmark(config, legacy_net)
    jax_sim_stats = _run_jax_simulation_benchmark(config, jax_params)

    df_legacy = _logs_to_dataframe(legacy_logs, "legacy_torch", config.batch_size)
    df_jax = _logs_to_dataframe(jax_logs, "jax", config.batch_size)
    df = pd.concat([df_legacy, df_jax], ignore_index=True)

    df["episode_reward_ma"] = (
        df.groupby("backend")["episode_reward"]
        .transform(lambda s: s.rolling(config.ma_window, min_periods=1).mean())
    )

    df.to_csv(os.path.join(config.output_dir, "training_curves.csv"), index=False)

    summary_rows = []
    for backend, backend_df in df.groupby("backend"):
        if backend == "legacy_torch":
            sim_stats = legacy_sim_stats
        else:
            sim_stats = jax_sim_stats

        summary_rows.append(
            {
                "backend": backend,
                "total_training_time_s": round(float(backend_df["cumulative_time_s"].iloc[-1]), 4),
                "mean_update_time_ms": round(float(backend_df["step_time_s"].mean() * 1000), 4),
                "updates_per_s": round(float(len(backend_df) / backend_df["cumulative_time_s"].iloc[-1]), 4),
                "final_reward_ma": round(float(backend_df["episode_reward_ma"].iloc[-1]), 6),
                "final_episode_length": round(float(backend_df["episode_length"].iloc[-1]), 4),
                "simulation_time_s": round(float(sim_stats["simulation_time_s"]), 4),
                "simulation_trials_per_s": round(float(sim_stats["simulation_trials_per_s"]), 4),
                "simulation_mean_choice_reward": round(float(sim_stats["simulation_mean_choice_reward"]), 6),
                "simulation_mean_action_length": round(float(sim_stats["simulation_mean_action_length"]), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(config.output_dir, "benchmark_summary.csv"), index=False)

    markdown = _to_markdown_table(summary_df)
    with open(os.path.join(config.output_dir, "benchmark_summary.md"), "w") as file:
        file.write(markdown + "\n")

    _plot_curves(df, config.output_dir)

    print("Saved benchmark outputs to:", config.output_dir)
    print(summary_df)


if __name__ == "__main__":
    main()
