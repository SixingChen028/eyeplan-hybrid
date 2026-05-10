from __future__ import annotations

import json
import os
import pickle
import time
from argparse import Namespace
from pathlib import Path

import jax
import numpy as np

from modules.a2c import save_jax_params
from modules.config import ENV_DYNAMIC_PARAM_KEYS, ENV_STATIC_PARAM_KEYS
from modules.environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from modules.results_layout import create_run_dir, write_run_metadata
from modules.simulation import JaxSimulator

EVAL_SUMMARY_NAME = "eval_summary_jax.json"
TRAINING_DATA_NAME = "data_training_jax.p"


def env_from_args(args: dict) -> JaxDecisionTreeEnv:
    return JaxDecisionTreeEnv(
        num_nodes=args["num_nodes"],
        t_max=args["t_max"],
        scale_factor=args["scale_factor"],
        shuffle_nodes=args["shuffle_nodes"],
        use_recency_obs=bool(args["use_recency_obs"]),
        use_best_open_value_obs=bool(args["use_best_open_value_obs"]),
        use_best_terminal_value_obs=bool(args["use_best_terminal_value_obs"]),
        wm_backup=bool(args["wm_backup"]),
        point_set=args["point_set"],
    )


def env_params_from_args(env: JaxDecisionTreeEnv, args: dict) -> JaxDecisionTreeParams:
    return env.make_params(
        **{key: args[key] for key in ENV_DYNAMIC_PARAM_KEYS},
    )


def env_cache_key(args: dict) -> tuple:
    return tuple(args[key] for key in (*ENV_STATIC_PARAM_KEYS, *ENV_DYNAMIC_PARAM_KEYS))


def metric_data(
    metrics,
    run_index: int,
    elapsed_seconds: float,
) -> dict[str, list[float]]:
    metric_slice = jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x[run_index])),
        metrics,
    )
    num_updates = int(metric_slice.loss.shape[0])
    step_time = elapsed_seconds / max(num_updates, 1)
    cumulative = np.linspace(step_time, elapsed_seconds, num_updates, dtype=np.float64)
    data = {
        "loss": metric_slice.loss.astype(float).tolist(),
        "policy_loss": metric_slice.policy_loss.astype(float).tolist(),
        "value_loss": metric_slice.value_loss.astype(float).tolist(),
        "entropy_loss": metric_slice.entropy_loss.astype(float).tolist(),
        "episode_length": metric_slice.episode_length.astype(float).tolist(),
        "episode_reward": metric_slice.episode_reward.astype(float).tolist(),
        "step_time_s": [float(step_time)] * num_updates,
        "cumulative_time_s": cumulative.tolist(),
    }
    data["grad_norm"] = metric_slice.grad_norm.astype(float).tolist()
    data["param_norm"] = metric_slice.param_norm.astype(float).tolist()
    return data


def state_slice(states, run_index: int):
    return jax.tree_util.tree_map(
        lambda x: jax.device_get(x[run_index]),
        states,
    )


def slug_value(value) -> str:
    text = str(value).strip()
    return "".join(char if char.isalnum() or char in {".", "-"} else "-" for char in text)


def run_prefix(run: dict, varied_keys: list[str]) -> str:
    param_parts = [f"{key}{slug_value(run[key])}" for key in varied_keys]
    return "_".join(param_parts)


def log_run_dirs_preview(run_dirs: list[str]) -> None:
    if len(run_dirs) <= 10:
        for run_dir in run_dirs:
            print(run_dir, flush=True)
        return
    for run_dir in run_dirs[:5]:
        print(run_dir, flush=True)
    print("...", flush=True)
    for run_dir in run_dirs[-5:]:
        print(run_dir, flush=True)


def prepare_run_dirs(
    runs: list[dict],
    *,
    path: str,
    experiment: str,
    config_path: Path,
    varied_keys: list[str],
) -> list[str]:
    run_dirs: list[str] = []
    for run in runs:
        run_args = dict(run)
        run_args["parallel_config"] = str(config_path)
        run_args["parallel_varied_keys"] = list(varied_keys)

        run_dir = create_run_dir(
            results_root=path,
            experiment=experiment,
            prefix=run_prefix(run, varied_keys),
        )
        write_run_metadata(run_dir=run_dir, args=Namespace(**run_args), cwd=os.getcwd())
        run_dirs.append(run_dir)
    return run_dirs


def save_results(
    result,
    runs: list[dict],
    run_dirs: list[str] | None = None,
    *,
    path: str | None = None,
    experiment: str | None = None,
    config_path: Path | None = None,
    varied_keys: list[str] | None = None,
    elapsed_seconds: float,
) -> list[str]:
    if run_dirs is None:
        if path is None or experiment is None or config_path is None:
            raise ValueError("When run_dirs is not provided, path/experiment/config_path are required.")
        run_dirs = prepare_run_dirs(
            runs,
            path=path,
            experiment=experiment,
            config_path=config_path,
            varied_keys=[] if varied_keys is None else varied_keys,
        )
    simulators: dict[tuple, JaxSimulator] = {}
    for run_index, run in enumerate(runs):
        env_key = env_cache_key(run)
        if env_key not in simulators:
            env = env_from_args(run)
            simulators[env_key] = JaxSimulator(env, env_params=env_params_from_args(env, run))
        simulator = simulators[env_key]
        run_dir = run_dirs[run_index]

        state = state_slice(result.states, run_index)
        data = metric_data(
            result.metrics,
            run_index,
            elapsed_seconds,
        )

        with open(os.path.join(run_dir, TRAINING_DATA_NAME), "wb") as file:
            pickle.dump(data, file)
        save_jax_params(state.params, os.path.join(run_dir, "net_jax.p"))

        eval_start = time.time()
        eval_episodes = int(run["eval_episodes"])
        eval_stats = simulator.evaluate_policy(
            params=state.params,
            seed=int(run["seed"]),
            num_trials=eval_episodes,
            greedy=True,
            batch_size=eval_episodes,
        )
        eval_elapsed_seconds = time.time() - eval_start
        eval_summary = {
            "num_trials": int(eval_stats["num_trials"]),
            "reward_mean": float(eval_stats["reward_mean"]),
            "reward_sd": float(eval_stats["reward_sd"]),
            "reward_no_cost_mean": float(eval_stats["reward_no_cost_mean"]),
            "reward_no_cost_sd": float(eval_stats["reward_no_cost_sd"]),
            "n_steps_mean": float(eval_stats["n_steps_mean"]),
            "n_steps_sd": float(eval_stats["n_steps_sd"]),
            "train_elapsed_seconds": float(elapsed_seconds),
            "eval_elapsed_seconds": float(eval_elapsed_seconds),
            "num_updates": int(run["num_updates"]),
        }
        with open(os.path.join(run_dir, EVAL_SUMMARY_NAME), "w") as file:
            json.dump(eval_summary, file, indent=2, sort_keys=True)

        log_path = os.path.join(run_dir, "training.log")
        with open(log_path, "a") as file:
            file.write("\n")
            file.write(
                "run_summary "
                f"run_index={run_index} "
                f"seed={int(run['seed'])} "
                f"train_elapsed_seconds={elapsed_seconds:.3f} "
                f"eval_elapsed_seconds={eval_elapsed_seconds:.3f}\n"
            )
            file.write(
                "eval_summary "
                f"episodes={eval_summary['num_trials']} "
                f"reward_mean={eval_summary['reward_mean']:.6f} "
                f"reward_sd={eval_summary['reward_sd']:.6f} "
                f"reward_no_cost_mean={eval_summary['reward_no_cost_mean']:.6f} "
                f"reward_no_cost_sd={eval_summary['reward_no_cost_sd']:.6f} "
                f"n_steps_mean={eval_summary['n_steps_mean']:.3f} "
                f"n_steps_sd={eval_summary['n_steps_sd']:.3f}\n"
            )
            file.write(f"training_log={log_path}\n")
    return run_dirs
