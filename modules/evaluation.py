from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from modules.a2c import load_jax_params
from modules.config import DEFAULT_META, DEFAULT_PARAMS, ENV_DYNAMIC_PARAM_KEYS, ENV_STATIC_PARAM_KEYS
from modules.environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from modules.simulation import JaxSimulator

EVAL_SUMMARY_NAME = "eval_summary_jax.json"
PARAMS_NAME = "net_jax.p"


def read_metadata(run_dir: str) -> dict:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")

    with open(metadata_path, "r") as file:
        return json.load(file)


def read_metadata_args(run_dir: str) -> dict:
    metadata = read_metadata(run_dir)
    args = metadata.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"Invalid metadata args payload: {os.path.join(run_dir, 'metadata.json')}")
    return args


def require_metadata_keys(metadata_args: dict, keys: tuple[str, ...], section_name: str) -> None:
    missing = [key for key in keys if key not in metadata_args]
    if missing:
        raise ValueError(
            f"Run metadata args missing required {section_name} keys: {', '.join(sorted(missing))}"
        )


def env_from_run_args(args: dict) -> JaxDecisionTreeEnv:
    required_keys = tuple(key for key in ENV_STATIC_PARAM_KEYS if key != "wm_only")
    require_metadata_keys(args, required_keys, "environment static")

    return JaxDecisionTreeEnv(
        num_nodes=int(args["num_nodes"]),
        t_max=int(args["t_max"]),
        scale_factor=float(args["scale_factor"]),
        shuffle_nodes=bool(args["shuffle_nodes"]),
        wm_only=bool(args.get("wm_only", DEFAULT_PARAMS["wm_only"])),
        use_recency_obs=bool(args["use_recency_obs"]),
        use_best_open_value_obs=bool(args["use_best_open_value_obs"]),
        use_best_terminal_value_obs=bool(args["use_best_terminal_value_obs"]),
        use_g_values_obs=bool(args["use_g_values_obs"]),
        use_q_values_obs=bool(args["use_q_values_obs"]),
        use_n_visits_obs=bool(args["use_n_visits_obs"]),
        use_is_terminal_obs=bool(args["use_is_terminal_obs"]),
        use_time_elapsed_obs=bool(args["use_time_elapsed_obs"]),
        backup_mode=str(args["backup_mode"]),
        point_set=args["point_set"],
    )


def env_params_from_run_args(env: JaxDecisionTreeEnv, args: dict) -> JaxDecisionTreeParams:
    required_keys = tuple(key for key in ENV_DYNAMIC_PARAM_KEYS if key != "wm_neighbor_activation")
    require_metadata_keys(args, required_keys, "environment dynamic")

    return env.make_params(
        beta_move=float(args["beta_move"]),
        eps_move=float(args["eps_move"]),
        learning_rate=float(args["learning_rate"]),
        lamda_backup=float(args["lamda_backup"]),
        backup_steps=int(args["backup_steps"]),
        wm_decay=float(args["wm_decay"]),
        wm_neighbor_activation=float(args.get("wm_neighbor_activation", 1.0)),
        q_drop_rate=float(args["q_drop_rate"]),
        q_drift=float(args["q_drift"]),
        q_decay=float(args["q_decay"]),
        recency_decay=float(args["recency_decay"]),
        cost=float(args["cost"]),
    )


def resolve_params_path_from_metadata(run_dir: str, metadata: dict) -> str:
    explicit = metadata.get("model_params_file")
    if isinstance(explicit, str) and explicit.strip() != "":
        candidate = explicit.strip()
        if not os.path.isabs(candidate):
            candidate = os.path.join(run_dir, candidate)
        if os.path.exists(candidate):
            return candidate

    argv = metadata.get("argv", [])
    entrypoint = ""
    if isinstance(argv, list) and len(argv) > 0:
        entrypoint = os.path.basename(str(argv[0]))

    if entrypoint in {"train_ppo.py", "train_jax_ppo.py"}:
        preferred = os.path.join(run_dir, "net_jax_ppo.p")
    else:
        preferred = os.path.join(run_dir, PARAMS_NAME)

    if os.path.exists(preferred):
        return preferred

    available = []
    for name in (PARAMS_NAME, "net_jax_ppo.p"):
        candidate = os.path.join(run_dir, name)
        if os.path.exists(candidate):
            available.append(candidate)

    if len(available) == 1:
        return available[0]

    raise FileNotFoundError(
        f"Unable to resolve model params file from metadata for run: {run_dir}. "
        f"preferred={preferred} available={available}"
    )


def build_simulator(args: dict) -> JaxSimulator:
    env = env_from_run_args(args)
    return JaxSimulator(env, env_params=env_params_from_run_args(env, args))


def evaluate_params(
    params: Any,
    args: dict,
    *,
    train_elapsed_seconds: float,
    eval_episodes: int | None = None,
    batch_size: int | None = None,
    simulator: JaxSimulator | None = None,
) -> dict:
    if simulator is None:
        simulator = build_simulator(args)
    default_eval_episodes = args.get("eval_episodes", DEFAULT_META["eval_episodes"])
    num_trials = int(default_eval_episodes if eval_episodes is None else eval_episodes)
    if batch_size is None:
        batch_size = num_trials

    eval_start = time.time()
    eval_stats = simulator.evaluate_policy(
        params=params,
        seed=int(args["seed"]),
        num_trials=num_trials,
        greedy=True,
        batch_size=int(batch_size),
    )
    eval_elapsed_seconds = time.time() - eval_start

    return {
        "num_trials": int(eval_stats["num_trials"]),
        "reward_mean": float(eval_stats["reward_mean"]),
        "reward_sd": float(eval_stats["reward_sd"]),
        "reward_no_cost_mean": float(eval_stats["reward_no_cost_mean"]),
        "reward_no_cost_sd": float(eval_stats["reward_no_cost_sd"]),
        "n_steps_mean": float(eval_stats["n_steps_mean"]),
        "n_steps_sd": float(eval_stats["n_steps_sd"]),
        "train_elapsed_seconds": float(train_elapsed_seconds),
        "eval_elapsed_seconds": float(eval_elapsed_seconds),
        "num_updates": int(args["num_updates"]),
    }


def write_eval_summary(run_dir: str, eval_summary: dict) -> str:
    eval_summary_path = os.path.join(run_dir, EVAL_SUMMARY_NAME)
    with open(eval_summary_path, "w") as file:
        json.dump(eval_summary, file, indent=2, sort_keys=True)
    return eval_summary_path


def read_train_elapsed_seconds_from_log(run_dir: str) -> float:
    log_path = os.path.join(run_dir, "training.log")
    if not os.path.exists(log_path):
        return 0.0

    pattern = re.compile(r"\btrain_elapsed_seconds=([0-9.]+)")
    elapsed = 0.0
    with open(log_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match is not None:
                elapsed = float(match.group(1))
    return elapsed


def evaluate_run_dir(
    run_dir: str,
    *,
    overwrite: bool = False,
    eval_episodes: int | None = None,
    batch_size: int | None = None,
) -> tuple[str, dict]:
    eval_summary_path = os.path.join(run_dir, EVAL_SUMMARY_NAME)
    if os.path.exists(eval_summary_path) and not overwrite:
        raise FileExistsError(f"Eval summary already exists: {eval_summary_path}")

    metadata = read_metadata(run_dir)
    args = read_metadata_args(run_dir)
    params_path = resolve_params_path_from_metadata(run_dir, metadata)
    params = load_jax_params(params_path)
    eval_summary = evaluate_params(
        params,
        args,
        train_elapsed_seconds=read_train_elapsed_seconds_from_log(run_dir),
        eval_episodes=eval_episodes,
        batch_size=batch_size,
    )
    return write_eval_summary(run_dir, eval_summary), eval_summary
