#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import pickle
import time
import tomllib
from argparse import Namespace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from modules.a2c import save_jax_params
from modules.environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from modules.parallel_a2c import A2CHyperParams, ParallelA2CResult, ParallelJaxBatchMaskA2C
from modules.run_dirs import create_timestamped_run_dir, write_run_metadata
from modules.simulation import JaxSimulator

jax.config.update('jax_compiler_enable_remat_pass', False)

EVAL_SUMMARY_NAME = "eval_summary_jax.json"
TRAINING_DATA_NAME = "data_training_jax.p"

DEFAULT_META = {
    "result_path": "./results",
}

DEFAULT_PARAMS = {
    "jobid": "",
    "seed": 15,
    "hidden_size": 128,
    "num_nodes": 15,
    "beta_move": 40.0,
    "eps_move": 0.0,
    "learning_rate": 1.0,
    "lamda_backup": 1.0,
    "wm_decay": 1.0,
    "t_max": 100,
    "cost": 0.01,
    "scale_factor": 1 / 8,
    "shuffle_nodes": True,
    "canonicalize": False,
    "use_recency_obs": False,
    "mask_fixation": True,
    "num_episodes": 16_000_000,
    "eval_episodes": 102_400,
    "lr": 5e-4,
    "batch_size": 64,
    "max_grad_norm": 1.0,
    "gamma": 1.0,
    "lamda": 0.9,
    "beta_v": 0.05,
    "beta_e": 0.05,
    "beta_e_init": 0.05,
    "beta_e_final": 0.001,
    "print_frequency": 100,
    "checkpoint_frequency": -1,
    "log_full_metrics": True,
}

ENV_SWEEP_KEYS = {
    "beta_move",
    "eps_move",
    "learning_rate",
    "lamda_backup",
    "wm_decay",
    "cost",
    "scale_factor",
    "shuffle_nodes",
}
TRAIN_SWEEP_KEYS = {
    "lr",
    "gamma",
    "lamda",
    "beta_v",
    "beta_e_init",
    "beta_e_final",
    "max_grad_norm",
}
SWEEP_KEYS = ENV_SWEEP_KEYS | TRAIN_SWEEP_KEYS | {"seed"}
SHAPE_KEYS = {
    "num_nodes",
    "hidden_size",
    "batch_size",
    "t_max",
    "num_episodes",
    "eval_episodes",
    "canonicalize",
    "use_recency_obs",
}


def _load_config(path: str) -> tuple[Path, dict]:
    config_path = Path(path)
    if not config_path.exists() and config_path.suffix != ".toml":
        candidate = Path("config") / f"{path}.toml"
        if candidate.exists():
            config_path = candidate
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with config_path.open("rb") as file:
        return config_path, tomllib.load(file)


def _is_list(value) -> bool:
    return isinstance(value, list)


def _validate_params(params: dict) -> None:
    for key, value in params.items():
        if not _is_list(value):
            continue
        if key in SHAPE_KEYS:
            raise ValueError(
                f"params.{key} cannot be an array in train_parallel_a2c.py because it changes compiled shapes."
            )
        if key not in SWEEP_KEYS:
            raise ValueError(f"params.{key} is not a supported parallel sweep parameter.")
        if len(value) == 0:
            raise ValueError(f"params.{key} must not be an empty array.")


def expand_sweep(params: dict) -> tuple[dict, list[dict], list[int], list[str]]:
    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    _validate_params(merged)

    seeds_raw = merged.pop("seed")
    seeds = seeds_raw if _is_list(seeds_raw) else [seeds_raw]
    seeds = [int(seed) for seed in seeds]

    sweep_items = [
        (key, value)
        for key, value in merged.items()
        if _is_list(value)
    ]
    fixed = {
        key: value
        for key, value in merged.items()
        if not _is_list(value)
    }

    if not sweep_items:
        combos = [dict(fixed)]
        return fixed, combos, seeds, []

    varied_keys = [key for key, _ in sweep_items]
    combos: list[dict] = []
    for values in itertools.product(*(value for _, value in sweep_items)):
        combo = dict(fixed)
        combo.update(dict(zip(varied_keys, values)))
        combos.append(combo)
    return fixed, combos, seeds, varied_keys


def build_hypers(combos: list[dict]) -> A2CHyperParams:
    def array(key: str, dtype=jnp.float32):
        return jnp.asarray([combo[key] for combo in combos], dtype=dtype)

    env = JaxDecisionTreeParams(
        beta_move=array("beta_move"),
        eps_move=array("eps_move"),
        learning_rate=array("learning_rate"),
        lamda_backup=array("lamda_backup"),
        wm_decay=array("wm_decay"),
        cost=array("cost"),
        scale_factor=array("scale_factor"),
        shuffle_nodes=array("shuffle_nodes", dtype=np.bool_),
    )
    return A2CHyperParams(
        env=env,
        lr=array("lr"),
        gamma=array("gamma"),
        lamda=array("lamda"),
        beta_v=array("beta_v"),
        beta_e_init=array("beta_e_init"),
        beta_e_final=array("beta_e_final"),
        max_grad_norm=array("max_grad_norm"),
    )


def _env_from_args(args: dict) -> JaxDecisionTreeEnv:
    return JaxDecisionTreeEnv(
        num_nodes=args["num_nodes"],
        beta_move=args["beta_move"],
        eps_move=args["eps_move"],
        learning_rate=args["learning_rate"],
        lamda_backup=args["lamda_backup"],
        wm_decay=args["wm_decay"],
        t_max=args["t_max"],
        cost=args["cost"],
        scale_factor=args["scale_factor"],
        shuffle_nodes=args["shuffle_nodes"],
        canonicalize=args["canonicalize"],
        use_recency_obs=args["use_recency_obs"],
    )


def _env_cache_key(args: dict) -> tuple:
    keys = (
        "num_nodes",
        "beta_move",
        "eps_move",
        "learning_rate",
        "lamda_backup",
        "wm_decay",
        "t_max",
        "cost",
        "scale_factor",
        "shuffle_nodes",
        "canonicalize",
        "use_recency_obs",
    )
    return tuple(args[key] for key in keys)


def _metric_data(metrics, hyper_index: int, seed_index: int, elapsed_seconds: float) -> dict[str, list[float]]:
    metric_slice = jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x[hyper_index, seed_index])),
        metrics,
    )
    num_updates = int(metric_slice.loss.shape[0])
    step_time = elapsed_seconds / max(num_updates, 1)
    cumulative = np.linspace(step_time, elapsed_seconds, num_updates, dtype=np.float64)
    return {
        "loss": metric_slice.loss.astype(float).tolist(),
        "policy_loss": metric_slice.policy_loss.astype(float).tolist(),
        "value_loss": metric_slice.value_loss.astype(float).tolist(),
        "entropy_loss": metric_slice.entropy_loss.astype(float).tolist(),
        "episode_length": metric_slice.episode_length.astype(float).tolist(),
        "episode_reward": metric_slice.episode_reward.astype(float).tolist(),
        "grad_norm": metric_slice.grad_norm.astype(float).tolist(),
        "param_norm": metric_slice.param_norm.astype(float).tolist(),
        "step_time_s": [float(step_time)] * num_updates,
        "cumulative_time_s": cumulative.tolist(),
    }


def _state_slice(states, hyper_index: int, seed_index: int):
    return jax.tree_util.tree_map(
        lambda x: jax.device_get(x[hyper_index, seed_index]),
        states,
    )


def _run_jobid(base_jobid: str, hyper_index: int, seed: int) -> str:
    suffix = f"h{hyper_index}_s{seed}"
    if str(base_jobid).strip():
        return f"{base_jobid}_{suffix}"
    return suffix


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def _entropy_schedule(hypers: A2CHyperParams, num_updates: int) -> jax.Array:
    progress = jnp.linspace(0.0, 1.0, num_updates, dtype=jnp.float32)
    return (
        hypers.beta_e_init[:, None]
        + (hypers.beta_e_final - hypers.beta_e_init)[:, None] * progress[None, :]
    ).astype(jnp.float32)


def train_with_progress(
    trainer: ParallelJaxBatchMaskA2C,
    hypers: A2CHyperParams,
    seeds: list[int],
    *,
    num_updates: int,
    print_frequency: int,
) -> tuple[ParallelA2CResult, float]:
    if print_frequency <= 0:
        start = time.time()
        result = jax.block_until_ready(trainer.train_sweep(hypers, seeds))
        return result, time.time() - start

    schedule = _entropy_schedule(hypers, num_updates)
    states = jax.block_until_ready(trainer.init_sweep_states(hypers, seeds))
    warmup_updates = min(print_frequency, num_updates)
    trainer.compile_train_sweep_chunk(
        states,
        hypers,
        schedule[:, :warmup_updates],
    )

    start = time.time()
    metrics_chunks = []

    for update_start in range(0, num_updates, print_frequency):
        update_end = min(update_start + print_frequency, num_updates)
        chunk = schedule[:, update_start:update_end]
        result = jax.block_until_ready(trainer.train_sweep_chunk(states, hypers, chunk))
        states = result.states
        metrics_chunks.append(result.metrics)

        elapsed_seconds = time.time() - start
        updates_done = update_end
        updates_per_second = updates_done / elapsed_seconds
        eta_seconds = (num_updates - updates_done) / updates_per_second
        print(
            "parallel_train_progress "
            f"updates={updates_done}/{num_updates} "
            f"elapsed={_format_duration(elapsed_seconds)} "
            f"eta={_format_duration(eta_seconds)} "
            f"updates_per_second={updates_per_second:.3f}",
            flush=True,
        )

    metrics = jax.tree_util.tree_map(
        lambda *chunks: jnp.concatenate(chunks, axis=2),
        *metrics_chunks,
    )
    return ParallelA2CResult(states=states, metrics=metrics), time.time() - start


def save_results(
    result,
    combos: list[dict],
    seeds: list[int],
    *,
    path: str,
    experiment: str,
    config_path: Path,
    varied_keys: list[str],
    elapsed_seconds: float,
) -> list[str]:
    run_dirs: list[str] = []
    simulators: dict[tuple, JaxSimulator] = {}
    for hyper_index, combo in enumerate(combos):
        env_key = _env_cache_key(combo)
        if env_key not in simulators:
            simulators[env_key] = JaxSimulator(_env_from_args(combo))
        simulator = simulators[env_key]

        for seed_index, seed in enumerate(seeds):
            run_args = dict(combo)
            run_args["seed"] = int(seed)
            run_args["parallel_config"] = str(config_path)
            run_args["parallel_hyper_index"] = int(hyper_index)
            run_args["parallel_seed_index"] = int(seed_index)
            run_args["parallel_varied_keys"] = list(varied_keys)

            run_dir = create_timestamped_run_dir(
                path=path,
                experiment=experiment,
                jobid=_run_jobid(str(combo.get("jobid", "")), hyper_index, seed),
            )
            write_run_metadata(run_dir=run_dir, args=Namespace(**run_args), cwd=os.getcwd())

            state = _state_slice(result.states, hyper_index, seed_index)
            data = _metric_data(result.metrics, hyper_index, seed_index, elapsed_seconds)

            with open(os.path.join(run_dir, TRAINING_DATA_NAME), "wb") as file:
                pickle.dump(data, file)
            save_jax_params(state.params, os.path.join(run_dir, "net_jax.p"))

            eval_start = time.time()
            eval_episodes = int(run_args["eval_episodes"])
            eval_stats = simulator.evaluate_policy(
                params=state.params,
                seed=int(seed),
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
                "num_updates": int(run_args["num_episodes"] / run_args["batch_size"]),
                "num_episodes": int(run_args["num_episodes"]),
            }
            with open(os.path.join(run_dir, EVAL_SUMMARY_NAME), "w") as file:
                json.dump(eval_summary, file, indent=2, sort_keys=True)

            run_dirs.append(run_dir)
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parallelized A2C TOML sweep.")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument("--experiment", help="Override experiment name. Defaults to [meta].experiment or config stem.")
    args = parser.parse_args()

    config_path, config = _load_config(args.config)
    meta = dict(DEFAULT_META)
    meta.update(config.get("meta", {}))
    params = config.get("params", {})

    fixed, combos, seeds, varied_keys = expand_sweep(params)
    output_path = args.path or str(meta["result_path"])
    experiment = args.experiment or str(meta.get("experiment", config_path.stem))

    num_updates = int(fixed["num_episodes"] / fixed["batch_size"])
    env = _env_from_args(combos[0])
    trainer = ParallelJaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        batch_size=fixed["batch_size"],
        num_updates=num_updates,
    )
    hypers = build_hypers(combos)

    devices = ", ".join(
        f"{device.platform}:{device.device_kind}"
        for device in jax.local_devices()
    )
    print(f"jax_backend={jax.default_backend()} jax_devices=[{devices}]")
    print(
        "parallel_run_config "
        f"hyper_combos={len(combos)} "
        f"seeds={len(seeds)} "
        f"num_updates={num_updates} "
        f"batch_size={fixed['batch_size']} "
        f"t_max={fixed['t_max']} "
        f"varied_keys={','.join(varied_keys)}"
    )

    result, elapsed_seconds = train_with_progress(
        trainer,
        hypers,
        seeds,
        num_updates=num_updates,
        print_frequency=int(fixed["print_frequency"]),
    )
    print(f"parallel_train_elapsed_seconds={elapsed_seconds:.3f}")

    run_dirs = save_results(
        result,
        combos,
        seeds,
        path=output_path,
        experiment=experiment,
        config_path=config_path,
        varied_keys=varied_keys,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"saved_runs={len(run_dirs)}")
    for run_dir in run_dirs:
        print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
