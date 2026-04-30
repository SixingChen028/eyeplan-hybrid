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
from modules.network import actor_critic_forward, apply_action_mask, sample_actions
from modules.simulation import (
    JaxSimulator,
    _child_array_to_dict,
    _compute_cum_points,
    _compute_depths,
    _leaf_nodes_from_children,
    _parent_array_to_dict,
    to_transformed_simulation_format,
)

jax.config.update('jax_compiler_enable_remat_pass', False)

EVAL_SUMMARY_NAME = "eval_summary_jax.json"
TRAINING_DATA_NAME = "data_training_jax.p"
SIMULATION_DATA_NAME = "data_simulation.json"

DEFAULT_META = {
    "result_path": "./results",
}

DEFAULT_PARAMS = {
    "jobid": "",
    "seed": 15,
    "network_type": "mlp",
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
    "recency_decay": "off",
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
    "recency_decay",
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
    "network_type",
}


_CONSOLE_LOG_LINES: list[str] = []


def _log(*args, **kwargs) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    line = sep.join(str(arg) for arg in args) + end
    _CONSOLE_LOG_LINES.append(line)
    print(*args, **kwargs)


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


def _resolve_recency_decay(value, wm_decay) -> float:
    enabled, auto, decay = JaxDecisionTreeEnv._parse_recency_decay(value)
    if not enabled:
        return 0.0
    if auto:
        wm_decay = float(wm_decay)
        return 0.5 if wm_decay == 1.0 else wm_decay
    return float(decay)


def _validate_params(params: dict) -> None:
    for key, value in params.items():
        if not _is_list(value):
            continue
        if len(value) == 0:
            raise ValueError(f"params.{key} must not be an empty array.")
        if key == "recency_decay":
            for item in value:
                enabled, _, _ = JaxDecisionTreeEnv._parse_recency_decay(item)
                if not enabled:
                    raise ValueError(
                        "params.recency_decay cannot include 'off' in train_parallel_a2c.py because it changes compiled shapes."
                    )
            continue
        if key in SHAPE_KEYS:
            raise ValueError(
                f"params.{key} cannot be an array in train_parallel_a2c.py because it changes compiled shapes."
            )
        if key not in SWEEP_KEYS:
            raise ValueError(f"params.{key} is not a supported parallel sweep parameter.")

    recency_decay = params.get("recency_decay", "off")
    if not _is_list(recency_decay):
        JaxDecisionTreeEnv._parse_recency_decay(recency_decay)


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
        recency_decay=jnp.asarray(
            [
                _resolve_recency_decay(combo["recency_decay"], combo["wm_decay"])
                for combo in combos
            ],
            dtype=jnp.float32,
        ),
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
        recency_decay=args["recency_decay"],
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
        "recency_decay",
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


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


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
        _log(
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


class ParallelJaxSimulator:
    def __init__(self, env: JaxDecisionTreeEnv, batch_size: int = 512):
        self.env = env
        self.batch_size = int(batch_size)
        self._simulate_sweep_batch_jit = jax.jit(
            self._simulate_sweep_batch,
            static_argnames=("greedy",),
        )

    def _run_trial(
        self,
        params,
        env_params: JaxDecisionTreeParams,
        rng_key: jax.Array,
        greedy: bool = False,
    ):
        state, obs, info = self.env.reset_with_params(rng_key, env_params)
        action_mask = info["mask"]

        action_seq = -jnp.ones((self.env.t_max,), dtype=jnp.int32)

        carry = (
            state,
            obs,
            action_mask,
            action_seq,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            rng_key,
        )

        def cond_fn(carry):
            _, _, _, _, step_count, done, _ = carry
            return (~done) & (step_count < self.env.t_max)

        def body_fn(carry):
            state, obs, action_mask, action_seq, step_count, _, rng_key = carry

            logits, _ = actor_critic_forward(params, obs[None, :], action_mask[None, :])
            logits = logits[0]

            def greedy_action(_):
                masked_logits = apply_action_mask(logits, action_mask)
                return jnp.argmax(masked_logits), rng_key

            def sampled_action(key):
                key, action_key = jax.random.split(key)
                action, _, _ = sample_actions(action_key, logits[None, :], action_mask[None, :])
                return action[0], key

            action, rng_key = jax.lax.cond(
                greedy,
                greedy_action,
                sampled_action,
                rng_key,
            )

            state, obs, _, done, _, info = self.env.step_with_params(state, action, env_params)
            action_mask = info["mask"]
            action_seq = action_seq.at[step_count].set(action)
            step_count = step_count + 1

            return state, obs, action_mask, action_seq, step_count, done, rng_key

        state, _, _, action_seq, action_len, _, rng_key = jax.lax.while_loop(
            cond_fn,
            body_fn,
            carry,
        )
        return state, action_seq, action_len, rng_key

    def _run_trial_batch(
        self,
        params,
        env_params: JaxDecisionTreeParams,
        trial_keys: jax.Array,
        greedy: bool = False,
    ):
        states, action_seqs, action_lens, _ = jax.vmap(
            lambda key: self._run_trial(params, env_params, key, greedy=greedy)
        )(trial_keys)
        return states, action_seqs, action_lens

    def _simulate_sweep_batch(
        self,
        params,
        env_params: JaxDecisionTreeParams,
        trial_keys: jax.Array,
        greedy: bool = False,
    ):
        simulate_seeds = jax.vmap(
            self._run_trial_batch,
            in_axes=(0, None, 0, None),
        )
        simulate_hypers = jax.vmap(
            simulate_seeds,
            in_axes=(0, 0, 0, None),
        )
        return simulate_hypers(params, env_params, trial_keys, greedy)

    def simulate_batch(
        self,
        params,
        env_params: JaxDecisionTreeParams,
        trial_keys: jax.Array,
        greedy: bool = False,
    ):
        return self._simulate_sweep_batch_jit(params, env_params, trial_keys, greedy=greedy)


def _initial_simulation_rng_keys(seeds: list[int], num_hypers: int) -> jax.Array:
    seed_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, dtype=jnp.int32))
    return jnp.broadcast_to(seed_keys[None, :, :], (num_hypers, len(seeds), 2))


def _split_simulation_rng_keys(rng_keys: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array]:
    split_keys = jax.vmap(jax.vmap(jax.random.split))(rng_keys)
    rng_keys = split_keys[:, :, 0]
    batch_keys = split_keys[:, :, 1]
    trial_keys = jax.vmap(jax.vmap(lambda key: jax.random.split(key, batch_size)))(batch_keys)
    return rng_keys, trial_keys


def _empty_simulation_data() -> dict[str, list]:
    return {
        "child_dicts": [],
        "parent_dicts": [],
        "root_nodes": [],
        "leaf_nodes": [],
        "depths": [],
        "points": [],
        "cum_points": [],
        "action_seqs": [],
        "choice_seqs": [],
    }


def _append_simulation_batch(
    data_by_run: list[list[dict[str, list]]],
    states,
    action_seqs,
    action_lens,
    *,
    trials_in_batch: int,
) -> None:
    states = jax.device_get(states)
    action_seqs = np.asarray(action_seqs)
    action_lens = np.asarray(action_lens)

    child_nodes_batch = np.asarray(states.child_nodes)
    parent_nodes_batch = np.asarray(states.parent_nodes)
    points_batch = np.asarray(states.points)
    root_nodes_batch = np.asarray(states.root_node)
    chosen_paths_batch = np.asarray(states.chosen_path)
    chosen_path_lens_batch = np.asarray(states.chosen_path_len)

    num_hypers = len(data_by_run)
    num_seeds = len(data_by_run[0]) if num_hypers else 0
    for hyper_index in range(num_hypers):
        for seed_index in range(num_seeds):
            data = data_by_run[hyper_index][seed_index]
            for trial_idx in range(trials_in_batch):
                child_nodes = child_nodes_batch[hyper_index, seed_index, trial_idx]
                parent_nodes = parent_nodes_batch[hyper_index, seed_index, trial_idx]
                points = points_batch[hyper_index, seed_index, trial_idx]
                root_node = int(root_nodes_batch[hyper_index, seed_index, trial_idx])

                action_len = int(action_lens[hyper_index, seed_index, trial_idx])
                action_seq = np.asarray(
                    action_seqs[hyper_index, seed_index, trial_idx, :action_len],
                    dtype=np.int32,
                ).tolist()

                choice_len = int(chosen_path_lens_batch[hyper_index, seed_index, trial_idx])
                choice_seq = np.asarray(
                    chosen_paths_batch[hyper_index, seed_index, trial_idx, :choice_len],
                    dtype=np.int32,
                ).tolist()

                data["child_dicts"].append(_child_array_to_dict(child_nodes))
                data["parent_dicts"].append(_parent_array_to_dict(parent_nodes))
                data["root_nodes"].append(root_node)
                data["leaf_nodes"].append(_leaf_nodes_from_children(child_nodes).tolist())
                data["depths"].append(_compute_depths(child_nodes, root_node).tolist())
                data["points"].append(points.tolist())
                data["cum_points"].append(_compute_cum_points(child_nodes, root_node, points).tolist())
                data["action_seqs"].append(action_seq)
                data["choice_seqs"].append(choice_seq)


def simulate_results(
    result,
    combos: list[dict],
    seeds: list[int],
    run_dirs: list[str],
    *,
    num_trials: int,
    greedy: bool,
    batch_size: int = 512,
) -> float:
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    start = time.time()
    env = _env_from_args(combos[0])
    simulator = ParallelJaxSimulator(env, batch_size=batch_size)
    hypers = build_hypers(combos)

    data_by_run = [
        [_empty_simulation_data() for _ in seeds]
        for _ in combos
    ]
    rng_keys = _initial_simulation_rng_keys(seeds, len(combos))
    num_batches = int(np.ceil(num_trials / batch_size))

    for batch_idx in range(num_batches):
        rng_keys, trial_keys = _split_simulation_rng_keys(rng_keys, batch_size)
        states, action_seqs, action_lens = simulator.simulate_batch(
            result.states.params,
            hypers.env,
            trial_keys,
            greedy=greedy,
        )
        trials_remaining = num_trials - (batch_idx * batch_size)
        trials_in_batch = min(batch_size, trials_remaining)
        _append_simulation_batch(
            data_by_run,
            states,
            action_seqs,
            action_lens,
            trials_in_batch=trials_in_batch,
        )

        elapsed_seconds = time.time() - start
        batches_done = batch_idx + 1
        batches_per_second = batches_done / elapsed_seconds
        eta_seconds = (num_batches - batches_done) / batches_per_second
        _log(
            "parallel_simulate_progress "
            f"batches={batches_done}/{num_batches} "
            f"elapsed={_format_duration(elapsed_seconds)} "
            f"eta={_format_duration(eta_seconds)}",
            flush=True,
        )

    run_dirs_by_index = np.asarray(run_dirs, dtype=object).reshape((len(combos), len(seeds)))
    for hyper_index, combo in enumerate(combos):
        env_for_output = _env_from_args(combo)
        for seed_index, _ in enumerate(seeds):
            run_dir = str(run_dirs_by_index[hyper_index, seed_index])
            transformed = to_transformed_simulation_format(
                data_by_run[hyper_index][seed_index],
                num_nodes=env_for_output.num_nodes,
                t_max=env_for_output.t_max,
                skip_timeout_trials=False,
                detailed=False,
            )
            output_path = os.path.join(run_dir, SIMULATION_DATA_NAME)
            with open(output_path, "w") as file:
                json.dump(_round_floats(transformed), file)
                file.write("\n")
            _log(f"simulation_json={output_path}")

    return time.time() - start


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

            log_path = os.path.join(run_dir, "training.log")
            with open(log_path, "w") as file:
                file.writelines(_CONSOLE_LOG_LINES)
                file.write(f"run_dir={run_dir}\n")
                file.write(
                    "run_summary "
                    f"hyper_index={hyper_index} "
                    f"seed={int(seed)} "
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

            run_dirs.append(run_dir)
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parallelized A2C TOML sweep.")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument("--experiment", help="Override experiment name. Defaults to [meta].experiment or config stem.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="After training, write data_simulation.json for each run using the same defaults as simulate.py.",
    )
    parser.add_argument(
        "--simulate-num-trials",
        type=int,
        default=10_240,
        help="Number of simulation trials per run when --simulate is set.",
    )
    parser.add_argument(
        "--simulate-greedy",
        action="store_true",
        help="Use greedy actions during --simulate, matching simulate.py --greedy.",
    )
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
        network_type=fixed["network_type"],
    )
    hypers = build_hypers(combos)

    devices = ", ".join(
        f"{device.platform}:{device.device_kind}"
        for device in jax.local_devices()
    )
    _log(f"jax_backend={jax.default_backend()} jax_devices=[{devices}]")
    _log(
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
    _log(f"parallel_train_elapsed_seconds={elapsed_seconds:.3f}")

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
    _log(f"saved_runs={len(run_dirs)}")
    for run_dir in run_dirs:
        _log(f"run_dir={run_dir}")

    if args.simulate:
        simulate_elapsed_seconds = simulate_results(
            result,
            combos,
            seeds,
            run_dirs,
            num_trials=int(args.simulate_num_trials),
            greedy=bool(args.simulate_greedy),
        )
        _log(f"parallel_simulate_elapsed_seconds={simulate_elapsed_seconds:.3f}")


if __name__ == "__main__":
    main()
