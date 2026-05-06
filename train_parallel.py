#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import pickle
import shutil
import statistics
import subprocess
import threading
import time
import tomllib
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from modules.a2c import A2CTrainParams, JaxBatchMaskA2C, JaxTrainState, StepMetrics, save_jax_params
from modules.argument import DEFAULT_PARAMS as ARG_DEFAULT_PARAMS
from modules.environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from modules.run_dirs import create_timestamped_run_dir, write_run_metadata
from modules.network import actor_critic_forward, apply_action_mask, sample_actions
from modules.simulation import (
    JaxSimulator,
    append_simulation_trial,
    empty_simulation_data,
)

jax.config.update('jax_compiler_enable_remat_pass', False)

EVAL_SUMMARY_NAME = "eval_summary_jax.json"
TRAINING_DATA_NAME = "data_training_jax.p"
SIMULATION_DATA_NAME = "data_simulation.json"

DEFAULT_META = {
    "result_path": "./results",
}


class A2CHyperParams(NamedTuple):
    env: JaxDecisionTreeParams
    lr: jax.Array
    gamma: jax.Array
    lamda: jax.Array
    beta_v: jax.Array
    beta_e_init: jax.Array
    beta_e_final: jax.Array
    max_grad_norm: jax.Array


class A2CSweepResult(NamedTuple):
    states: JaxTrainState
    metrics: StepMetrics


DEFAULT_PARAMS = {
    **ARG_DEFAULT_PARAMS,
    "max_compiled_updates_per_chunk": -1,
}

ENV_SWEEP_KEYS = {
    "beta_move",
    "eps_move",
    "learning_rate",
    "lamda_backup",
    "backup_steps",
    "wm_decay",
    "wm_backup",
    "q_drop_rate",
    "q_drift",
    "q_decay",
    "recency_decay",
    "cost",
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
    "num_envs",
    "rollout_length",
    "t_max",
    "num_updates",
    "eval_episodes",
    "network_type",
    "max_compiled_updates_per_chunk",
    "scale_factor",
    "shuffle_nodes",
}


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _query_gpu_stats() -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None

    gpu_utils: list[float] = []
    mem_utils: list[float] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_util = float(parts[0])
        _ = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        gpu_utils.append(gpu_util)
        mem_utils.append((mem_used / mem_total) * 100.0 if mem_total > 0 else 0.0)

    if not gpu_utils:
        return None
    return {
        "gpu_util_mean": float(sum(gpu_utils) / len(gpu_utils)),
        "gpu_util_max": float(max(gpu_utils)),
        "gpu_mem_util_mean": float(sum(mem_utils) / len(mem_utils)),
        "gpu_mem_util_max": float(max(mem_utils)),
    }


def _summarize_gpu_samples(samples: list[dict[str, float]]) -> str:
    if not samples:
        return "parallel_train_gpu_summary unavailable=true"

    gpu_util_mean = [sample["gpu_util_mean"] for sample in samples]
    gpu_util_max = [sample["gpu_util_max"] for sample in samples]
    gpu_mem_mean = [sample["gpu_mem_util_mean"] for sample in samples]
    gpu_mem_max = [sample["gpu_mem_util_max"] for sample in samples]

    return (
        "parallel_train_gpu_summary "
        f"samples={len(samples)} "
        f"gpu_util_mean={statistics.mean(gpu_util_mean):.1f} "
        f"gpu_util_p95={np.percentile(np.asarray(gpu_util_mean), 95):.1f} "
        f"gpu_util_peak={max(gpu_util_max):.1f} "
        f"gpu_mem_util_mean={statistics.mean(gpu_mem_mean):.1f} "
        f"gpu_mem_util_peak={max(gpu_mem_max):.1f}"
    )


class _GpuSampler:
    def __init__(self, interval_seconds: float = 0.5):
        self._interval_seconds = float(interval_seconds)
        self._samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval_seconds * 3.0)
        self._thread = None

    def snapshot_from(self, start_index: int) -> list[dict[str, float]]:
        return list(self._samples[start_index:])

    def count(self) -> int:
        return len(self._samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = _query_gpu_stats()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self._interval_seconds)


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


def _resolve_q_decay(value, q_drift, scale_factor) -> float:
    auto, decay = JaxDecisionTreeEnv._parse_q_decay(value)
    if not auto:
        return float(decay)

    drift_var = float(q_drift) ** 2
    point_set = np.asarray([-8, -4, -2, -1, 1, 2, 4, 8], dtype=np.float32)
    prior_var = max(float(np.var(point_set * float(scale_factor))), 1e-8)
    return drift_var / (drift_var + prior_var)


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
                        "params.recency_decay cannot include 'off' in train_parallel.py because it changes compiled shapes."
                    )
            continue
        if key in SHAPE_KEYS:
            raise ValueError(
                f"params.{key} cannot be an array in train_parallel.py because it changes compiled shapes."
            )
        if key not in SWEEP_KEYS:
            raise ValueError(f"params.{key} is not a supported parallel sweep parameter.")

    recency_decay = params.get("recency_decay", "off")
    if not _is_list(recency_decay):
        JaxDecisionTreeEnv._parse_recency_decay(recency_decay)
    q_decay = params.get("q_decay", 0.0)
    if _is_list(q_decay):
        for item in q_decay:
            JaxDecisionTreeEnv._parse_q_decay(item)
    else:
        JaxDecisionTreeEnv._parse_q_decay(q_decay)


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


def _resolve_training_geometry(params: dict) -> tuple[int, int, int]:
    if "num_updates" in params and "num_envs" in params and "rollout_length" in params:
        return int(params["num_updates"]), int(params["num_envs"]), int(params["rollout_length"])

    raise ValueError(
        "Training geometry must be specified as num_updates + num_envs + rollout_length."
    )


def _parse_cli_value(raw: str, template_value):
    if isinstance(template_value, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw!r}")
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw)
    if isinstance(template_value, float):
        return float(raw)
    return raw


def _apply_cli_param_overrides(params: dict, override_tokens: list[str]) -> dict:
    if not override_tokens:
        return dict(params)

    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    updated = dict(params)
    pairs: list[tuple[str, str]] = []
    idx = 0
    while idx < len(override_tokens):
        token = override_tokens[idx]
        if not token.startswith("--"):
            raise ValueError(f"Invalid override key {token!r}; expected '--<name>'.")
        if "=" in token:
            key_token, value_token = token.split("=", 1)
            if value_token == "":
                raise ValueError(f"Missing value for override {key_token!r}.")
            pairs.append((key_token, value_token))
            idx += 1
            continue
        if idx + 1 >= len(override_tokens):
            raise ValueError("Parameter overrides must be provided as '--key value' or '--key=value'.")
        pairs.append((token, override_tokens[idx + 1]))
        idx += 2

    for key_token, value_token in pairs:
        if not key_token.startswith("--"):
            raise ValueError(f"Invalid override key {key_token!r}; expected '--<name>'.")
        key = key_token[2:]
        if key not in merged:
            raise ValueError(f"Unknown parameter override: {key}")
        updated[key] = _parse_cli_value(value_token, merged[key])
    return updated


def build_hypers(combos: list[dict]) -> A2CHyperParams:
    def array(key: str, dtype=jnp.float32):
        return jnp.asarray([combo[key] for combo in combos], dtype=dtype)

    env = JaxDecisionTreeParams(
        beta_move=array("beta_move"),
        eps_move=array("eps_move"),
        learning_rate=array("learning_rate"),
        lamda_backup=array("lamda_backup"),
        backup_steps=array("backup_steps", dtype=jnp.int32),
        wm_decay=array("wm_decay"),
        wm_backup=array("wm_backup", dtype=np.bool_),
        q_drop_rate=array("q_drop_rate"),
        q_drift=array("q_drift"),
        q_decay=jnp.asarray(
            [
                _resolve_q_decay(combo["q_decay"], combo["q_drift"], combo["scale_factor"])
                for combo in combos
            ],
            dtype=jnp.float32,
        ),
        recency_decay=jnp.asarray(
            [
                _resolve_recency_decay(combo["recency_decay"], combo["wm_decay"])
                for combo in combos
            ],
            dtype=jnp.float32,
        ),
        cost=array("cost"),
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


class VmappedA2CTrainer:
    def __init__(
        self,
        env: JaxDecisionTreeEnv,
        feature_size: int,
        action_size: int,
        hidden_size: int,
        num_envs: int,
        num_updates: int,
        rollout_length: int | None = None,
        network_type: str = "mlp",
    ):
        self.num_updates = int(num_updates)
        self.trainer = JaxBatchMaskA2C(
            env=env,
            feature_size=feature_size,
            action_size=action_size,
            hidden_size=hidden_size,
            num_envs=num_envs,
            rollout_length=rollout_length,
            lr=1.0,
            gamma=1.0,
            lamda=1.0,
            beta_v=1.0,
            beta_e=0.0,
            network_type=network_type,
        )
        self._train_one_jit = jax.jit(self._train_one)
        self._train_sweep_jit = jax.jit(self._train_sweep)
        self._init_sweep_states_jit = jax.jit(self._init_sweep_states)
        self._train_sweep_chunk_jit = jax.jit(self._train_sweep_chunk)

    @staticmethod
    def _train_params(hyper: A2CHyperParams) -> A2CTrainParams:
        return A2CTrainParams(
            env=hyper.env,
            lr=hyper.lr,
            gamma=hyper.gamma,
            lamda=hyper.lamda,
            beta_v=hyper.beta_v,
            max_grad_norm=hyper.max_grad_norm,
        )

    def _train_one(self, hyper: A2CHyperParams, seed: jax.Array):
        state = self.trainer.init_state_with_params(seed, hyper.env)
        entropy_schedule = jnp.linspace(
            hyper.beta_e_init,
            hyper.beta_e_final,
            self.num_updates,
            dtype=jnp.float32,
        )
        return self.trainer._train_many(state, entropy_schedule, self._train_params(hyper))

    def _train_one_from_state(
        self,
        state: JaxTrainState,
        hyper: A2CHyperParams,
        entropy_schedule: jax.Array,
    ):
        return self.trainer._train_many(state, entropy_schedule, self._train_params(hyper))

    def _train_sweep(self, hypers: A2CHyperParams, seeds: jax.Array):
        train_seeds = jax.vmap(self._train_one, in_axes=(None, 0))
        train_hypers = jax.vmap(train_seeds, in_axes=(0, None))
        states, metrics = train_hypers(hypers, seeds)
        return A2CSweepResult(states=states, metrics=metrics)

    def _init_sweep_states(self, hypers: A2CHyperParams, seeds: jax.Array):
        def init_hyper(hyper):
            return jax.vmap(lambda seed: self.trainer.init_state_with_params(seed, hyper.env))(seeds)

        return jax.vmap(init_hyper)(hypers)

    def _train_sweep_chunk(
        self,
        states: JaxTrainState,
        hypers: A2CHyperParams,
        entropy_schedule: jax.Array,
    ):
        train_seeds = jax.vmap(self._train_one_from_state, in_axes=(0, None, None))
        train_hypers = jax.vmap(train_seeds, in_axes=(0, 0, 0))
        states, metrics = train_hypers(states, hypers, entropy_schedule)
        return A2CSweepResult(states=states, metrics=metrics)

    def train_sweep(self, hypers: A2CHyperParams, seeds):
        return self._train_sweep_jit(hypers, jnp.asarray(seeds, dtype=jnp.int32))

    def init_sweep_states(self, hypers: A2CHyperParams, seeds):
        return self._init_sweep_states_jit(hypers, jnp.asarray(seeds, dtype=jnp.int32))

    def train_sweep_chunk(
        self,
        states: JaxTrainState,
        hypers: A2CHyperParams,
        entropy_schedule,
    ):
        return self._train_sweep_chunk_jit(states, hypers, jnp.asarray(entropy_schedule, dtype=jnp.float32))

    def compile_train_sweep_chunk(
        self,
        states: JaxTrainState,
        hypers: A2CHyperParams,
        entropy_schedule,
    ) -> None:
        schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        self._train_sweep_chunk_jit.lower(states, hypers, schedule).compile()


def _env_from_args(args: dict) -> JaxDecisionTreeEnv:
    return JaxDecisionTreeEnv(
        num_nodes=args["num_nodes"],
        beta_move=args["beta_move"],
        eps_move=args["eps_move"],
        learning_rate=args["learning_rate"],
        lamda_backup=args["lamda_backup"],
        backup_steps=args["backup_steps"],
        wm_decay=args["wm_decay"],
        wm_backup=args["wm_backup"],
        q_drop_rate=args["q_drop_rate"],
        q_drift=args["q_drift"],
        q_decay=args["q_decay"],
        t_max=args["t_max"],
        cost=args["cost"],
        scale_factor=args["scale_factor"],
        shuffle_nodes=args["shuffle_nodes"],
        recency_decay=args["recency_decay"],
    )


def _env_cache_key(args: dict) -> tuple:
    keys = (
        "num_nodes",
        "beta_move",
        "eps_move",
        "learning_rate",
        "lamda_backup",
        "backup_steps",
        "wm_decay",
        "wm_backup",
        "q_drop_rate",
        "q_drift",
        "q_decay",
        "t_max",
        "cost",
        "scale_factor",
        "shuffle_nodes",
        "recency_decay",
    )
    return tuple(args[key] for key in keys)


def _metric_data(
    metrics,
    hyper_index: int,
    seed_index: int,
    elapsed_seconds: float,
) -> dict[str, list[float]]:
    metric_slice = jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x[hyper_index, seed_index])),
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


def _state_slice(states, hyper_index: int, seed_index: int):
    return jax.tree_util.tree_map(
        lambda x: jax.device_get(x[hyper_index, seed_index]),
        states,
    )


def _slug_value(value) -> str:
    text = str(value).strip()
    return "".join(char if char.isalnum() or char in {".", "-"} else "-" for char in text)


def _run_jobid(base_jobid: str, combo: dict, varied_keys: list[str], seed: int) -> str:
    param_parts = [f"{key}{_slug_value(combo[key])}" for key in varied_keys]
    param_parts.append(f"seed{int(seed)}")
    suffix = "_".join(param_parts)
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


def _concat_metric_chunks(chunks: list):
    return jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=2),
        *chunks,
    )


def _resolve_compiled_updates_per_chunk(
    requested_updates: int,
    *,
    max_compiled_updates_per_chunk: int,
) -> int:
    requested_updates = int(requested_updates)
    if requested_updates <= 0:
        raise ValueError("requested_updates must be positive")
    max_compiled_updates_per_chunk = int(max_compiled_updates_per_chunk)
    if max_compiled_updates_per_chunk <= 0:
        return requested_updates
    return min(requested_updates, max_compiled_updates_per_chunk)


def _init_per_run_training_logs(run_dirs: list[str]) -> None:
    col_sep = "   "
    header = col_sep.join(
        [
            f"{'update':>8}",
            f"{'ep_num':>10}",
            f"{'ep_rew':>8}",
            f"{'ep_len':>8}",
            f"{'loss':>8}",
            f"{'policy':>8}",
            f"{'value':>8}",
            f"{'entropy':>8}",
            f"{'grad_n':>8}",
            f"{'param_n':>8}",
        ]
    )
    for run_dir in run_dirs:
        log_path = os.path.join(run_dir, "training.log")
        with open(log_path, "w") as file:
            file.write(f"run_dir={run_dir}\n")
            file.write(header + "\n")
            file.write("-" * len(header) + "\n")


def _append_per_run_progress_logs(
    run_dirs: list[str],
    chunk_metrics,
    *,
    num_hypers: int,
    num_seeds: int,
    update_end: int,
    env_steps_per_update: int = 1,
    cumulative_episode_counts: np.ndarray | None = None,
) -> None:
    metrics = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), chunk_metrics)
    run_dirs_by_index = np.asarray(run_dirs, dtype=object).reshape((num_hypers, num_seeds))
    col_sep = "   "

    def _fmt_num(value: float, width: int = 8, decimals: int = 3) -> str:
        return f"{value: {width}.{decimals}f}"

    def _fmt_ep_num(value: int, width: int = 10) -> str:
        ep_num_k = int(round(value / 1_000.0))
        return f"{ep_num_k:>{width}d}K"

    for hyper_index in range(num_hypers):
        for seed_index in range(num_seeds):
            run_dir = str(run_dirs_by_index[hyper_index, seed_index])
            log_path = os.path.join(run_dir, "training.log")
            chunk_episode_count = float(np.sum(metrics.episode_count[hyper_index, seed_index]))
            if cumulative_episode_counts is not None:
                cumulative_episode_counts[hyper_index, seed_index] += chunk_episode_count
                ep_num = int(round(cumulative_episode_counts[hyper_index, seed_index]))
            else:
                ep_num = update_end * env_steps_per_update
            tail_a = _fmt_num(float(np.mean(metrics.grad_norm[hyper_index, seed_index])))
            tail_b = _fmt_num(float(np.mean(metrics.param_norm[hyper_index, seed_index])))
            line = col_sep.join(
                [
                    f"{update_end:>8d}",
                    _fmt_ep_num(ep_num),
                    _fmt_num(float(np.mean(metrics.episode_reward[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.episode_length[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.policy_loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.value_loss[hyper_index, seed_index]))),
                    _fmt_num(float(np.mean(metrics.entropy_loss[hyper_index, seed_index]))),
                    tail_a,
                    tail_b,
                ]
            )
            with open(log_path, "a") as file:
                file.write(line + "\n")


def train_with_progress(
    trainer: VmappedA2CTrainer,
    hypers: A2CHyperParams,
    seeds: list[int],
    *,
    run_dirs: list[str] | None = None,
    num_hypers: int | None = None,
    num_updates: int,
    env_steps_per_update: int = 1,
    print_frequency: int,
    max_compiled_updates_per_chunk: int = -1,
    include_gpu_summary: bool = False,
):
    has_run_dirs = run_dirs is not None
    if run_dirs is None:
        run_dirs = []
    if num_hypers is None:
        num_hypers = int(getattr(hypers.lr, "shape", [1])[0])
    if print_frequency <= 0:
        start = time.time()
        schedule = _entropy_schedule(hypers, num_updates)
        compiled_updates_per_chunk = _resolve_compiled_updates_per_chunk(
            num_updates,
            max_compiled_updates_per_chunk=max_compiled_updates_per_chunk,
        )
        states = jax.block_until_ready(trainer.init_sweep_states(hypers, seeds))
        trainer.compile_train_sweep_chunk(
            states,
            hypers,
            schedule[:, :compiled_updates_per_chunk],
        )
        metrics_chunks = []
        for update_start in range(0, num_updates, compiled_updates_per_chunk):
            update_end = min(update_start + compiled_updates_per_chunk, num_updates)
            result = jax.block_until_ready(
                trainer.train_sweep_chunk(states, hypers, schedule[:, update_start:update_end])
            )
            states = result.states
            metrics_chunks.append(result.metrics)
        result = A2CSweepResult(states=states, metrics=_concat_metric_chunks(metrics_chunks))
        gpu_stats = _query_gpu_stats()
        samples = [gpu_stats] if gpu_stats is not None else []
        if include_gpu_summary:
            return result, time.time() - start, _summarize_gpu_samples(samples)
        return result, time.time() - start

    schedule = _entropy_schedule(hypers, num_updates)
    compiled_updates_per_chunk = _resolve_compiled_updates_per_chunk(
        min(print_frequency, num_updates),
        max_compiled_updates_per_chunk=max_compiled_updates_per_chunk,
    )
    _log("parallel_train_init starting=true", flush=True)
    states = jax.block_until_ready(trainer.init_sweep_states(hypers, seeds))
    _log(
        "parallel_train_compile "
        f"updates_per_chunk={compiled_updates_per_chunk}",
        flush=True,
    )
    trainer.compile_train_sweep_chunk(
        states,
        hypers,
        schedule[:, :compiled_updates_per_chunk],
    )
    _log("parallel_train_compile done=true", flush=True)
    if has_run_dirs:
        _init_per_run_training_logs(run_dirs)

    start = time.time()
    metrics_chunks = []
    gpu_samples: list[dict[str, float]] = []
    cumulative_episode_counts = np.zeros((num_hypers, len(seeds)), dtype=np.float64)
    cumulative_episode_count_total = 0.0
    gpu_sampler = _GpuSampler(interval_seconds=0.5)
    gpu_sampler.start()
    col_sep = "   "
    header = col_sep.join(
        [
            f"{'update':>8}",
            f"{'ep_num':>10}",
            f"{'elapsed':>8}",
            f"{'ETA':>8}",
            f"{'upd/s':>8}",
            f"{'gpu%':>8}",
            f"{'mem%':>8}",
        ]
    )
    _log(header)
    _log("-" * len(header))

    try:
        for update_start in range(0, num_updates, print_frequency):
            update_end = min(update_start + print_frequency, num_updates)
            chunk_sample_start = gpu_sampler.count()
            window_metric_chunks = []

            for exec_start in range(update_start, update_end, compiled_updates_per_chunk):
                exec_end = min(exec_start + compiled_updates_per_chunk, update_end)
                chunk = schedule[:, exec_start:exec_end]
                result = jax.block_until_ready(trainer.train_sweep_chunk(states, hypers, chunk))
                states = result.states
                metrics_chunks.append(result.metrics)
                window_metric_chunks.append(result.metrics)

            window_metrics = _concat_metric_chunks(window_metric_chunks)
            chunk_episode_count_total = float(np.sum(np.asarray(jax.device_get(window_metrics.episode_count))))
            cumulative_episode_count_total += chunk_episode_count_total
            if has_run_dirs:
                _append_per_run_progress_logs(
                    run_dirs,
                    window_metrics,
                    num_hypers=num_hypers,
                    num_seeds=len(seeds),
                    update_end=update_end,
                    env_steps_per_update=env_steps_per_update,
                    cumulative_episode_counts=cumulative_episode_counts,
                )

            elapsed_seconds = time.time() - start
            updates_done = update_end
            updates_per_second = updates_done / elapsed_seconds
            eta_seconds = (num_updates - updates_done) / updates_per_second

            chunk_gpu_samples = gpu_sampler.snapshot_from(chunk_sample_start)
            if chunk_gpu_samples:
                gpu_stats = {
                    "gpu_util_mean": float(statistics.mean([s["gpu_util_mean"] for s in chunk_gpu_samples])),
                    "gpu_util_max": float(max(s["gpu_util_max"] for s in chunk_gpu_samples)),
                    "gpu_mem_util_mean": float(statistics.mean([s["gpu_mem_util_mean"] for s in chunk_gpu_samples])),
                    "gpu_mem_util_max": float(max(s["gpu_mem_util_max"] for s in chunk_gpu_samples)),
                }
                gpu_samples.append(gpu_stats)
                gpu_util_text = f"{gpu_stats['gpu_util_mean']:.1f}"
                gpu_mem_text = f"{gpu_stats['gpu_mem_util_mean']:.1f}"
            else:
                gpu_util_text = "n/a"
                gpu_mem_text = "n/a"

            if not has_run_dirs:
                _log(
                    "parallel_train_progress "
                    f"updates={updates_done}/{num_updates} "
                    f"elapsed={elapsed_seconds:.3f} "
                    f"updates_per_second={updates_per_second:.6f}",
                    flush=True,
                )
            _log(
                col_sep.join(
                    [
                        f"{updates_done:>8d}",
                        f"{int(round(cumulative_episode_count_total / 1_000.0)):>9d}K",
                        f"{_format_duration(elapsed_seconds):>8}",
                        f"{_format_duration(eta_seconds):>8}",
                        f"{updates_per_second:>8.3f}",
                        f"{gpu_util_text:>8}",
                        f"{gpu_mem_text:>8}",
                    ]
                ),
                flush=True,
            )
    finally:
        gpu_sampler.stop()

    metrics = _concat_metric_chunks(metrics_chunks)
    output = (A2CSweepResult(states=states, metrics=metrics), time.time() - start)
    if include_gpu_summary:
        return output[0], output[1], _summarize_gpu_samples(gpu_samples)
    return output


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
        final_action_index = jnp.maximum(action_len - 1, 0)
        final_action = action_seq[final_action_index]

        def sample_choice_path(payload):
            state, rng_key = payload
            sample_state = state._replace(rng_key=rng_key)
            _, path, path_len, rng_key = self.env._sample_move_path(sample_state, env_params)
            return path, path_len, rng_key

        def empty_choice_path(payload):
            _, rng_key = payload
            return self.env.empty_path, jnp.int32(0), rng_key

        choice_path, choice_path_len, rng_key = jax.lax.cond(
            final_action == self.env.num_nodes,
            sample_choice_path,
            empty_choice_path,
            (state, rng_key),
        )

        return state, action_seq, choice_path, action_len, choice_path_len, rng_key

    def _run_trial_batch(
        self,
        params,
        env_params: JaxDecisionTreeParams,
        trial_keys: jax.Array,
        greedy: bool = False,
    ):
        states, action_seqs, choice_paths, action_lens, choice_path_lens, _ = jax.vmap(
            lambda key: self._run_trial(params, env_params, key, greedy=greedy)
        )(trial_keys)
        return states, action_seqs, choice_paths, action_lens, choice_path_lens

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
    return empty_simulation_data(detailed=False)


def _append_simulation_batch(
    data_by_run: list[list[dict[str, list]]],
    states,
    action_seqs,
    choice_paths,
    action_lens,
    choice_path_lens,
    *,
    trials_in_batch: int,
) -> None:
    states = jax.device_get(states)
    action_seqs = np.asarray(action_seqs)
    choice_paths = np.asarray(choice_paths)
    action_lens = np.asarray(action_lens)
    choice_path_lens = np.asarray(choice_path_lens)

    child_nodes_batch = np.asarray(states.child_nodes)
    points_batch = np.asarray(states.points)
    root_nodes_batch = np.asarray(states.root_node)

    num_hypers = len(data_by_run)
    num_seeds = len(data_by_run[0]) if num_hypers else 0
    for hyper_index in range(num_hypers):
        for seed_index in range(num_seeds):
            data = data_by_run[hyper_index][seed_index]
            for trial_idx in range(trials_in_batch):
                child_nodes = child_nodes_batch[hyper_index, seed_index, trial_idx]
                points = points_batch[hyper_index, seed_index, trial_idx]
                root_node = int(root_nodes_batch[hyper_index, seed_index, trial_idx])

                action_len = int(action_lens[hyper_index, seed_index, trial_idx])
                action_seq = np.asarray(
                    action_seqs[hyper_index, seed_index, trial_idx, :action_len],
                    dtype=np.int32,
                ).tolist()

                choice_len = int(choice_path_lens[hyper_index, seed_index, trial_idx])
                choice_seq = np.asarray(
                    choice_paths[hyper_index, seed_index, trial_idx, :choice_len],
                    dtype=np.int32,
                ).tolist()

                append_simulation_trial(
                    data,
                    child_nodes=child_nodes,
                    root_node=root_node,
                    points=points,
                    action_seq=action_seq,
                    choice_seq=choice_seq,
                    num_nodes=child_nodes.shape[0],
                    t_max=action_seqs.shape[-1],
                    skip_timeout_trials=False,
                )


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
        states, action_seqs, choice_paths, action_lens, choice_path_lens = simulator.simulate_batch(
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
            choice_paths,
            action_lens,
            choice_path_lens,
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
    for hyper_index, _ in enumerate(combos):
        for seed_index, _ in enumerate(seeds):
            run_dir = str(run_dirs_by_index[hyper_index, seed_index])
            output_path = os.path.join(run_dir, SIMULATION_DATA_NAME)
            with open(output_path, "w") as file:
                json.dump(_round_floats(data_by_run[hyper_index][seed_index]), file)
                file.write("\n")
            _log(f"simulation_json={output_path}")

    return time.time() - start


def save_results(
    result,
    combos: list[dict],
    seeds: list[int],
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
            combos,
            seeds,
            path=path,
            experiment=experiment,
            config_path=config_path,
            varied_keys=[] if varied_keys is None else varied_keys,
        )
    simulators: dict[tuple, JaxSimulator] = {}
    run_dirs_by_index = np.asarray(run_dirs, dtype=object).reshape((len(combos), len(seeds)))
    for hyper_index, combo in enumerate(combos):
        env_key = _env_cache_key(combo)
        if env_key not in simulators:
            simulators[env_key] = JaxSimulator(_env_from_args(combo))
        simulator = simulators[env_key]

        for seed_index, seed in enumerate(seeds):
            run_args = dict(combo)
            run_args["seed"] = int(seed)
            run_dir = str(run_dirs_by_index[hyper_index, seed_index])

            state = _state_slice(result.states, hyper_index, seed_index)
            data = _metric_data(
                result.metrics,
                hyper_index,
                seed_index,
                elapsed_seconds,
            )

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
                "num_updates": int(run_args["num_updates"]),
            }
            with open(os.path.join(run_dir, EVAL_SUMMARY_NAME), "w") as file:
                json.dump(eval_summary, file, indent=2, sort_keys=True)

            log_path = os.path.join(run_dir, "training.log")
            with open(log_path, "a") as file:
                file.write("\n")
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
    return run_dirs


def prepare_run_dirs(
    combos: list[dict],
    seeds: list[int],
    *,
    path: str,
    experiment: str,
    config_path: Path,
    varied_keys: list[str],
) -> list[str]:
    run_dirs: list[str] = []
    for hyper_index, combo in enumerate(combos):
        for seed in seeds:
            run_args = dict(combo)
            run_args["seed"] = int(seed)
            run_args["parallel_config"] = str(config_path)
            run_args["parallel_varied_keys"] = list(varied_keys)

            run_dir = create_timestamped_run_dir(
                path=path,
                experiment=experiment,
                jobid=_run_jobid(str(combo.get("jobid", "")), combo, varied_keys, seed),
            )
            write_run_metadata(run_dir=run_dir, args=Namespace(**run_args), cwd=os.getcwd())
            run_dirs.append(run_dir)
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parallelized RL TOML sweep (A2C).")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument("--experiment", help="Override experiment name. Defaults to [meta].experiment or config stem.")
    parser.add_argument(
        "--simulate-num-trials",
        type=int,
        default=10_240,
        help="Number of simulation trials per run.",
    )
    parser.add_argument(
        "--simulate-greedy",
        action="store_true",
        help="Use greedy actions during simulation, matching simulate.py --greedy.",
    )
    parser.add_argument(
        "--skip-simulate",
        action="store_true",
        help="Skip post-training simulation.",
    )
    args, override_tokens = parser.parse_known_args()

    config_path, config = _load_config(args.config)
    meta = dict(DEFAULT_META)
    meta.update(config.get("meta", {}))
    params = _apply_cli_param_overrides(config.get("params", {}), override_tokens)

    fixed, combos, seeds, varied_keys = expand_sweep(params)
    algo = str(fixed.get("algo", "a2c")).lower()
    if algo != "a2c":
        raise ValueError(f"Unsupported algo {algo!r}; expected 'a2c'.")
    output_path = args.path or str(meta["result_path"])
    experiment = args.experiment or str(meta.get("experiment", config_path.stem))
    results_dir_display = os.path.join(output_path, "runs", experiment, "")
    if results_dir_display.startswith("./"):
        results_dir_display = results_dir_display[2:]
    _log(f"writing results to {results_dir_display}")

    num_updates, num_envs, rollout_length = _resolve_training_geometry(fixed)
    env = _env_from_args(combos[0])
    trainer = VmappedA2CTrainer(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=num_envs,
        num_updates=num_updates,
        rollout_length=rollout_length,
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
        f"algo={algo} "
        f"hyper_combos={len(combos)} "
        f"seeds={len(seeds)} "
        f"num_updates={num_updates} "
        f"num_envs={num_envs} "
        f"rollout_length={rollout_length} "
        f"t_max={fixed['t_max']} "
        f"varied_keys={','.join(varied_keys)}"
    )
    run_dirs = prepare_run_dirs(
        combos,
        seeds,
        path=output_path,
        experiment=experiment,
        config_path=config_path,
        varied_keys=varied_keys,
    )
    _log(f"prepared_runs={len(run_dirs)}")

    result, elapsed_seconds, gpu_summary_line = train_with_progress(
        trainer,
        hypers,
        seeds,
        run_dirs=run_dirs,
        num_hypers=len(combos),
        num_updates=num_updates,
        env_steps_per_update=num_envs * rollout_length,
        print_frequency=int(fixed["print_frequency"]),
        max_compiled_updates_per_chunk=int(fixed["max_compiled_updates_per_chunk"]),
        include_gpu_summary=True,
    )
    _log(f"parallel_train_elapsed_seconds={elapsed_seconds:.3f}")
    _log(gpu_summary_line)

    run_dirs = save_results(
        result,
        combos,
        seeds,
        run_dirs,
        elapsed_seconds=elapsed_seconds,
    )
    _log(f"saved_runs={len(run_dirs)}")
    for run_dir in run_dirs:
        _log(f"run_dir={run_dir}")

    if args.skip_simulate:
        _log("parallel_simulate_skipped=true")
    else:
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
