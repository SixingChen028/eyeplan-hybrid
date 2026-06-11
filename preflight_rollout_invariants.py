#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable

from modules.config import ENV_DYNAMIC_PARAM_KEYS, ENV_STATIC_PARAM_KEYS, expand_config_runs, load_config
from modules.rollout_invariants import collect_random_fixation_rollouts, assert_fixation_rollout_invariants
from modules.train_results import env_from_args, env_params_from_args


ENVIRONMENT_KEYS = (*ENV_STATIC_PARAM_KEYS, *ENV_DYNAMIC_PARAM_KEYS)


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def unique_environment_runs(config: dict) -> list[dict]:
    runs = []
    conditions = config.get("conditions", [])
    condition_indices: Iterable[int | None] = range(len(conditions)) if conditions else (None,)

    for condition_index in condition_indices:
        _, condition_runs, _, _, _ = expand_config_runs(config, condition_index=condition_index)
        runs.extend(condition_runs)

    unique = {}
    for run in runs:
        key = tuple((env_key, _freeze(run[env_key])) for env_key in ENVIRONMENT_KEYS)
        unique.setdefault(key, run)
    return list(unique.values())


def expects_max_consistent_q(run: dict) -> bool:
    return (
        float(run["beta_move"]) >= 1000.0
        and float(run["eps_move"]) == 0.0
        and float(run["learning_rate"]) == 1.0
        and float(run["lamda_backup"]) == 1.0
        and int(run["backup_steps"]) >= int(run["num_nodes"])
        and float(run["wm_decay"]) == 1.0
        and float(run["wm_neighbor_activation"]) == 1.0
        and float(run["forget_rate"]) == 0.0
        and float(run["q_drift"]) == 0.0
        and float(run["q_decay"]) == 1.0
    )


def validate_environment_run(
    run: dict,
    *,
    seed: int,
    num_rollouts: int,
    noisy_steps: int,
    max_consistency_steps: int,
) -> tuple[bool, int]:
    expect_max_consistent_q = expects_max_consistent_q(run)
    num_steps = max_consistency_steps if expect_max_consistent_q else noisy_steps
    env = env_from_args(run)
    params = env_params_from_args(env, run)
    trace = collect_random_fixation_rollouts(
        env,
        params,
        seed=seed,
        num_rollouts=num_rollouts,
        num_steps=num_steps,
    )
    assert_fixation_rollout_invariants(
        env,
        params,
        trace,
        expect_max_consistent_q=expect_max_consistent_q,
    )
    return expect_max_consistent_q, num_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight rollout invariants for every unique config environment.")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--num-rollouts", type=int, default=1000)
    parser.add_argument("--noisy-steps", type=int, default=30)
    parser.add_argument("--max-consistency-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    runs = unique_environment_runs(config)
    print(f"preflight_config={config_path} unique_environments={len(runs)}", flush=True)

    for index, run in enumerate(runs):
        expect_max_consistent_q, num_steps = validate_environment_run(
            run,
            seed=args.seed + index,
            num_rollouts=args.num_rollouts,
            noisy_steps=args.noisy_steps,
            max_consistency_steps=args.max_consistency_steps,
        )
        mode = "max_consistency" if expect_max_consistent_q else "general"
        print(
            f"preflight_environment index={index} mode={mode} "
            f"num_nodes={run['num_nodes']} t_max={run['t_max']} "
            f"rollouts={args.num_rollouts} steps={num_steps}",
            flush=True,
        )

    print("preflight_rollout_invariants=passed", flush=True)


if __name__ == "__main__":
    main()
