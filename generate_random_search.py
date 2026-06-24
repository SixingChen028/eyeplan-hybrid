#!/usr/bin/env python3
import argparse
import json
import os
import time

from modules.config import DEFAULT_META, expand_config_runs, load_config
from modules.random_search import (
    RANDOM_SEARCH_STOP_GAMMA_SCALE,
    RANDOM_SEARCH_STOP_GAMMA_SHAPE,
    RANDOM_SEARCH_STOP_MAX_FIXATIONS,
    RandomSearchSimulator,
)
from modules.train_results import env_cache_key, env_from_args, env_params_from_args, prepare_run_dirs


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _write_training_log(
    run_dir: str,
    *,
    run_index: int,
    seed: int,
    elapsed_seconds: float,
    num_trials: int,
) -> None:
    log_path = os.path.join(run_dir, "training.log")
    with open(log_path, "a") as file:
        file.write("\n")
        file.write(
            "random_search_summary "
            f"run_index={run_index} "
            f"seed={seed} "
            f"elapsed_seconds={elapsed_seconds:.3f} "
            f"num_trials={num_trials} "
            f"gamma_shape={RANDOM_SEARCH_STOP_GAMMA_SHAPE} "
            f"gamma_scale={RANDOM_SEARCH_STOP_GAMMA_SCALE} "
            f"max_fixations={RANDOM_SEARCH_STOP_MAX_FIXATIONS}\n"
        )
        file.write(f"training_log={log_path}\n")


def _with_random_search_metadata(run: dict) -> dict:
    out = dict(run)
    out["label"] = "random_search"
    out["lesion_policy"] = "random_search_gamma_stopping"
    out["random_search_stop_gamma_shape"] = RANDOM_SEARCH_STOP_GAMMA_SHAPE
    out["random_search_stop_gamma_scale"] = RANDOM_SEARCH_STOP_GAMMA_SCALE
    out["random_search_stop_max_fixations"] = RANDOM_SEARCH_STOP_MAX_FIXATIONS
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random-search lesion simulations for a TOML sweep.")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument("--experiment", help="Override experiment name. Defaults to <config stem>_random_search.")
    parser.add_argument("--condition", type=int, help="0-based [[conditions]] table index to generate.")
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--skip_timeout_trials", action="store_true")
    args, override_tokens = parser.parse_known_args()

    config_path, config = load_config(args.config)
    meta = dict(DEFAULT_META)
    meta.update(config.get("meta", {}))

    _, runs, varied_keys, _, condition_index = expand_config_runs(
        config,
        condition_index=args.condition,
        override_tokens=override_tokens,
    )
    runs = [_with_random_search_metadata(run) for run in runs]

    output_path = args.path or str(meta["result_path"])
    experiment = args.experiment or f"{config_path.stem}_random_search"
    run_dirs = prepare_run_dirs(
        runs,
        path=output_path,
        experiment=experiment,
        config_path=config_path,
        varied_keys=varied_keys,
        label=None,
        condition_index=condition_index,
        run_eval=False,
        eval_episodes=None,
    )

    start = time.time()
    simulators = {}
    for run_index, (run, run_dir) in enumerate(zip(runs, run_dirs)):
        env_key = env_cache_key(run)
        if env_key not in simulators:
            env = env_from_args(run)
            simulators[env_key] = RandomSearchSimulator(env, env_params_from_args(env, run))
        simulator = simulators[env_key]

        data = simulator.simulate(
            seed=int(run["seed"]),
            num_trials=int(args.num_trials),
            batch_size=int(args.batch_size),
            skip_timeout_trials=bool(args.skip_timeout_trials),
        )
        output_file = os.path.join(run_dir, "data_simulation.json")
        with open(output_file, "w") as file:
            json.dump(_round_floats(data), file)
            file.write("\n")

        _write_training_log(
            run_dir,
            run_index=run_index,
            seed=int(run["seed"]),
            elapsed_seconds=time.time() - start,
            num_trials=len(data["actions"]),
        )
        print(f"{run_index + 1}/{len(runs)} {output_file} trials={len(data['actions'])}", flush=True)


if __name__ == "__main__":
    main()
