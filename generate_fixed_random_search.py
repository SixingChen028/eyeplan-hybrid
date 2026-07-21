#!/usr/bin/env python3
import argparse
import json
import os
import time

from generate_random_search import _round_floats
from modules.config import DEFAULT_META, SHAPE_KEYS, expand_config_runs, is_list, load_config
from modules.random_search import RandomSearchSimulator
from modules.train_results import env_cache_key, env_from_args, env_params_from_args, prepare_run_dirs


def _with_fixed_random_search_metadata(
    run: dict,
    *,
    total_fixations: int,
    base_label: str | None,
) -> dict:
    out = dict(run)
    out["label"] = "undirected"
    out["lesion_policy"] = "random_search_fixed_stopping"
    out["fixed_fixations"] = total_fixations
    if base_label is not None:
        out["random_search_base_label"] = base_label
    return out


def _write_training_log(
    run_dir: str,
    *,
    run_index: int,
    seed: int,
    elapsed_seconds: float,
    num_trials: int,
    total_fixations: int,
) -> None:
    log_path = os.path.join(run_dir, "training.log")
    with open(log_path, "a") as file:
        file.write("\n")
        file.write(
            "fixed_random_search_summary "
            f"run_index={run_index} "
            f"seed={seed} "
            f"elapsed_seconds={elapsed_seconds:.3f} "
            f"num_trials={num_trials} "
            f"total_fixations={total_fixations}\n"
        )
        file.write(f"training_log={log_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-fixation random-search simulations for a TOML sweep.")
    parser.add_argument("config", help="TOML config path or config stem under ./config.")
    parser.add_argument("--path", help="Override output path from [meta].result_path.")
    parser.add_argument(
        "--experiment",
        help="Override experiment name. Defaults to <config stem>_fixed_random_search.",
    )
    parser.add_argument("--condition", type=int, help="0-based [[conditions]] table index to generate.")
    parser.add_argument("--fixations", type=int, nargs="+", default=list(range(1, 31)))
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--skip_timeout_trials", action="store_true")
    args, override_tokens = parser.parse_known_args()

    if len(set(args.fixations)) != len(args.fixations):
        parser.error("--fixations values must be unique")

    config_path, config = load_config(args.config)
    meta = dict(DEFAULT_META)
    meta.update(config.get("meta", {}))
    num_trials = args.num_trials or meta.get("sim_trials") or 1000

    params = config.get("params", {})
    for key in list(params):
        if key in SHAPE_KEYS and is_list(params[key]):
            params[key] = params[key][0]

    _, base_runs, varied_keys, condition_label, condition_index = expand_config_runs(
        config,
        condition_index=args.condition,
        override_tokens=override_tokens,
    )
    runs = [
        _with_fixed_random_search_metadata(
            run,
            total_fixations=total_fixations,
            base_label=condition_label,
        )
        for run in base_runs
        for total_fixations in args.fixations
    ]
    varied_keys = [*varied_keys, "fixed_fixations"]

    output_path = args.path or str(meta["result_path"])
    experiment = args.experiment or f"{config_path.stem}_fixed_random_search"
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
        total_fixations = int(run["fixed_fixations"])
        simulator_key = (env_cache_key(run), total_fixations)
        if simulator_key not in simulators:
            env = env_from_args(run)
            simulators[simulator_key] = RandomSearchSimulator(
                env,
                env_params_from_args(env, run),
                total_fixations=total_fixations,
            )
        simulator = simulators[simulator_key]

        data = simulator.simulate(
            seed=int(run["seed"]),
            num_trials=int(num_trials),
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
            total_fixations=total_fixations,
        )
        print(f"{run_index + 1}/{len(runs)} {output_file} trials={len(data['actions'])}", flush=True)


if __name__ == "__main__":
    main()
