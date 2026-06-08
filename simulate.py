import argparse
import json
import os
import pickle
import subprocess
import sys
import traceback


def _configure_jax_platform(argv: list[str], environ: dict[str, str]) -> None:
    if "--gpu" in argv[1:]:
        return

    environ["JAX_PLATFORMS"] = "cpu"
    environ["JAX_PLATFORM_NAME"] = "cpu"


_configure_jax_platform(sys.argv, os.environ)

from modules.a2c import load_jax_params
from modules.evaluation import (
    PARAMS_NAME,
    env_from_run_args as _build_env_from_metadata_args,
    env_params_from_run_args as _build_env_params_from_metadata_args,
    read_metadata as _read_metadata,
    read_metadata_args as _read_metadata_args,
    resolve_params_path_from_metadata as _resolve_params_path_from_metadata,
)
from modules.results_layout import resolve_analysis_target
from modules.simulation import JaxSimulator
from modules.pipeline_compat import get_pipeline_compat_version


def _uses_node_shared_network(params) -> bool:
    return isinstance(params, dict) and "node_fc1" in params


def _metadata_args_with_simulation_overrides(
    metadata_args: dict,
    params,
    *,
    num_nodes: int | None,
    t_max: int | None,
) -> dict:
    args = dict(metadata_args)

    if num_nodes is not None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not _uses_node_shared_network(params):
            raise ValueError(
                "--num_nodes can only be used with node_shared or global_shared network parameters. "
                "Dense MLP parameters include environment-size-dependent input and action dimensions."
            )
        args["num_nodes"] = int(num_nodes)

    if t_max is not None:
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        args["t_max"] = int(t_max)

    return args


def _default_output_name(*, detailed: bool, num_nodes: int | None, t_max: int | None) -> str:
    stem = "data_simulation_detailed" if detailed else "data_simulation"
    suffix_parts = []
    if num_nodes is not None:
        suffix_parts.append(f"num_nodes{num_nodes}")
    if t_max is not None:
        suffix_parts.append(f"t_max{t_max}")
    if suffix_parts:
        stem = f"{stem}_{'_'.join(suffix_parts)}"
    return f"{stem}.json"


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _expected_num_updates(metadata_args: dict) -> int:
    num_updates = int(metadata_args.get("num_updates", 0))
    if num_updates <= 0:
        return 0
    return num_updates


def _read_checkpoint_next_update(run_dir: str) -> int | None:
    checkpoint_meta_path = os.path.join(run_dir, "checkpoints", "train_state_latest.json")
    if not os.path.exists(checkpoint_meta_path):
        return None
    with open(checkpoint_meta_path, "r") as file:
        checkpoint_meta = json.load(file)
    next_update = checkpoint_meta.get("next_update")
    if next_update is None:
        return None
    return int(next_update)


def _read_recorded_updates_from_training_data(run_dir: str) -> int | None:
    training_data_path = os.path.join(run_dir, "data_training_jax.p")
    if not os.path.exists(training_data_path):
        return None
    with open(training_data_path, "rb") as file:
        training_data = pickle.load(file)
    if not isinstance(training_data, dict):
        return None
    rewards = training_data.get("episode_reward")
    if not isinstance(rewards, list):
        return None
    return len(rewards)


def _read_eval_summary_updates(run_dir: str) -> int | None:
    eval_summary_path = os.path.join(run_dir, "eval_summary_jax.json")
    if not os.path.exists(eval_summary_path):
        return None
    with open(eval_summary_path, "r") as file:
        eval_summary = json.load(file)
    if not isinstance(eval_summary, dict):
        return None
    num_updates = eval_summary.get("num_updates")
    if num_updates is None:
        return None
    return int(num_updates)


def _is_complete_run(run_dir: str) -> bool:
    try:
        metadata = _read_metadata(run_dir)
        metadata_args = metadata.get("args")
        if not isinstance(metadata_args, dict):
            return False

        if os.path.exists(os.path.join(run_dir, PARAMS_NAME)):
            return True

        # If model parameters cannot be resolved, the run cannot be simulated.
        _resolve_params_path_from_metadata(run_dir, metadata)

        expected_updates = _expected_num_updates(metadata_args)
        if expected_updates <= 0:
            return False

        checkpoint_next_update = _read_checkpoint_next_update(run_dir)
        if checkpoint_next_update is not None and checkpoint_next_update >= expected_updates:
            return True

        recorded_updates = _read_recorded_updates_from_training_data(run_dir)
        if recorded_updates is not None and recorded_updates >= expected_updates:
            return True

        eval_summary_updates = _read_eval_summary_updates(run_dir)
        if eval_summary_updates is not None and eval_summary_updates >= expected_updates:
            return True

        return False
    except Exception:
        return False


def _simulate_run(
    run_dir: str,
    *,
    output_path: str,
    num_trials: int,
    num_nodes: int | None,
    t_max: int | None,
    greedy: bool,
    skip_timeout_trials: bool,
    detailed: bool,
    allow_unversioned_params: bool,
) -> tuple[str, int, int, int]:

    metadata = _read_metadata(run_dir)
    metadata_args = _read_metadata_args(run_dir)
    params_path = _resolve_params_path_from_metadata(run_dir, metadata)
    params = load_jax_params(
        params_path,
        allow_unversioned=allow_unversioned_params,
        expected_pipeline_compat_version=get_pipeline_compat_version(metadata),
    )

    metadata_args = _metadata_args_with_simulation_overrides(
        metadata_args,
        params,
        num_nodes=num_nodes,
        t_max=t_max,
    )
    env = _build_env_from_metadata_args(metadata_args)
    env_params = _build_env_params_from_metadata_args(env, metadata_args)

    seed = int(metadata_args.get("seed", 15))
    simulator = JaxSimulator(env, env_params=env_params)
    data = simulator.simulate(
        params=params,
        seed=seed,
        num_trials=num_trials,
        greedy=greedy,
        detailed=detailed,
        skip_timeout_trials=skip_timeout_trials,
    )

    with open(output_path, "w") as file:
        json.dump(_round_floats(data), file)
        file.write("\n")

    return params_path, seed, num_trials, len(data["actions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="+",
        type=str,
        help="One or more targets: <experiment>, results/runs/<experiment>, or results/runs/<experiment>/<run_id>.",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--skip_timeout_trials", action="store_true")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seed-filter", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-unversioned-params", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Allow JAX to use GPU devices.")
    args = parser.parse_args()
    if args.viewer:
        args.detailed = True
    if args.seed_filter is None and args.detailed:
        args.seed_filter = 1
    num_trials = args.num_trials
    if num_trials is None:
        num_trials = 100 if args.detailed else 5000

    runs_to_simulate: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    had_error = False
    for target_arg in args.targets:
        try:
            target = resolve_analysis_target(target_arg, results_root=args.results_root)
            if target.kind == "experiment":
                run_dirs = [run_dir for run_dir in target.run_dirs if _is_complete_run(run_dir)]
                skipped_incomplete = len(target.run_dirs) - len(run_dirs)
                print(
                    f"target={target_arg} target_kind={target.kind} "
                    f"experiment={target.experiment} runs={len(run_dirs)} "
                    f"skipped_incomplete={skipped_incomplete}"
                )
                if not run_dirs:
                    had_error = True
                    print(f"No complete runs found for experiment target: {target_arg}", file=sys.stderr)
                    continue
            else:
                run_dirs = target.run_dirs
                print(f"target={target_arg} target_kind={target.kind} experiment={target.experiment} runs={len(run_dirs)}")
            for run_dir in run_dirs:
                if args.seed_filter is not None:
                    metadata_args = _read_metadata_args(run_dir)
                    if int(metadata_args.get("seed", 15)) != args.seed_filter:
                        continue
                run_key = (target.experiment, run_dir)
                if run_key in seen:
                    continue
                seen.add(run_key)
                runs_to_simulate.append(run_key)
        except Exception:
            had_error = True
            print(f"Error resolving target: {target_arg}", file=sys.stderr)
            traceback.print_exc()
            continue

    if args.output != "" and len(runs_to_simulate) != 1:
        print("--output can only be used when exactly one run is resolved.", file=sys.stderr)
        had_error = True
        runs_to_simulate = []

    total = len(runs_to_simulate)
    simulated_experiments: set[str] = set()

    for idx, (experiment, run_dir) in enumerate(runs_to_simulate, start=1):
        try:
            output_path = args.output
            if output_path == "":
                output_name = _default_output_name(
                    detailed=args.detailed,
                    num_nodes=args.num_nodes,
                    t_max=args.t_max,
                )
                output_path = os.path.join(run_dir, output_name)
            output_path = os.path.abspath(os.path.expanduser(output_path))

            if not args.overwrite and os.path.exists(output_path):
                print(f"{idx:>2}/{total:<3} skip existing {output_path}")
                continue

            params_path, seed, num_trials_raw, num_trials_exported = _simulate_run(
                run_dir=run_dir,
                output_path=output_path,
                num_trials=num_trials,
                num_nodes=args.num_nodes,
                t_max=args.t_max,
                greedy=args.greedy,
                skip_timeout_trials=args.skip_timeout_trials,
                detailed=args.detailed,
                allow_unversioned_params=args.allow_unversioned_params,
            )
            print(f"{idx:>2}/{total:<3} {output_path}")
            simulated_experiments.add(experiment)
            # print(
            #     f"output_json={output_path} "
            #     f"run_dir={run_dir} "
            #     f"params_path={params_path} "
            #     f"seed={seed} "
            #     f"num_trials_raw={num_trials_raw} "
            #     f"num_trials_exported={num_trials_exported} "
            #     f"experiment={experiment}"
            # )
        except Exception:
            had_error = True
            print(f"Error simulating run: {run_dir}", file=sys.stderr)
            traceback.print_exc()
            continue

    if args.viewer and simulated_experiments:
        viewer_root = os.path.expanduser("~/projects/eyeplan/tree-viewer")
        experiment_dirs = [
            os.path.abspath(os.path.join(args.results_root, "runs", experiment))
            for experiment in sorted(simulated_experiments)
        ]
        subprocess.run(
            ["bun", "register", *experiment_dirs],
            cwd=viewer_root,
            check=True,
        )

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
