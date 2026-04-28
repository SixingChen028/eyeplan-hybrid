import argparse
import json
import os
import pickle
import sys
import traceback

from modules.a2c import load_jax_params
from modules.analysis_targets import resolve_analysis_target
from modules.environment import JaxDecisionTreeEnv
from modules.simulation import JaxSimulator, to_transformed_simulation_format


def _read_metadata(run_dir: str) -> dict:
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")

    with open(metadata_path, "r") as file:
        return json.load(file)


def _read_metadata_args(run_dir: str) -> dict:
    metadata = _read_metadata(run_dir)
    args = metadata.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"Invalid metadata args payload: {os.path.join(run_dir, 'metadata.json')}")
    return args


def _build_env_from_metadata_args(metadata_args: dict) -> JaxDecisionTreeEnv:
    return JaxDecisionTreeEnv(
        num_nodes=int(metadata_args.get("num_nodes", 15)),
        beta_move=float(metadata_args.get("beta_move", 40.0)),
        eps_move=float(metadata_args.get("eps_move", 0.0)),
        learning_rate=float(metadata_args.get("learning_rate", 1.0)),
        lamda_backup=float(metadata_args.get("lamda_backup", 1.0)),
        wm_decay=float(metadata_args.get("wm_decay", 1.0)),
        t_max=int(metadata_args.get("t_max", 100)),
        cost=float(metadata_args.get("cost", 0.01)),
        scale_factor=float(metadata_args.get("scale_factor", 1 / 8)),
        shuffle_nodes=bool(metadata_args.get("shuffle_nodes", True)),
        use_recency_obs=bool(metadata_args.get("use_recency_obs", False)),
    )


def _resolve_params_path_from_metadata(run_dir: str, metadata: dict) -> str:
    # Prefer explicit metadata field when available.
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
        preferred = os.path.join(run_dir, "net_jax.p")

    if os.path.exists(preferred):
        return preferred

    available = []
    for name in ("net_jax.p", "net_jax_ppo.p"):
        candidate = os.path.join(run_dir, name)
        if os.path.exists(candidate):
            available.append(candidate)

    if len(available) == 1:
        return available[0]

    raise FileNotFoundError(
        f"Unable to resolve model params file from metadata for run: {run_dir}. "
        f"preferred={preferred} available={available}"
    )


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def _expected_num_updates(metadata_args: dict) -> int:
    num_episodes = int(metadata_args.get("num_episodes", 0))
    batch_size = int(metadata_args.get("batch_size", 0))
    if batch_size <= 0:
        return 0
    return num_episodes // batch_size


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
    greedy: bool,
    skip_timeout_trials: bool,
    detailed: bool,
) -> tuple[int, int]:
    print(f"run_dir={run_dir}")

    metadata = _read_metadata(run_dir)
    metadata_args = _read_metadata_args(run_dir)
    env = _build_env_from_metadata_args(metadata_args)

    params_path = _resolve_params_path_from_metadata(run_dir, metadata)
    params = load_jax_params(params_path)
    print(f"params_path={params_path}")

    seed = int(metadata_args.get("seed", 15))
    print(f"seed={seed}")
    simulator = JaxSimulator(env)
    data = simulator.simulate(
        params=params,
        seed=seed,
        num_trials=num_trials,
        greedy=greedy,
        detailed=detailed,
    )

    transformed = to_transformed_simulation_format(
        data,
        num_nodes=env.num_nodes,
        t_max=env.t_max,
        skip_timeout_trials=skip_timeout_trials,
        detailed=detailed,
    )

    with open(output_path, "w") as file:
        json.dump(_round_floats(transformed), file)
        file.write("\n")

    return len(data["action_seqs"]), len(transformed["actions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="+",
        type=str,
        help="One or more targets: <experiment>, <experiment>/<run_id>, <experiment>/*, or full path in runs/analysis.",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--skip_timeout_trials", action="store_true")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--seed-filter", type=int, default=None)
    args = parser.parse_args()
    num_trials = args.num_trials
    if num_trials is None:
        num_trials = 100 if args.detailed else 10_240

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

    for experiment, run_dir in runs_to_simulate:
        try:
            print(f"experiment={experiment}")
            output_path = args.output
            if output_path == "":
                output_name = "data_simulation_detailed.json" if args.detailed else "data_simulation.json"
                output_path = os.path.join(run_dir, output_name)
            output_path = os.path.abspath(os.path.expanduser(output_path))

            num_trials_raw, num_trials_exported = _simulate_run(
                run_dir=run_dir,
                output_path=output_path,
                num_trials=num_trials,
                greedy=args.greedy,
                skip_timeout_trials=args.skip_timeout_trials,
                detailed=args.detailed,
            )

            print(f"output_json={output_path}")
            print(f"num_trials_raw={num_trials_raw}")
            print(f"num_trials_exported={num_trials_exported}")
        except Exception:
            had_error = True
            print(f"Error simulating run: {run_dir}", file=sys.stderr)
            traceback.print_exc()
            continue

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
