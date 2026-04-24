import argparse
import json
import os
import sys
import traceback

from modules.a2c import load_jax_params
from modules.analysis_targets import resolve_analysis_target, select_most_recent_run
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


def _resolve_run_dir(target: str, results_root: str) -> tuple[str, str]:
    resolved = resolve_analysis_target(target, results_root=results_root)
    if resolved.kind == "run":
        return resolved.run_dirs[0], resolved.experiment
    return select_most_recent_run(resolved.run_dirs), resolved.experiment


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        type=str,
        help="Target run path or experiment (uses most recent run for an experiment).",
    )
    parser.add_argument("--results_root", type=str, default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("--num_trials", type=int, default=10_240)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--include_timeout_trials", action="store_true")
    args = parser.parse_args()

    had_error = False
    try:
        run_dir, experiment = _resolve_run_dir(args.target, results_root=args.results_root)
        print(f"target={args.target} experiment={experiment}")
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
            num_trials=args.num_trials,
            greedy=args.greedy,
        )

        transformed = to_transformed_simulation_format(
            data,
            num_nodes=env.num_nodes,
            t_max=env.t_max,
            skip_timeout_trials=not args.include_timeout_trials,
        )

        output_path = args.output
        if output_path == "":
            output_path = os.path.join(run_dir, "data_simulation.json")
        output_path = os.path.abspath(os.path.expanduser(output_path))

        with open(output_path, "w") as file:
            json.dump(transformed, file)
            file.write("\n")

        print(f"output_json={output_path}")
        print(f"num_trials_raw={len(data['action_seqs'])}")
        print(f"num_trials_exported={len(transformed['actions'])}")
    except Exception:
        had_error = True
        traceback.print_exc(file=sys.stderr)

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
