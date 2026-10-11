"""Estimate action-only path likelihoods for frozen checkpoints on fixed problems."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from modules.a2c import load_jax_params
from modules.compat import assert_compat_version, get_compat_version
from modules.evaluation import (
    env_from_run_args, env_params_from_run_args, read_metadata, resolve_params_path_from_metadata,
)
from modules.path_likelihood import PathLikelihoodEstimator, TreeProblem


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--data", required=True, type=Path, help="JSON list of graph/start/rewards/moves records, zero indexed",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New JSON output; existing files are not overwritten",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.repeats < 1 or args.limit is not None and args.limit < 1:
        parser.error("repeats and limit must be positive")
    if not 2 <= args.samples < 2**32 or args.batch_size < 1:
        parser.error("samples must be in [2, 2**32) and batch-size must be positive")
    records = json.loads(args.data.read_text())
    if not isinstance(records, list) or not records:
        parser.error("data must be a nonempty JSON list")
    if args.limit is not None:
        records = records[:args.limit]
    trials = []
    for record in records:
        problem = TreeProblem.from_record(record)
        observed = tuple(record["moves"])
        if observed not in problem.paths:
            raise ValueError("moves must be a complete root-to-leaf path, excluding the root")
        trials.append((record, problem, problem.paths.index(observed)))

    output = {
        "method": "stopping_state_mc_exact_execution", "data_file": str(args.data.resolve()),
        "data_sha256": file_hash(args.data), "samples": args.samples, "batch_size": args.batch_size,
        "repeats": args.repeats, "seed": args.seed, "trial_count": len(trials),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runs": [],
    }
    for run_dir in args.run_dirs:
        metadata = read_metadata(str(run_dir))
        assert_compat_version(get_compat_version(metadata), source=str(run_dir))
        run_args = metadata["args"]
        env = env_from_run_args(run_args)
        env_params = env_params_from_run_args(env, run_args)
        undirected = run_args.get("lesion_policy") == "random_search_gamma_stopping"
        params_path = None if undirected else resolve_params_path_from_metadata(str(run_dir), metadata)
        params = {} if undirected else load_jax_params(
            params_path, expected_compat_version=get_compat_version(metadata),
        )
        estimator = PathLikelihoodEstimator(env, undirected=undirected)
        run = {
            "run_dir": str(run_dir.resolve()), "metadata": metadata,
            "metadata_sha256": file_hash(run_dir / "metadata.json"),
            "params_file": params_path, "params_sha256": None if undirected else file_hash(params_path),
            "trials": [],
        }
        for trial_index, (record, problem, observed_index) in enumerate(trials):
            estimates = []
            for repeat in range(args.repeats):
                seed = args.seed + trial_index * args.repeats + repeat
                started = time.perf_counter()
                result = estimator.estimate(
                    problem, params, env_params, samples=args.samples, batch_size=args.batch_size, seed=seed,
                )
                probability = result["probabilities"][observed_index]
                se = result["standard_errors"][observed_index]
                result.update({
                    "seed": seed, "elapsed_seconds": time.perf_counter() - started,
                    "observed_probability": probability, "observed_standard_error": se,
                    "observed_log_probability": float(np.log(probability)) if probability > 0 else None,
                    "delta_log_standard_error": se / probability if probability > 0 else None,
                    "zero_estimate": probability == 0,
                })
                estimates.append(result)
            run["trials"].append({
                "pid": record.get("pid"), "trial_index": record.get("trial_index", trial_index),
                "moves": list(problem.paths[observed_index]), "estimates": estimates,
            })
            print(f"run={run_dir.name} trial={trial_index + 1}/{len(trials)} estimates={len(estimates)}", flush=True)
        output["runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
