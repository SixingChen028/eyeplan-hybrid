"""Cache action-only path distributions for all execution lapse rates in one MC pass."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from modules.a2c import load_jax_params
from modules.compat import assert_compat_version, get_compat_version
from modules.evaluation import (
    env_from_run_args, env_params_from_run_args, read_metadata, resolve_params_path_from_metadata,
)
from modules.path_likelihood import PathLikelihoodEstimator, TreeProblem
from modules.path_likelihood_cache import CACHE_VERSION


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Exclusive JSON manifest, with companion NPZ")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    array_path = args.output.with_suffix(".npz")
    if args.output.suffix != ".json" or args.output.exists() or array_path.exists():
        parser.error("output must be a new .json path and its companion .npz must not exist")
    if args.repeats < 1 or args.trial_start < 0 or args.limit is not None and args.limit < 1:
        parser.error("repeats and limit must be positive; trial-start must be nonnegative")
    if not 2 <= args.samples < 2**32 or args.batch_size < 1:
        parser.error("samples must be in [2, 2**32) and batch-size must be positive")
    records = json.loads(args.data.read_text())
    if not isinstance(records, list) or not records:
        parser.error("data must be a nonempty JSON list")
    stop = None if args.limit is None else args.trial_start + args.limit
    selected = list(enumerate(records))[args.trial_start:stop]
    if not selected:
        parser.error("selected trial range is empty")
    trials = []
    trial_metadata = []
    for index, record in selected:
        problem = TreeProblem.from_record(record)
        observed = tuple(record["moves"])
        if observed not in problem.paths:
            raise ValueError("moves must be a complete root-to-leaf path excluding the root")
        canonical = {key: record[key] for key in ("graph", "start", "rewards")}
        problem_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        trials.append(problem)
        trial_metadata.append({
            "record_index": index, "pid": record.get("pid"), "trial_index": record.get("trial_index", index),
            "paths": [list(path) for path in problem.paths], "observed_index": problem.paths.index(observed),
            "problem_sha256": problem_hash,
        })
    source_files = [
        "lapse_likelihood_cache.py", "modules/path_likelihood.py", "modules/path_likelihood_cache.py",
        "modules/environment.py", "modules/random_search.py", "modules/network.py", "modules/evaluation.py",
        "modules/a2c.py", "modules/compat.py", "modules/config.py",
    ]
    source_root = Path(__file__).resolve().parent
    output = {
        "cache_version": CACHE_VERSION, "method": "planning_mc_bernstein_moments_exact_execution",
        "arrays_file": array_path.name, "data_file": str(args.data.resolve()), "data_sha256": file_hash(args.data),
        "samples": args.samples, "batch_size": args.batch_size, "repeats": args.repeats, "seed": args.seed,
        "trial_start": args.trial_start, "trial_count": len(trials), "trials": trial_metadata,
        "lapse_rule": "independent_uniform_legal_child_at_each_movement_step",
        "rng_rule": "PRNGKey(uint32_sha256(base:checkpoint_identity:record_index:repeat)); fold_in(sample_index)",
        "jax_version": jax.__version__, "jax_backend": jax.default_backend(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: file_hash(source_root / name) for name in source_files}, "runs": [],
    }
    arrays = {}
    for run_index, run_dir in enumerate(args.run_dirs):
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
        metadata_hash = file_hash(run_dir / "metadata.json")
        params_hash = None if undirected else file_hash(params_path)
        checkpoint_identity = metadata_hash + ":" + str(params_hash)
        run = {
            "run_dir": str(run_dir.resolve()), "metadata": metadata,
            "metadata_sha256": metadata_hash, "checkpoint_identity": checkpoint_identity,
            "params_file": params_path, "params_sha256": params_hash,
            "estimates": [],
        }
        for trial_index, problem in enumerate(trials):
            estimates = []
            for repeat in range(args.repeats):
                # Independent checkpoint streams; stable when run/trial/repetition shards change.
                seed_text = f"{args.seed}:{checkpoint_identity}:{selected[trial_index][0]}:{repeat}"
                seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "little")
                started = time.perf_counter()
                cache = estimator.estimate_lapse_cache(
                    problem, params, env_params, samples=args.samples, batch_size=args.batch_size, seed=seed,
                )
                key = f"r{run_index}_t{trial_index}_s{repeat}"
                arrays[key + "_mean"] = cache.mean_coefficients
                arrays[key + "_m2"] = cache.coefficient_m2
                estimates.append({
                    "key": key, "samples": cache.samples, "seed": cache.seed,
                    "mean_latent_fixations": cache.mean_latent_fixations, "horizon_fraction": cache.horizon_fraction,
                    "elapsed_seconds": time.perf_counter() - started,
                })
            run["estimates"].append(estimates)
            print(f"run={run_dir.name} record={selected[trial_index][0]} repeats={len(estimates)}", flush=True)
        output["runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with array_path.open("xb") as file:
        np.savez_compressed(file, **arrays)
    output["arrays_sha256"] = file_hash(array_path)
    with args.output.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} arrays={array_path}", flush=True)


if __name__ == "__main__":
    main()
