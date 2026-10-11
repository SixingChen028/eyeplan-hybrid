"""Cache undirected planning trajectories for action-only Gamma stopping and lapse fits."""

import argparse
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from modules.compat import assert_compat_version, get_compat_version
from modules.evaluation import env_from_run_args, env_params_from_run_args, read_metadata
from modules.path_likelihood import TreeProblem
from modules.undirected_likelihood_cache import CACHE_VERSION, UndirectedTrajectoryEstimator


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=256, help="Independent trajectories per repetition")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wm-decay", type=float, nargs="+")
    parser.add_argument("--wm-neighbor-activation", type=float, nargs="+")
    args = parser.parse_args()
    arrays_path = args.output.with_suffix(".npz")
    if args.output.exists() or arrays_path.exists():
        parser.error("output manifest or arrays already exist")
    if not 2 <= args.samples < 2**32 or args.batch_size < 1 or args.repeats < 2:
        parser.error("samples must be in [2, 2**32), batch-size positive, and repeats at least two")
    if args.trial_start < 0 or args.limit is not None and args.limit < 1:
        parser.error("trial-start must be nonnegative and limit must be positive")
    records = json.loads(args.data.read_text())
    if not isinstance(records, list) or not records:
        parser.error("data must be a nonempty JSON list")
    total_records = len(records)
    end = None if args.limit is None else args.trial_start + args.limit
    records = records[args.trial_start:end]
    if not records:
        parser.error("selected trial range is empty")
    problems = [TreeProblem.from_record(record) for record in records]
    trials = []
    for index, (record, problem) in enumerate(zip(records, problems, strict=True), start=args.trial_start):
        observed = tuple(record["moves"])
        if observed not in problem.paths:
            parser.error("moves must be a complete root-to-leaf path excluding the root")
        trials.append({"pid": record.get("pid"), "trial_index": record.get("trial_index", index),
                       "moves": list(observed), "paths": [list(path) for path in problem.paths],
                       "observed_index": problem.paths.index(observed)})
    manifest = {
        "cache_version": CACHE_VERSION, "method": "undirected_gamma_trajectory_cache",
        "data_file": str(args.data.resolve()), "data_sha256": file_hash(args.data),
        "samples_per_repeat": args.samples, "repeats": args.repeats, "batch_size": args.batch_size,
        "seed": args.seed, "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: file_hash(Path(__file__).resolve().parent / path) for path in (
            "undirected_likelihood_cache.py", "modules/undirected_likelihood_cache.py",
            "modules/path_likelihood.py", "modules/environment.py", "modules/random_search.py",
        )},
        "trial_start": args.trial_start, "trial_count": len(records),
        "trials": trials, "runs": [], "arrays_file": arrays_path.name,
        "stopping_rule": "max(min(floor(Gamma(shape,scale)),min(t_max,100))-1,0)",
        "fixation_rule": "uniform_legal_including_immediate_repeats",
        "uncertainty": "standard_error_across_independent_trajectory_repeat_means",
    }
    arrays = {}
    for run_dir in args.run_dirs:
        metadata = read_metadata(str(run_dir))
        assert_compat_version(get_compat_version(metadata), source=str(run_dir))
        run_args = metadata["args"]
        if run_args.get("lesion_policy") != "random_search_gamma_stopping":
            parser.error("run directories must specify lesion_policy=random_search_gamma_stopping")
        env = env_from_run_args(run_args)
        estimator = UndirectedTrajectoryEstimator(env)
        decays = args.wm_decay if args.wm_decay is not None else [run_args["wm_decay"]]
        neighbors = args.wm_neighbor_activation if args.wm_neighbor_activation is not None else [
            run_args["wm_neighbor_activation"],
        ]
        for decay, neighbor in itertools.product(decays, neighbors):
            params_args = {**run_args, "wm_decay": decay, "wm_neighbor_activation": neighbor}
            params = env_params_from_run_args(env, params_args)
            run_index = len(manifest["runs"])
            run = {"run_dir": str(run_dir.resolve()), "metadata": metadata,
                   "metadata_sha256": file_hash(run_dir / "metadata.json"),
                   "wm_decay": decay, "wm_neighbor_activation": neighbor,
                   "max_fixations": estimator.max_fixations, "estimates": []}
            for trial_index, problem in enumerate(problems):
                started = time.perf_counter()
                seed = args.seed + (run_index * total_records + args.trial_start + trial_index) * args.repeats
                cache = estimator.build_cache(problem, params, samples=args.samples, repeats=args.repeats,
                                              batch_size=args.batch_size, seed=seed)
                key = f"run{run_index}_trial{trial_index}"
                arrays[key] = cache.mean_coefficients
                elapsed = time.perf_counter() - started
                run["estimates"].append({"key": key, "samples_per_repeat": cache.samples_per_repeat,
                                         "seeds": list(cache.seeds), "elapsed_seconds": elapsed})
                print(f"run={run_index} trial={trial_index + 1}/{len(problems)} seconds={elapsed:.3f}", flush=True)
            manifest["runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open("xb") as file:
        np.savez_compressed(file, **arrays)
    manifest["arrays_sha256"] = file_hash(arrays_path)
    with args.output.open("x") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
