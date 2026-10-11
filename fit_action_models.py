"""Fit pooled action models from reusable latent-planning caches on Torch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from modules.action_fitting import (
    cognitive_grid_diagnostics, evaluate_test, fit_all_candidates, freeze_participant_split,
    posterior_predictive_checks, repeat_score_diagnostics,
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache", nargs="+", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-split-only", action="store_true")
    parser.add_argument("--scope", default="primary")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--profile-points", type=int, default=101)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    records = json.loads(args.data.read_text())
    split = freeze_participant_split(records, seed=args.seed)
    if args.split.exists():
        existing = json.loads(args.split.read_text())
        if existing != split:
            parser.error("existing participant split differs from data, seed, or method")
    else:
        write_new_json(args.split, split)
    print(f"frozen_split={args.split} counts={split['participant_counts']} hash={split['split_sha256']}", flush=True)
    if args.freeze_split_only:
        return
    if not args.cache or not args.candidates or not args.output:
        parser.error("fitting requires --cache, --candidates, and --output")
    if args.output.exists():
        parser.error("output already exists")
    from modules.action_cache_fitting import stream_cache_candidates

    candidate_manifest = json.loads(args.candidates.read_text())
    candidate_manifest = {**candidate_manifest, "candidates": [
        candidate for candidate in candidate_manifest["candidates"] if candidate.get("scope", "primary") == args.scope
    ]}
    if not candidate_manifest["candidates"]:
        parser.error("no candidates in selected scope")
    manifest, candidates, observed, repeat_candidates = stream_cache_candidates(args.cache, candidate_manifest)
    record_by_key = {(str(r["pid"]), int(r["trial_index"])): r for r in records}
    cache_keys = [(str(t["pid"]), int(t["trial_index"])) for t in manifest["trials"]]
    if set(cache_keys) != set(record_by_key):
        parser.error("cache trials differ from frozen complete data")
    ordered_records = [record_by_key[key] for key in cache_keys]
    for record, trial in zip(ordered_records, manifest["trials"]):
        problem = {key: record[key] for key in ("graph", "start", "rewards")}
        problem_hash = hashlib.sha256(json.dumps(problem, sort_keys=True).encode()).hexdigest()
        if problem_hash != trial["problem_sha256"]:
            raise ValueError("cache problem hash differs from complete data")
        if record["moves"] != trial["paths"][trial["observed_index"]]:
            raise ValueError("cache observed path differs from complete data")
    assignment = split["participant_assignment"]
    indices = {name: [i for i, record in enumerate(ordered_records) if assignment[str(record["pid"])] == name]
               for name in ("train", "validation", "test")}
    print(f"candidates={len(candidates)} trials={len(observed)}", flush=True)
    fitted = fit_all_candidates(candidates, observed, indices, profile_points=args.profile_points)
    random_probability = np.asarray([2.0**-len(record["moves"]) for record in ordered_records])
    report = {
        "method": "pooled_action_only_finite_cognitive_grid_continuous_movement_lapse",
        "objective": "sum_trial_log_probability_equal_trial_weight",
        "checkpoint_seed_rule": "average_path_probabilities_before_logs_equal_checkpoint_weight",
        "split": split, "scope": args.scope, "fits": fitted,
        "test": evaluate_test(candidates, fitted, observed, indices, [r["pid"] for r in ordered_records],
                              random_probability, bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed),
        "grid_diagnostics": cognitive_grid_diagnostics(candidates, fitted),
        "predictive_checks": posterior_predictive_checks(
            candidates, fitted, observed, ordered_records, indices["test"],
            bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed,
        ),
        "numerical_precision": repeat_score_diagnostics(candidates, repeat_candidates, fitted, observed, indices),
        "provenance": {
            "data_file": str(args.data.resolve()), "data_sha256": file_hash(args.data),
            "candidate_manifest": str(args.candidates.resolve()), "candidate_manifest_sha256": file_hash(args.candidates),
            "cache_manifests": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in args.cache],
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {path: file_hash(path) for path in ["fit_action_models.py", "modules/action_fitting.py", "modules/action_cache_fitting.py"]},
        },
    }
    write_new_json(args.output, report)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
