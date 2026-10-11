"""Fit pooled action models from reusable latent-planning caches on Torch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from modules.action_fitting import (
    cognitive_grid_diagnostics, evaluate_test, fit_all_candidates, freeze_participant_split,
    independent_repeat_groups, posterior_predictive_checks, repeat_score_diagnostics, select_fitted_candidates,
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
    parser.add_argument("--undirected-cache", nargs="+", type=Path)
    parser.add_argument("--gamma-shapes", nargs="+", type=float, default=[.5, 1, 2, 4, 8])
    parser.add_argument("--gamma-means", nargs="+", type=float, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--coarse-gamma-only", action="store_true", help="development diagnostic; skip continuous refinement",
    )
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
    if not (args.cache or args.undirected_cache) or not args.candidates or not args.output:
        parser.error("fitting requires --cache and/or --undirected-cache, --candidates, and --output")
    if args.output.exists():
        parser.error("output already exists")
    from modules.action_cache_fitting import (
        align_candidate_trials, stream_cache_candidates, stream_undirected_candidates,
    )

    data_hash = file_hash(args.data)
    for path in (args.cache or []) + (args.undirected_cache or []):
        if json.loads(path.read_text())["data_sha256"] != data_hash:
            raise ValueError("cache data hash differs from complete fitting data")
    candidate_manifest = json.loads(args.candidates.read_text())
    candidate_manifest = {**candidate_manifest, "candidates": [
        candidate for candidate in candidate_manifest["candidates"] if candidate.get("scope", "primary") == args.scope
    ]}
    if not candidate_manifest["candidates"]:
        parser.error("no candidates in selected scope")
    if args.undirected_cache:
        candidate_manifest["candidates"] = [candidate for candidate in candidate_manifest["candidates"]
                                             if candidate["model"] != "undirected"]
    if args.cache:
        manifest, candidates, observed, repeat_candidates = stream_cache_candidates(args.cache, candidate_manifest)
    else:
        manifest, candidates, observed, repeat_candidates = None, [], None, []
    repeat_grouping = {"ordinary": [[r] for r in range(len(repeat_candidates))]} if args.cache else {}
    if args.undirected_cache:
        first_gamma_manifest = json.loads(args.undirected_cache[0].read_text())
        raw_repeat_count = len(first_gamma_manifest["runs"][0]["estimates"][0]["seeds"])
        fold_count = len(repeat_candidates) if args.cache else raw_repeat_count
        repeat_grouping["undirected"] = independent_repeat_groups(raw_repeat_count, fold_count)
        undirected_manifest, undirected_candidates, undirected_repeats = stream_undirected_candidates(
            args.undirected_cache, args.gamma_shapes, args.gamma_means,
            repeat_groups=repeat_grouping["undirected"],
        )
        if manifest is None:
            manifest = undirected_manifest
            observed = np.asarray([trial["observed_index"] for trial in manifest["trials"]])
            candidates, repeat_candidates = undirected_candidates, undirected_repeats
        else:
            candidates += align_candidate_trials(
                undirected_candidates, undirected_manifest["trials"], manifest["trials"],
            )
            for ordinary, undirected in zip(repeat_candidates, undirected_repeats, strict=True):
                ordinary += align_candidate_trials(undirected, undirected_manifest["trials"], manifest["trials"])
            del ordinary, undirected
    record_by_key = {(str(r["pid"]), int(r["trial_index"])): r for r in records}
    cache_keys = [(str(t["pid"]), int(t["trial_index"])) for t in manifest["trials"]]
    if set(cache_keys) != set(record_by_key):
        parser.error("cache trials differ from frozen complete data")
    ordered_records = [record_by_key[key] for key in cache_keys]
    for record, trial in zip(ordered_records, manifest["trials"]):
        problem = {key: record[key] for key in ("graph", "start", "rewards")}
        problem_hash = hashlib.sha256(json.dumps(problem, sort_keys=True).encode()).hexdigest()
        if "problem_sha256" in trial and problem_hash != trial["problem_sha256"]:
            raise ValueError("cache problem hash differs from complete data")
        if record["moves"] != trial["paths"][trial["observed_index"]]:
            raise ValueError("cache observed path differs from complete data")
    assignment = split["participant_assignment"]
    indices = {name: [i for i, record in enumerate(ordered_records) if assignment[str(record["pid"])] == name]
               for name in ("train", "validation", "test")}
    print(f"candidates={len(candidates)} trials={len(observed)}", flush=True)

    def progress(current, total, fit):
        print(f"fitted={current}/{total} candidate={fit['candidate_id']} "
              f"train_ll={fit['total_log_likelihood']:.4f} val_ll={fit['validation_log_likelihood']:.4f}", flush=True)

    fitted = fit_all_candidates(candidates, observed, indices, profile_points=args.profile_points, progress=progress)
    coarse_gamma_fits = None
    gamma_refinement = None
    if args.undirected_cache and not args.coarse_gamma_only:
        from modules.action_gamma_fitting import refine_all_gamma

        coarse_gamma_fits = [fit for fit in fitted["fits"] if fit["family"] == "undirected"]
        # Release the large coarse Gamma coefficient grid before continuous refinement.
        candidates = [candidate for candidate in candidates if candidate.family != "undirected"]
        repeat_candidates = [[candidate for candidate in repeated if candidate.family != "undirected"]
                             for repeated in repeat_candidates]
        del undirected_candidates, undirected_repeats

        def refinement_progress(current, total, fit):
            print(f"refined_gamma={current}/{total} decay={fit['wm_decay']} neighbor={fit['wm_neighbor_activation']} "
                  f"train_ll={fit['train_log_likelihood']:.4f}", flush=True)

        refined_specs, gamma_refinement = refine_all_gamma(
            args.undirected_cache, manifest["trials"], fitted, indices["train"], progress=refinement_progress,
        )
        refined_manifest, refined_candidates, refined_repeats = stream_undirected_candidates(
            args.undirected_cache, candidate_specs_by_law=refined_specs,
            repeat_groups=repeat_grouping["undirected"],
        )
        refined_candidates = align_candidate_trials(refined_candidates, refined_manifest["trials"], manifest["trials"])
        new_fitted = fit_all_candidates(refined_candidates, observed, indices,
                                        profile_points=args.profile_points, progress=progress)
        candidates += refined_candidates
        for ordinary, repeated in zip(repeat_candidates, refined_repeats, strict=True):
            ordinary += align_candidate_trials(repeated, refined_manifest["trials"], manifest["trials"])
        fitted = select_fitted_candidates([fit for fit in fitted["fits"] if fit["family"] != "undirected"]
                                          + new_fitted["fits"])
    random_probability = np.asarray([2.0**-len(record["moves"]) for record in ordered_records])
    report = {
        "method": "pooled_action_only_finite_cognitive_grid_continuous_movement_lapse",
        "objective": "sum_trial_log_probability_equal_trial_weight",
        "checkpoint_seed_rule": "average_path_probabilities_before_logs_equal_checkpoint_weight",
        "split": split, "scope": args.scope, "fits": fitted,
        "coarse_gamma_fits": coarse_gamma_fits, "gamma_refinement": gamma_refinement,
        "refined_undirected_candidate_specs": [
            {"id": candidate.candidate_id, "model": candidate.family, "parameters": candidate.parameters}
            for candidate in candidates if candidate.family == "undirected"
        ] if gamma_refinement is not None else [],
        "test": evaluate_test(candidates, fitted, observed, indices, [r["pid"] for r in ordered_records],
                              random_probability, bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed),
        "grid_diagnostics": cognitive_grid_diagnostics(candidates, fitted),
        "predictive_checks": posterior_predictive_checks(
            candidates, fitted, observed, ordered_records, indices["test"],
            bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed,
        ),
        "numerical_precision": repeat_score_diagnostics(
            candidates, repeat_candidates, fitted, observed, indices, repeat_grouping=repeat_grouping,
        ),
        "provenance": {
            "data_file": str(args.data.resolve()), "data_sha256": file_hash(args.data),
            "candidate_manifest": str(args.candidates.resolve()),
            "candidate_manifest_sha256": file_hash(args.candidates),
            "cache_manifests": [{"path": str(path.resolve()), "sha256": file_hash(path)}
                                for path in (args.cache or []) + (args.undirected_cache or [])],
            "numerical_repeat_groups": repeat_grouping,
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {path: file_hash(path) for path in [
                "fit_action_models.py", "modules/action_fitting.py", "modules/action_cache_fitting.py",
                "modules/action_gamma_fitting.py",
            ]},
        },
    }
    write_new_json(args.output, report)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
