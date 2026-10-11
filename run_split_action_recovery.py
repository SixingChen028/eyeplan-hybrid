"""Replay the empirical fitting protocol using separately cached generating histories."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from action_recovery import cache_repeat_count, cache_trial_metadata
from modules.action_cache_fitting import align_candidate_trials, stream_cache_candidates, stream_undirected_candidates
from modules.action_fitting import Candidate, canonical_hash
from modules.action_recovery_rng import assert_independent_manifest_rng
from split_action_recovery import split_recovery


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_provenance(paths):
    return [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in paths]


def selected_run_dirs(paths, inventory):
    """Resolve frozen relative checkpoint names to their actual cached identities."""
    expected = {str(Path(run["run_dir"])) for candidate in inventory["candidates"] for run in candidate["runs"]}
    selected = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        for run in manifest["runs"]:
            actual = str(Path(run["run_dir"]))
            if any(actual == name or actual.endswith("/" + name) for name in expected):
                selected.add(actual)
    return sorted(selected)


def validate_frozen_design(trials, split, fitted):
    if fitted["split"] != split:
        raise ValueError("empirical fit and recovery frozen splits differ")
    if canonical_hash({key: value for key, value in split.items() if key != "split_sha256"}) != split["split_sha256"]:
        raise ValueError("frozen participant split hash mismatch")
    original = sorted(trials, key=lambda trial: trial["record_index"])
    if [trial["record_index"] for trial in original] != list(range(len(original))):
        raise ValueError("complete frozen designs need unique contiguous original record indices")
    keys = [[str(trial["pid"]), int(trial["trial_index"])] for trial in original]
    if canonical_hash(keys) != split["trial_keys_sha256"]:
        raise ValueError("cached task identities differ from the frozen split trial-key hash")
    assignment = split["participant_assignment"]
    if set(assignment) != {str(trial["pid"]) for trial in trials}:
        raise ValueError("frozen split participants differ from cached task designs")
    for name, indices in split["indices"].items():
        expected = [index for index, trial in enumerate(original) if assignment[str(trial["pid"])] == name]
        if indices != expected or len(indices) != split["trial_counts"][name]:
            raise ValueError("frozen split trial indices or counts differ from cached task designs")


def generating_fits(report):
    fitted = {fit["candidate_id"]: fit for fit in report["fits"]["fits"]}
    selected = report["fits"]["selected_by_validation"]
    result = []
    for family in ("main", "unbounded", "undirected"):
        fit = fitted[selected[family]]
        if fit["family"] != family or not 0.01 <= fit["lapse_rate"] <= 1:
            raise ValueError("selected empirical truth has an invalid family or lapse")
        result.append(fit)
    return result


def matching_parameters(first, second):
    return first.keys() == second.keys() and all(
        np.isclose(first[key], second[key], atol=1e-12, rtol=1e-10)
        if isinstance(first[key], (int, float)) and isinstance(second[key], (int, float))
        else first[key] == second[key] for key in first
    )


def precision_metadata(paths, *, run_dirs=None, wm_laws=None):
    counts = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        for run in manifest["runs"]:
            if run_dirs is not None and run["run_dir"] not in run_dirs:
                continue
            if wm_laws is not None and (run["wm_decay"], run["wm_neighbor_activation"]) not in wm_laws:
                continue
            for entry in run["estimates"]:
                if isinstance(entry, list):
                    counts.add(tuple(repeat["samples"] for repeat in entry))
                else:
                    samples = entry.get("samples_per_repeat", manifest.get("samples_per_repeat"))
                    counts.add((samples,) * len(entry["seeds"]))
    return {"raw_repeat_count": cache_repeat_count(paths),
            "sample_counts": [list(count) for count in sorted(counts, key=str)],
            "pooled_samples_per_checkpoint_or_wm_law": sorted({sum(count) for count in counts}),
            "pooled_all_repeats": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True, type=Path)
    parser.add_argument("--undirected-cache", nargs="+", required=True, type=Path)
    parser.add_argument("--truth-cache", nargs="+", required=True, type=Path)
    parser.add_argument("--truth-undirected-cache", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--fit-results", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--repetitions", default=20, type=int)
    parser.add_argument("--profile-points", default=101, type=int)
    parser.add_argument("--optimizer-maxiter", default=300, type=int)
    parser.add_argument("--seed", default=20261011, type=int)
    parser.add_argument("--gamma-shapes", nargs="+", type=float, default=[.5, 1, 2, 4, 8])
    parser.add_argument("--gamma-means", nargs="+", type=float, default=[2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.repetitions < 1 or args.profile_points < 3 or args.optimizer_maxiter < 1:
        parser.error("repetitions/maxiter must be positive and profile-points at least three")
    report = json.loads(args.fit_results.read_text())
    split = json.loads(args.split.read_text())
    if report["provenance"]["candidate_manifest_sha256"] != file_hash(args.candidate_manifest):
        raise ValueError("candidate inventory differs from the empirical fit's frozen inventory")
    baseline_hashes = {file_hash(path) for path in args.cache + args.undirected_cache}
    if baseline_hashes != {entry["sha256"] for entry in report["provenance"]["cache_manifests"]}:
        raise ValueError("baseline cache manifests differ from the empirical fit's pooled MC caches")
    inventory = json.loads(args.candidate_manifest.read_text())
    inventory = {**inventory, "candidates": [candidate for candidate in inventory["candidates"]
                 if candidate.get("scope", "primary") == "primary" and candidate["model"] != "undirected"]}
    truths = generating_fits(report)
    learned_truth_ids = {fit["candidate_id"] for fit in truths if fit["family"] != "undirected"}
    truth_inventory = {**inventory, "candidates": [candidate for candidate in inventory["candidates"]
                                                  if candidate["id"] in learned_truth_ids]}
    if {candidate["id"] for candidate in truth_inventory["candidates"]} != learned_truth_ids:
        raise ValueError("selected learned generating candidates are absent from the primary frozen inventory")
    learned_specs = {candidate["id"]: candidate for candidate in truth_inventory["candidates"]}
    for fit in truths:
        if fit["family"] != "undirected":
            specification = learned_specs[fit["candidate_id"]]
            if specification["model"] != fit["family"] or not matching_parameters(
                specification["parameters"], fit["parameters"],
            ):
                raise ValueError("selected learned truth parameters differ from the frozen candidate inventory")
    gamma_truth = next(fit for fit in truths if fit["family"] == "undirected")
    if not any(matching_parameters(spec["parameters"], gamma_truth["parameters"])
               for spec in report["refined_undirected_candidate_specs"]):
        raise ValueError("selected Gamma truth parameters are absent from the empirical refined candidate specs")
    parameters = gamma_truth["parameters"]
    truth_law = (parameters["wm_decay"], parameters["wm_neighbor_activation"])
    custom_specs = {truth_law: [{"gamma_shape": parameters["gamma_shape"], "gamma_mean": parameters["gamma_mean"]}]}
    trials = cache_trial_metadata(args.cache)
    for paths in (args.undirected_cache, args.truth_cache, args.truth_undirected_cache):
        if cache_trial_metadata(paths) != trials:
            raise ValueError("baseline and generating cache task identities/path layouts differ")
    validate_frozen_design(trials, split, report)
    fit_paths = args.cache + args.undirected_cache
    truth_paths = args.truth_cache + args.truth_undirected_cache
    fit_runs = selected_run_dirs(args.cache, inventory)
    truth_runs = selected_run_dirs(args.truth_cache, truth_inventory)
    audit = assert_independent_manifest_rng(fit_paths, truth_paths, fit_run_dirs=fit_runs,
                                             truth_run_dirs=truth_runs, truth_wm_laws=[truth_law])
    print(json.dumps({"stage": "rng_audit", "trials": len(trials), "audit": audit}), flush=True)
    manifest, candidates, _, repeats = stream_cache_candidates(args.cache, inventory)
    del repeats
    raw_repeats = cache_repeat_count(args.undirected_cache)
    gamma_manifest, gamma_candidates, repeats = stream_undirected_candidates(
        args.undirected_cache, args.gamma_shapes, args.gamma_means, repeat_groups=[list(range(raw_repeats))],
    )
    del repeats
    candidates += align_candidate_trials(gamma_candidates, gamma_manifest["trials"], manifest["trials"])
    truth_manifest, truth_candidates, _, repeats = stream_cache_candidates(args.truth_cache, truth_inventory)
    del repeats
    truth_candidates = align_candidate_trials(truth_candidates, truth_manifest["trials"], manifest["trials"])
    raw_truth_repeats = cache_repeat_count(args.truth_undirected_cache)
    gamma_manifest, gamma_candidates, repeats = stream_undirected_candidates(
        args.truth_undirected_cache, candidate_specs_by_law=custom_specs,
        repeat_groups=[list(range(raw_truth_repeats))],
    )
    del repeats
    gamma_candidates = align_candidate_trials(gamma_candidates, gamma_manifest["trials"], manifest["trials"])
    if len(gamma_candidates) != 1 or not matching_parameters(gamma_candidates[0].parameters, parameters):
        raise ValueError("generating Gamma integration differs from selected empirical parameters")
    gamma = gamma_candidates[0]
    truth_candidates.append(Candidate(gamma_truth["candidate_id"], "undirected", parameters,
                                      gamma.evaluate, gamma.evaluate_subset, gamma.evaluate_observed))
    result = split_recovery(
        candidates, truth_candidates, manifest["trials"], split,
        truth_lapses={fit["candidate_id"]: fit["lapse_rate"] for fit in truths}, repetitions=args.repetitions,
        seed=args.seed, profile_points=args.profile_points, continuous_gamma_paths=args.undirected_cache,
        optimizer_maxiter=args.optimizer_maxiter, checkpoint_dir=args.checkpoint_dir,
        progress=lambda value: print(json.dumps(value), flush=True),
    )
    result["provenance"] = {
        "cache_manifests": manifest_provenance(fit_paths), "truth_cache_manifests": manifest_provenance(truth_paths),
        "candidate_manifest_sha256": file_hash(args.candidate_manifest),
        "fit_results_sha256": file_hash(args.fit_results),
        "split_file_sha256": file_hash(args.split), "rng_audit": audit,
        "full_pooled_precision": {
            "fitting_learned": precision_metadata(args.cache, run_dirs=fit_runs),
            "fitting_undirected": precision_metadata(args.undirected_cache),
            "generating_learned": precision_metadata(args.truth_cache, run_dirs=truth_runs),
            "generating_undirected": precision_metadata(args.truth_undirected_cache, wm_laws=[truth_law]),
        },
        "generating_fits": truths,
        "gamma_initialization_grid": {"shapes": args.gamma_shapes, "means": args.gamma_means},
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: file_hash(path) for path in [
            "run_split_action_recovery.py", "split_action_recovery.py", "modules/action_fitting.py",
            "modules/action_cache_fitting.py", "modules/action_recovery_rng.py",
            "modules/undirected_likelihood_cache.py",
        ]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} datasets={len(result['datasets'])}", flush=True)


if __name__ == "__main__":
    main()
