"""Confirm frozen selected action-model scores using new independent MC caches."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from action_mc_precision import SPLITS, file_hash, read_json, stream_selected_precision
from modules.action_cache_fitting import stream_undirected_candidates
from modules.action_recovery_rng import assert_independent_manifest_rng, iter_manifest_seeds


def baseline_paths(fit, relocated_paths=None):
    """Resolve all original fit cache manifests, requiring their frozen hashes."""
    expected = fit["provenance"]["cache_manifests"]
    paths = list(map(Path, relocated_paths)) if relocated_paths is not None else [Path(row["path"]) for row in expected]
    by_hash = {file_hash(path): path for path in paths}
    if set(by_hash) != {row["sha256"] for row in expected}:
        raise ValueError("baseline cache manifests do not match the complete frozen fit provenance")
    return [by_hash[row["sha256"]] for row in expected]


def validate_collections(baseline, confirmation, fit, gamma_law):
    """Match full problem/response identities and the selected Gamma source law."""
    trials = {}
    gamma_sources = set()
    source_hashes = {}
    data_hash = fit["provenance"]["data_sha256"]
    for path in baseline:
        manifest = read_json(path)
        if manifest["data_sha256"] != data_hash:
            raise ValueError("baseline cache data hash differs from frozen fit")
        if "source_sha256" in manifest:
            source_hashes.setdefault(manifest.get("method"), set()).add(
                json.dumps(manifest["source_sha256"], sort_keys=True),
            )
        for trial in manifest["trials"]:
            key = str(trial["pid"]), int(trial["trial_index"])
            if key in trials:
                compare_trial(trials[key], trial)
            trials[key] = trial if "problem_sha256" in trial else trials.get(key, trial)
        if manifest.get("method") == "undirected_gamma_trajectory_cache":
            for run in manifest["runs"]:
                if (run["wm_decay"], run["wm_neighbor_activation"]) == gamma_law:
                    gamma_sources.add((run["metadata_sha256"], run["max_fixations"]))
    count = sum(fit["split"]["trial_counts"].values())
    if len(trials) != count:
        raise ValueError("original cache manifests do not cover the frozen complete trial count")
    selected_gamma_found = gamma_law is None
    for path in confirmation:
        manifest = read_json(path)
        if manifest["data_sha256"] != data_hash:
            raise ValueError("new confirmation cache data hash differs from frozen fit")
        if manifest.get("method") in source_hashes and json.dumps(
            manifest.get("source_sha256"), sort_keys=True,
        ) not in source_hashes[manifest.get("method")]:
            raise ValueError("new cache frozen simulator source hashes differ from original fitting caches")
        for trial in manifest["trials"]:
            key = str(trial["pid"]), int(trial["trial_index"])
            if key not in trials:
                raise ValueError("new cache contains a problem absent from the original fit")
            compare_trial(trials[key], trial)
        if manifest.get("method") == "undirected_gamma_trajectory_cache":
            for run in manifest["runs"]:
                if (run["wm_decay"], run["wm_neighbor_activation"]) == gamma_law:
                    if (run["metadata_sha256"], run["max_fixations"]) not in gamma_sources:
                        raise ValueError("new Gamma WM law has a different source metadata hash or stopping horizon")
                    selected_gamma_found = True
    if not selected_gamma_found:
        raise ValueError("new Gamma cache does not contain the frozen selected WM law")
    return trials


def compare_trial(original, new):
    for field in ("paths", "observed_index"):
        if original[field] != new[field]:
            raise ValueError(f"new/original cache trial {field} differs")
    if ("problem_sha256" in original and "problem_sha256" in new
            and original["problem_sha256"] != new["problem_sha256"]):
        raise ValueError("new/original cache trial problem hash differs")


def original_scores(fit, row):
    return {"train": row["total_log_likelihood"], "validation": row["validation_log_likelihood"],
            "test": fit["test"]["total_log_likelihoods"][row["family"]]}


def baseline_se(fit, precision, candidate_id, split):
    """Prefer cached within-history M2; retain repeat-only estimates separately."""
    if precision is not None and candidate_id in precision.get("candidates", {}):
        row = precision["candidates"][candidate_id]
        return row["splits"][split]["delta_total_log_likelihood_mc_se"], "ordinary_within_history_M2_delta"
    row = fit.get("numerical_precision", {}).get("candidates", {}).get(candidate_id, {}).get(split, {})
    return row.get("approx_pooled_log_likelihood_mc_se"), "independent_repeat_only"


def score_splits(probabilities, repeat_probabilities, trials, assignment, variances=None):
    """Evaluate fixed complete-path probabilities on the original observed data."""
    probabilities = np.asarray(probabilities, dtype=float)
    repeats = np.asarray(repeat_probabilities, dtype=float)
    if (probabilities.shape != (len(trials),) or repeats.ndim != 2 or repeats.shape[1] != len(trials)
            or np.any(probabilities <= 0) or np.any(repeats <= 0)
            or np.any(~np.isfinite(probabilities)) or np.any(~np.isfinite(repeats))):
        raise ValueError("confirmation probabilities must be finite and positive for every observed complete path")
    result = {}
    for split in SPLITS:
        rows = np.asarray([i for i, trial in enumerate(trials) if assignment[str(trial["pid"])] == split])
        if not len(rows):
            raise ValueError(f"confirmation has no {split} trials")
        repeat_scores = np.log(repeats[:, rows]).sum(axis=1)
        repeat_se = float(repeat_scores.std(ddof=1) / np.sqrt(len(repeats))) if len(repeats) > 1 else None
        analytic_se = (float(np.sqrt(np.sum(variances[rows] / probabilities[rows]**2)))
                       if variances is not None else None)
        result[split] = {
            "trial_count": len(rows), "confirmation_total_log_likelihood": float(np.log(probabilities[rows]).sum()),
            "confirmation_mc_se": analytic_se if analytic_se is not None else repeat_se,
            "confirmation_mc_se_method": "ordinary_within_history_M2_delta" if variances is not None
            else "Gamma_independent_raw_trajectory_repeats_only",
            "confirmation_analytic_mc_se": analytic_se, "confirmation_repeat_only_mc_se": repeat_se,
            "confirmation_independent_repeat_count": len(repeats),
            "confirmation_repeat_total_log_likelihoods": repeat_scores.tolist(),
        }
    return result


def confirmation_report(fit, inventory, ordinary_paths, gamma_paths, baseline, precision=None):
    selected = fit["fits"]["selected_by_validation"]
    fitted = {row["candidate_id"]: row for row in fit["fits"]["fits"]}
    ordinary_ids = [candidate_id for family, candidate_id in selected.items() if family != "undirected"]
    gamma = fitted[selected["undirected"]] if "undirected" in selected else None
    law = ((gamma["parameters"]["wm_decay"], gamma["parameters"]["wm_neighbor_activation"])
           if gamma is not None else None)
    if bool(ordinary_ids) != bool(ordinary_paths) or (gamma is not None) != bool(gamma_paths):
        raise ValueError("new caches must cover exactly the frozen selected ordinary and Gamma families")
    baseline_trials = validate_collections(baseline, ordinary_paths + gamma_paths, fit, law)
    keys = sorted(baseline_trials)
    selected_runs = [run["run_dir"] for candidate in inventory["candidates"] if candidate["id"] in ordinary_ids
                     for run in candidate["runs"]]

    def actual_run_dirs(paths):
        result = set()
        for path in paths:
            for run in read_json(path)["runs"]:
                if any(str(run["run_dir"]) == wanted or str(run["run_dir"]).endswith("/" + wanted)
                       for wanted in selected_runs):
                    result.add(run["run_dir"])
        return result

    truth_run_dirs = actual_run_dirs(ordinary_paths)
    independence = assert_independent_manifest_rng(
        baseline, ordinary_paths + gamma_paths, truth_run_dirs=truth_run_dirs,
        truth_wm_laws=[law] if law is not None else [], trial_keys=keys,
    )
    # Gap SEs also require independence BETWEEN selected models within each side.
    for label, paths, run_dirs in [
        ("new", ordinary_paths + gamma_paths, truth_run_dirs),
        ("baseline_selected", baseline, actual_run_dirs(baseline)),
    ]:
        seeds_by_trial, count = {}, 0
        for reference in iter_manifest_seeds(
            paths, run_dirs=run_dirs, wm_laws=[law] if law is not None else [], trial_keys=keys,
        ):
            seen = seeds_by_trial.setdefault(reference.trial_key, set())
            if reference.seed in seen:
                raise ValueError(f"{label} selected candidate streams reuse a same-problem RNG seed")
            seen.add(reference.seed)
            count += 1
        independence[f"{label}_internal_same_problem_streams_checked"] = count
    scores = {}
    if ordinary_ids:
        streamed = stream_selected_precision(ordinary_paths, fit, inventory, ordinary_ids, include_repeats=True)
        for candidate_id in ordinary_ids:
            scores[candidate_id] = score_splits(
                streamed["probabilities"][candidate_id], streamed["repeat_probabilities"][candidate_id],
                streamed["trials"], fit["split"]["participant_assignment"], streamed["variances"][candidate_id],
            )
    if gamma is not None:
        parameter = gamma["parameters"]
        manifest, pooled, repeats = stream_undirected_candidates(gamma_paths, candidate_specs_by_law={
            law: [{"gamma_shape": parameter["gamma_shape"], "gamma_mean": parameter["gamma_mean"]}],
        }, trial_keys=keys)
        observed = np.asarray([trial["observed_index"] for trial in manifest["trials"]])

        def chosen(candidate):
            return candidate.probabilities(gamma["lapse_rate"])[np.arange(len(observed)), observed]

        scores[gamma["candidate_id"]] = score_splits(
            chosen(pooled[0]), np.stack([chosen(repeated[0]) for repeated in repeats]),
            manifest["trials"], fit["split"]["participant_assignment"],
        )
    candidates = []
    for family, candidate_id in selected.items():
        row = fitted[candidate_id]
        original = original_scores(fit, row)
        for split, summary in scores[candidate_id].items():
            old_se, method = baseline_se(fit, precision, candidate_id, split)
            new_se = summary["confirmation_mc_se"]
            combined = float(np.hypot(old_se, new_se)) if old_se is not None and new_se is not None else None
            summary.update({
                "original_total_log_likelihood": original[split],
                "confirmation_minus_original": summary["confirmation_total_log_likelihood"] - original[split],
                "baseline_mc_se": old_se, "baseline_mc_se_method": method, "combined_delta_mc_se": combined,
            })
        candidates.append({"candidate_id": candidate_id, "family": family, "parameters": row["parameters"],
                           "lapse_rate": row["lapse_rate"], "splits": scores[candidate_id]})
    gaps = []
    for i, model in enumerate(candidates):
        for reference in candidates[:i]:
            splits = {}
            for split in SPLITS:
                left, right = model["splits"][split], reference["splits"][split]
                original_gap = left["original_total_log_likelihood"] - right["original_total_log_likelihood"]
                new_gap = left["confirmation_total_log_likelihood"] - right["confirmation_total_log_likelihood"]
                new_errors = left["confirmation_mc_se"], right["confirmation_mc_se"]
                all_errors = (*new_errors, left["baseline_mc_se"], right["baseline_mc_se"])
                splits[split] = {
                    "original_gap": original_gap, "confirmation_gap": new_gap,
                    "change_in_gap": new_gap - original_gap,
                    "confirmation_gap_mc_se": float(np.linalg.norm(new_errors))
                    if all(value is not None for value in new_errors) else None,
                    "combined_change_mc_se": float(np.linalg.norm(all_errors))
                    if all(value is not None for value in all_errors) else None,
                }
            gaps.append({"model": model["family"], "reference": reference["family"], "splits": splits})
    return {
        "schema_version": 1, "method": "independent_MC_confirmation_of_frozen_selected_parameters",
        "parameters_refitted": False, "models_reselected": False,
        "split_sha256": fit["split"]["split_sha256"], "candidates": candidates,
        "paired_family_gaps": gaps, "independence": independence,
        "limitations": [
            "Both evaluations use identical observed responses; differences assess numerical integration only.",
            "No optimization, validation reselection, participant resampling, or changed statistical comparison "
            "is claimed.",
            "Ordinary uncertainty uses cached within-history covariance and a delta-log approximation.",
            "Gamma uncertainty is estimated only from independent raw trajectory repeats and can be noisy "
            "with few repeats.",
            "Combined errors assume independent original/new and distinct-family integration streams; cross-problem "
            "uint32 seed coincidences are outside the same-problem independence audit.",
            "If baseline analytic precision is absent, combined errors use the original fit's repeat-only approximation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache", nargs="+", type=Path, default=[])
    parser.add_argument("--undirected-cache", nargs="+", type=Path, default=[])
    parser.add_argument("--baseline-cache", nargs="+", type=Path)
    parser.add_argument("--baseline-precision", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    fit, inventory = read_json(args.fit), read_json(args.candidates)
    if file_hash(args.candidates) != fit["provenance"]["candidate_manifest_sha256"]:
        parser.error("candidate inventory hash differs from frozen fit provenance")
    precision = read_json(args.baseline_precision) if args.baseline_precision else None
    if precision is not None and precision["provenance"]["fit"]["sha256"] != file_hash(args.fit):
        parser.error("baseline precision belongs to a different frozen fit report")
    baseline = baseline_paths(fit, args.baseline_cache)
    report = confirmation_report(fit, inventory, args.cache, args.undirected_cache, baseline, precision)
    report["provenance"] = {
        "fit": {"path": str(args.fit.resolve()), "sha256": file_hash(args.fit)},
        "candidate_inventory": {"path": str(args.candidates.resolve()), "sha256": file_hash(args.candidates)},
        "confirmation_cache_manifests": [{"path": str(path.resolve()), "sha256": file_hash(path)}
                                         for path in args.cache + args.undirected_cache],
        "baseline_cache_manifests": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in baseline],
        "baseline_precision": {"path": str(args.baseline_precision.resolve()),
                               "sha256": file_hash(args.baseline_precision)}
        if args.baseline_precision else None,
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: file_hash(path) for path in [
            "action_truth_confirmation.py", "action_mc_precision.py", "modules/action_cache_fitting.py",
            "modules/action_recovery_rng.py",
        ]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} candidates={len(report['candidates'])}", flush=True)


if __name__ == "__main__":
    main()
