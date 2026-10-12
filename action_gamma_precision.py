"""Rescore frozen leading Gamma candidates using original independent raw repeats."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from action_mc_precision import SPLITS, describe, file_hash, read_json
from action_truth_confirmation import compare_trial, original_scores, score_splits
from modules.action_cache_fitting import stream_undirected_candidates
from modules.action_recovery_rng import iter_manifest_seeds


def top_gamma_fits(fit, top_k):
    """Retain the established validation ranking without fitting or reselection."""
    if top_k < 1:
        raise ValueError("top-k must be positive")
    rows = [row for row in fit["fits"]["fits"] if row["family"] == "undirected"]
    laws = [(row["parameters"]["wm_decay"], row["parameters"]["wm_neighbor_activation"]) for row in rows]
    if not rows or len(set(laws)) != len(laws):
        raise ValueError("expected one frozen training-fitted Gamma candidate per WM law")
    return sorted(rows, key=lambda row: (row["validation_log_likelihood"], row["candidate_id"]), reverse=True)[:top_k]


def frozen_gamma_manifests(paths, fit, laws):
    """Select original law shards and verify their exact frozen manifest identities."""
    frozen = {row["sha256"] for row in fit["provenance"]["cache_manifests"]}
    selected, seen_hashes, trial_metadata, horizons = [], set(), {}, {}
    for path in paths:
        path = Path(path)
        digest = file_hash(path)
        if digest not in frozen:
            raise ValueError("Gamma cache manifest is absent from frozen fit provenance")
        if digest in seen_hashes:
            raise ValueError("duplicate Gamma cache manifest")
        seen_hashes.add(digest)
        manifest = read_json(path)
        if (manifest.get("method") != "undirected_gamma_trajectory_cache"
                or manifest["data_sha256"] != fit["provenance"]["data_sha256"]):
            raise ValueError("Gamma cache method/data differs from frozen fit")
        matching = [run for run in manifest["runs"]
                    if (run["wm_decay"], run["wm_neighbor_activation"]) in laws]
        if not matching:
            continue
        selected.append(path)
        for run in matching:
            law = run["wm_decay"], run["wm_neighbor_activation"]
            identity = run["metadata_sha256"], run["max_fixations"]
            if law in horizons and horizons[law] != identity:
                raise ValueError("Gamma metadata/stopping horizon differs across original shards")
            horizons[law] = identity
        for trial in manifest["trials"]:
            key = str(trial["pid"]), int(trial["trial_index"])
            if key in trial_metadata:
                compare_trial(trial_metadata[key], trial)
            trial_metadata[key] = trial
    if set(horizons) != set(laws) or len(trial_metadata) != sum(fit["split"]["trial_counts"].values()):
        raise ValueError("selected Gamma laws do not cover frozen complete trials")
    return selected, trial_metadata, horizons


def repeat_tail_diagnostics(probabilities, repeats, trials, assignment):
    """Estimate per-trial delta-log precision from raw-repeat probability variation."""
    variances = repeats.var(axis=0, ddof=1) / len(repeats)
    relative = np.sqrt(variances) / probabilities
    result = {}
    for split in SPLITS:
        rows = np.asarray([i for i, trial in enumerate(trials) if assignment[str(trial["pid"])] == split])
        log_variances = relative[rows]**2
        total = float(log_variances.sum())
        worst = sorted(rows, key=lambda index: relative[index], reverse=True)[:10]
        result[split] = {
            "observed_path_probability_quantiles": describe(probabilities[rows]),
            "repeat_delta_log_mc_se_quantiles": describe(relative[rows]),
            "repeat_delta_total_log_likelihood_mc_se": float(np.sqrt(total)),
            "repeat_delta_leading_Jensen_bias_diagnostic": -.5 * total,
            "trials_relative_mc_se_above_0p1": int((relative[rows] > .1).sum()),
            "trials_relative_mc_se_above_0p25": int((relative[rows] > .25).sum()),
            "top10_fraction_of_repeat_delta_log_variance": float(np.sort(log_variances)[-10:].sum() / total)
            if total else 0.,
            "worst_relative_mc_se_trials": [{"pid": str(trials[t]["pid"]),
                "trial_index": int(trials[t]["trial_index"]), "probability": float(probabilities[t]),
                "repeat_delta_log_mc_se": float(relative[t])} for t in worst],
        }
    return result


def gamma_precision_report(fit, cache_paths, top_k=3):
    frozen = top_gamma_fits(fit, top_k)
    wm_values = {name: sorted({row["parameters"][name] for row in fit["fits"]["fits"]
                               if row["family"] == "undirected"})
                 for name in ("wm_decay", "wm_neighbor_activation")}
    laws = {(row["parameters"]["wm_decay"], row["parameters"]["wm_neighbor_activation"]) for row in frozen}
    paths, trials_by_key, horizons = frozen_gamma_manifests(cache_paths, fit, laws)
    seeds_by_trial, stream_count = {}, 0
    for reference in iter_manifest_seeds(paths, run_dirs=[], wm_laws=laws):
        seen = seeds_by_trial.setdefault(reference.trial_key, set())
        if reference.seed in seen:
            raise ValueError("Gamma raw repeats/laws reuse a same-problem RNG seed")
        seen.add(reference.seed)
        stream_count += 1
    specifications = {(row["parameters"]["wm_decay"], row["parameters"]["wm_neighbor_activation"]): [{
        "gamma_shape": row["parameters"]["gamma_shape"], "gamma_mean": row["parameters"]["gamma_mean"],
    }] for row in frozen}
    manifest, pooled, repeated = stream_undirected_candidates(
        paths, candidate_specs_by_law=specifications, trial_keys=sorted(trials_by_key),
    )
    if len(repeated) < 2:
        raise ValueError("Gamma precision requires at least two independent raw repetitions")
    trials = manifest["trials"]
    observed = np.asarray([trial["observed_index"] for trial in trials])
    assignment = fit["split"]["participant_assignment"]
    by_law = {(candidate.parameters["wm_decay"], candidate.parameters["wm_neighbor_activation"]): i
              for i, candidate in enumerate(pooled)}
    refinements = {(row["wm_decay"], row["wm_neighbor_activation"]): row for row in fit.get("gamma_refinement") or []}
    candidates = []
    for rank, row in enumerate(frozen, start=1):
        parameter = row["parameters"]
        law = parameter["wm_decay"], parameter["wm_neighbor_activation"]
        index = by_law[law]
        def chosen(candidate):
            return candidate.probabilities(row["lapse_rate"])[np.arange(len(observed)), observed]
        probabilities = chosen(pooled[index])
        repeats = np.stack([chosen(repetition[index]) for repetition in repeated])
        summaries = score_splits(probabilities, repeats, trials, assignment)
        tails = repeat_tail_diagnostics(probabilities, repeats, trials, assignment)
        for split, original in original_scores(fit, row).items():
            # Nonselected Gamma candidates have no reported test score; their fixed
            # test score is diagnostic and never used to revise the selection.
            if split == "test" and fit["fits"]["selected_by_validation"]["undirected"] != row["candidate_id"]:
                original = None
            pooled_score = summaries[split]["confirmation_total_log_likelihood"]
            if original is not None and not np.isclose(pooled_score, original, atol=1e-7, rtol=1e-10):
                raise ValueError(f"Gamma {row['candidate_id']} {split} pooled likelihood differs from frozen fit")
            summaries[split] = {
                "trial_count": summaries[split]["trial_count"], "pooled_total_log_likelihood": pooled_score,
                "frozen_fit_total_log_likelihood": original,
                "pooled_minus_frozen_fit": pooled_score - original if original is not None else None,
                "raw_repeat_total_log_likelihoods": summaries[split]["confirmation_repeat_total_log_likelihoods"],
                "raw_repeat_total_log_likelihood_mc_se": summaries[split]["confirmation_repeat_only_mc_se"],
                "raw_repeat_count": len(repeated), "repeat_variance_degrees_of_freedom": len(repeated) - 1,
                **tails[split],
            }
        refinement = refinements.get(law, {})
        candidates.append({"validation_rank": rank, "candidate_id": row["candidate_id"],
            "parameters": parameter, "lapse_rate": row["lapse_rate"],
            "lapse_at_boundary": bool(np.isclose(row["lapse_rate"], .01) or np.isclose(row["lapse_rate"], 1)),
            "wm_at_available_grid_boundary": {
                name: len(values) > 1 and parameter[name] in (values[0], values[-1])
                for name, values in wm_values.items()
            },
            "stopping_horizon": horizons[law][1], "training_refinement_diagnostics": {
                key: refinement.get(key) for key in ["bounds", "boundary_flags", "mean_total_fixations",
                                                     "maximum_target_fraction", "initial_target_fraction", "flatness"]
            }, "splits": summaries})
    comparisons = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            a, b = left["splits"]["validation"], right["splits"]["validation"]
            gap = a["pooled_total_log_likelihood"] - b["pooled_total_log_likelihood"]
            se = float(np.hypot(a["raw_repeat_total_log_likelihood_mc_se"],
                                b["raw_repeat_total_log_likelihood_mc_se"]))
            comparisons.append({"model": left["candidate_id"], "reference": right["candidate_id"],
                                "validation_log_likelihood_gap": gap, "independent_law_gap_mc_se": se,
                                "gap_over_numerical_se": gap / se if se else None,
                                "method": "quadrature_of_independent_law_raw_repeat_total_log_likelihood_SEs"})
    return {"schema_version": 1, "method": "frozen_top_validation_Gamma_raw_repeat_precision",
        "parameters_refitted": False, "models_reselected": False, "split_sha256": fit["split"]["split_sha256"],
        "selected_undirected_candidate": fit["fits"]["selected_by_validation"]["undirected"],
        "requested_top_k": top_k, "candidates": candidates, "validation_comparisons": comparisons,
        "same_problem_seed_independence": {"checked": True, "streams_checked": stream_count},
        "used_cache_manifests": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in paths],
        "limitations": [
            "All parameters and the established validation ranking remain frozen; this is a numerical diagnostic.",
            "Raw-repeat likelihood SEs have few degrees of freedom and approximate precision of the pooled score.",
            "Per-trial delta-log SE and leading Jensen bias use repeat-estimated probability variances; "
            "they are noisy diagnostics rather than within-history covariance estimates or score corrections.",
            "Summed delta variances assume independent problem integrations; same-problem RNG reuse is rejected, "
            "but cross-problem uint32 seed coincidences are outside the independence check.",
            "Nonselected candidates' test scores are diagnostic only and cannot be used to revise selection.",
            "Participant sampling, parameter-selection uncertainty, and trained-policy uncertainty are excluded.",
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--undirected-cache", required=True, nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = gamma_precision_report(read_json(args.fit), args.undirected_cache, args.top_k)
    report["provenance"] = {
        "fit": {"path": str(args.fit.resolve()), "sha256": file_hash(args.fit)},
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: file_hash(name) for name in [
            "action_gamma_precision.py", "action_truth_confirmation.py", "action_mc_precision.py",
            "modules/action_cache_fitting.py", "modules/action_recovery_rng.py",
        ]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} candidates={len(report['candidates'])}", flush=True)


if __name__ == "__main__":
    main()
