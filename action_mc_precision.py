"""Propagate cached ordinary planning-MC moments through frozen action fits.

This standalone analysis reads mean/M2 arrays only. It never changes fitted
parameters, generates planning samples, or modifies simulation/cache sources.
"""

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import subprocess

import numpy as np


SPLITS = ("train", "validation", "test")
QUANTILES = (0, .5, .9, .95, .99, 1)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def observed_moments(mean, m2, path_length, lapse_rate, samples):
    """Probability and variance of its independent-sample arithmetic MC mean."""
    mean, m2 = np.asarray(mean, dtype=float), np.asarray(m2, dtype=float)
    if (mean.ndim != 1 or m2.shape != (len(mean), len(mean)) or not 1 <= path_length < len(mean)
            or samples < 2 or not .01 <= lapse_rate <= 1):
        raise ValueError("invalid observed-path moments, lapse, path length, or sample count")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(m2)):
        raise ValueError("cached moments must be finite")
    k = np.arange(len(mean))
    binomial = np.asarray([comb(path_length, int(i)) if i <= path_length else 0 for i in k], dtype=float)
    weights = binomial * lapse_rate**k * (1 - lapse_rate)**np.maximum(path_length - k, 0)
    probability = float(mean @ weights)
    numerator = float(weights @ m2 @ weights)
    # At lapse one, exact cancellation can make a theoretically zero quadratic
    # form slightly negative. Reject material negative variance, clip roundoff.
    tolerance = 1e-10 * max(float(np.abs(m2).max()) * float(weights.sum())**2, 1e-20)
    if not 0 < probability <= 1 + 1e-10 or numerator < -tolerance:
        raise ValueError("cached moments imply an invalid observed probability or variance")
    return probability, max(numerator, 0) / (samples * (samples - 1))


def aggregate_independent_moments(probabilities, variances, weights):
    """Variance propagation for an arithmetic mixture of independent estimates."""
    probabilities, variances, weights = map(lambda value: np.asarray(value, dtype=float),
                                            (probabilities, variances, weights))
    if (probabilities.ndim != 1 or probabilities.shape != variances.shape or probabilities.shape != weights.shape
            or not len(probabilities) or np.any(weights <= 0) or np.any(variances < 0)
            or np.any(~np.isfinite(probabilities)) or np.any(~np.isfinite(variances))
            or np.any(~np.isfinite(weights))):
        raise ValueError("nonempty finite independent moments and positive mixture weights required")
    weights = weights / weights.sum()
    return float(weights @ probabilities), float(np.square(weights) @ variances)


def describe(values):
    values = np.asarray(values, dtype=float)
    return {str(q): float(value) for q, value in zip(QUANTILES, np.quantile(values, QUANTILES))}


def summarize_trials(probabilities, variances, trials, assignment, *, tail_count=10, include_trials=True):
    probabilities, variances = np.asarray(probabilities), np.asarray(variances)
    if probabilities.shape != (len(trials),) or variances.shape != probabilities.shape or not len(trials):
        raise ValueError("one probability and variance is needed for every complete trial")
    log_scores = np.log(probabilities)
    log_variances = variances / np.square(probabilities)
    relative_errors = np.sqrt(log_variances)
    # This is the leading Jensen/log-transform bias at a fixed true probability.
    # A plug-in p and estimated variance make it a diagnostic, not a correction
    # adopted for fitting or model selection.
    estimated_log_bias = -.5 * log_variances
    trial_rows = []
    for t, trial in enumerate(trials):
        trial_rows.append({
            "pid": str(trial["pid"]), "trial_index": int(trial["trial_index"]),
            "split": assignment[str(trial["pid"])], "observed_path": trial["paths"][trial["observed_index"]],
            "probability": float(probabilities[t]), "probability_mc_se": float(np.sqrt(variances[t])),
            "log_probability": float(log_scores[t]), "delta_log_mc_se": float(relative_errors[t]),
            "estimated_log_bias": float(estimated_log_bias[t]),
        })
    splits = {}
    for split in SPLITS:
        rows = np.asarray([i for i, trial in enumerate(trials) if assignment[str(trial["pid"])] == split])
        if not len(rows):
            raise ValueError(f"no cached complete trials in frozen {split} split")
        variance = float(log_variances[rows].sum())
        bias = float(estimated_log_bias[rows].sum())
        total = float(log_scores[rows].sum())
        splits[split] = {
            "trial_count": len(rows), "total_log_likelihood": total,
            "delta_total_log_likelihood_mc_variance": variance,
            "delta_total_log_likelihood_mc_se": float(np.sqrt(variance)),
            "estimated_total_log_bias": bias,
            "bias_corrected_total_log_likelihood_diagnostic": total - bias,
            "probability_quantiles": describe(probabilities[rows]),
            "probability_mc_se_quantiles": describe(np.sqrt(variances[rows])),
            "delta_log_mc_se_quantiles": describe(relative_errors[rows]),
            "trials_relative_mc_se_above_0p1": int((relative_errors[rows] > .1).sum()),
            "trials_relative_mc_se_above_0p25": int((relative_errors[rows] > .25).sum()),
            "largest_tail_fraction_of_log_mc_variance": float(np.sort(log_variances[rows])[-tail_count:].sum()
                                                              / variance) if variance > 0 else 0.0,
            "worst_relative_mc_se_trials": [trial_rows[i] for i in sorted(
                rows, key=lambda index: relative_errors[index], reverse=True,
            )[:tail_count]],
        }
    return {"splits": splits, "trials": trial_rows if include_trials else None}


def stream_selected_precision(cache_paths, fit, candidate_inventory, candidate_ids=None):
    """Stream raw within-history moments for frozen cognitive candidates."""
    fitted = {row["candidate_id"]: row for row in fit["fits"]["fits"]}
    selected = fit["fits"]["selected_by_validation"]
    specifications = {row["id"]: row for row in candidate_inventory["candidates"]}
    if candidate_ids is None:
        candidate_ids = [candidate_id for family, candidate_id in selected.items() if family != "undirected"]
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("choose at least one unique ordinary fitted candidate")
    expected = {}
    for candidate_id in candidate_ids:
        if candidate_id not in fitted or candidate_id not in specifications:
            raise ValueError(f"candidate is missing from fit or frozen checkpoint inventory: {candidate_id}")
        if fitted[candidate_id]["family"] == "undirected":
            raise ValueError("Gamma trajectory caches have repeat-only precision; select ordinary candidates here")
        for run in specifications[candidate_id]["runs"]:
            key = str(Path(run["run_dir"]))
            if key in expected:
                raise ValueError("selected candidates must have disjoint checkpoint streams")
            expected[key] = (candidate_id, run)

    def match(run_dir):
        matches = [key for key in expected if str(Path(run_dir)) == key or str(Path(run_dir)).endswith("/" + key)]
        if len(matches) > 1:
            raise ValueError("ambiguous checkpoint path")
        return matches[0] if matches else None

    data_hash = fit["provenance"]["data_sha256"]
    trials_by_key, available = {}, set()
    rng_rules = set()
    for path in cache_paths:
        manifest = read_json(path)
        if (manifest.get("cache_version") != 1
                or manifest.get("method") != "planning_mc_bernstein_moments_exact_execution"):
            raise ValueError("input is not a supported ordinary Bernstein mean/M2 cache")
        if manifest["data_sha256"] != data_hash:
            raise ValueError("cache data hash differs from fitted complete data")
        rng_rules.add(manifest.get("rng_rule", "unspecified"))
        relevant = [run for run in manifest["runs"] if match(run["run_dir"]) is not None]
        if not relevant:
            continue
        for trial in manifest["trials"]:
            key = (str(trial["pid"]), int(trial["trial_index"]))
            if key in trials_by_key and trials_by_key[key] != trial:
                raise ValueError(f"inconsistent cached trial identity: {key}")
            trials_by_key[key] = trial
        for run in relevant:
            key = match(run["run_dir"])
            available.add(key)
            _, specification = expected[key]
            for field in ("metadata_sha256", "params_sha256"):
                if field in specification and specification[field] != run[field]:
                    raise ValueError(f"cached {field} differs from frozen candidate inventory")
    if set(expected) != available:
        raise ValueError(f"missing {len(set(expected) - available)} selected checkpoint caches")
    keys = sorted(trials_by_key)
    trials = [trials_by_key[key] for key in keys]
    trial_count = sum(fit["split"]["trial_counts"].values())
    if len(trials) != trial_count:
        raise ValueError("selected caches do not cover the frozen complete trial count")
    assignment = fit["split"]["participant_assignment"]
    if {key[0] for key in keys} != set(assignment):
        raise ValueError("selected cache participants differ from the frozen split")
    index = {key: t for t, key in enumerate(keys)}
    probabilities = {candidate_id: np.zeros(len(trials)) for candidate_id in candidate_ids}
    variances = {candidate_id: np.zeros(len(trials)) for candidate_id in candidate_ids}
    seen = {key: np.zeros(len(trials), dtype=bool) for key in expected}
    # Exact same-trial seed reuse would invalidate independent-stream propagation.
    # Cross-problem coincidences of uint32 hashes are not treated as duplicated
    # observations; the calculation assumes independent problem-level integration.
    trial_seeds = [set() for _ in trials]
    samples_by_candidate = {candidate_id: [] for candidate_id in candidate_ids}
    for path in cache_paths:
        path = Path(path)
        manifest = read_json(path)
        relevant = [run for run in manifest["runs"] if match(run["run_dir"]) is not None]
        if not relevant:
            continue
        array_path = path.parent / manifest["arrays_file"]
        if file_hash(array_path) != manifest["arrays_sha256"]:
            raise ValueError("cached moment array hash mismatch")
        with np.load(array_path, allow_pickle=False) as arrays:
            for run in relevant:
                run_key = match(run["run_dir"])
                candidate_id, _ = expected[run_key]
                lapse = fitted[candidate_id]["lapse_rate"]
                checkpoint_count = len(specifications[candidate_id]["runs"])
                for trial, estimates in zip(manifest["trials"], run["estimates"], strict=True):
                    t = index[(str(trial["pid"]), int(trial["trial_index"]))]
                    if seen[run_key][t]:
                        raise ValueError("duplicate selected checkpoint/trial cache")
                    seen[run_key][t] = True
                    means, moment_variances, counts = [], [], []
                    observed = trial["observed_index"]
                    length = len(trial["paths"][observed])
                    for entry in estimates:
                        seed = int(entry["seed"])
                        if seed in trial_seeds[t]:
                            raise ValueError(
                                f"same-trial RNG seed reuse prevents independent MC propagation: {keys[t]}",
                            )
                        trial_seeds[t].add(seed)
                        mean = arrays[entry["key"] + "_mean"][observed]
                        m2 = arrays[entry["key"] + "_m2"][observed]
                        probability, variance = observed_moments(mean, m2, length, lapse, entry["samples"])
                        means.append(probability)
                        moment_variances.append(variance)
                        counts.append(entry["samples"])
                    probability, variance = aggregate_independent_moments(means, moment_variances, counts)
                    probabilities[candidate_id][t] += probability / checkpoint_count
                    variances[candidate_id][t] += variance / checkpoint_count**2
                    samples_by_candidate[candidate_id].append(sum(counts))
    if any(not np.all(mask) for mask in seen.values()):
        raise ValueError("a selected checkpoint is missing complete trial coverage")
    return {
        "trials": trials, "probabilities": probabilities, "variances": variances,
        "checkpoint_counts": {candidate_id: len(specifications[candidate_id]["runs"])
                              for candidate_id in candidate_ids},
        "samples_per_checkpoint_trial": {candidate_id: {
            "minimum": min(counts), "maximum": max(counts),
        } for candidate_id, counts in samples_by_candidate.items()},
        "cache_rng_rules": sorted(rng_rules),
    }


def precision_report(cache_paths, fit, candidate_inventory, *, candidate_ids=None, include_trials=True, tail_count=10):
    streamed = stream_selected_precision(cache_paths, fit, candidate_inventory, candidate_ids)
    trials = streamed["trials"]
    assignment = fit["split"]["participant_assignment"]
    fitted = {row["candidate_id"]: row for row in fit["fits"]["fits"]}
    candidates = {}
    for candidate_id, probabilities in streamed["probabilities"].items():
        candidates[candidate_id] = {
            "family": fitted[candidate_id]["family"], "parameters": fitted[candidate_id]["parameters"],
            "fixed_lapse_rate": fitted[candidate_id]["lapse_rate"],
            "checkpoint_count": streamed["checkpoint_counts"][candidate_id],
            "samples_per_checkpoint_trial": streamed["samples_per_checkpoint_trial"][candidate_id],
            **summarize_trials(probabilities, streamed["variances"][candidate_id], trials, assignment,
                               tail_count=tail_count, include_trials=include_trials),
        }
        for split, summary in candidates[candidate_id]["splits"].items():
            reported = None
            if split == "train":
                reported = fitted[candidate_id]["total_log_likelihood"]
            elif split == "validation":
                reported = fitted[candidate_id]["validation_log_likelihood"]
            elif fit["fits"]["selected_by_validation"].get(fitted[candidate_id]["family"]) == candidate_id:
                reported = fit["test"]["total_log_likelihoods"][fitted[candidate_id]["family"]]
            summary["cached_minus_reported_total_log_likelihood"] = (
                summary["total_log_likelihood"] - reported if reported is not None else None
            )
            if reported is not None and not np.isclose(
                summary["total_log_likelihood"], reported, atol=1e-6, rtol=1e-10,
            ):
                raise ValueError(f"cached probabilities disagree with frozen {candidate_id} {split} fit score")
    # Uniform legal movement is exact and has zero numerical integration variance.
    random = np.asarray([2.0**-len(trial["paths"][trial["observed_index"]]) for trial in trials])
    random_summary = summarize_trials(random, np.zeros(len(trials)), trials, assignment,
                                      tail_count=tail_count, include_trials=False)
    comparison_summaries = {candidate_id: row["splits"] for candidate_id, row in candidates.items()}
    comparison_summaries["random"] = random_summary["splits"]
    comparisons = []
    names = list(comparison_summaries)
    for i, model in enumerate(names):
        for reference in names[:i]:
            summaries = {}
            for split in SPLITS:
                left, right = comparison_summaries[model][split], comparison_summaries[reference][split]
                variance = left["delta_total_log_likelihood_mc_variance"] + right[
                    "delta_total_log_likelihood_mc_variance"
                ]
                gap = left["total_log_likelihood"] - right["total_log_likelihood"]
                bias = left["estimated_total_log_bias"] - right["estimated_total_log_bias"]
                summaries[split] = {"total_log_likelihood_gap": gap, "delta_gap_mc_se": float(np.sqrt(variance)),
                                    "estimated_gap_log_bias": bias, "bias_corrected_gap_diagnostic": gap - bias}
            comparisons.append({"model": model, "reference": reference, "splits": summaries})
    gamma_id = fit["fits"]["selected_by_validation"].get("undirected")
    empirical = fit.get("numerical_precision", {})
    gamma_repeat_only = None
    mixed_gamma_comparisons = []
    if gamma_id is not None:
        gamma_repeat_only = {
            "candidate_id": gamma_id, "method": "independent_trajectory_repeats_only_not_within_history_M2",
            "independent_repeat_count": empirical.get("independent_repeat_count"),
            "splits": empirical.get("candidates", {}).get(gamma_id),
            "scope": "Copied from the frozen fit report; no Gamma moment propagation is claimed.",
        }
        if gamma_repeat_only["splits"] is not None:
            for candidate_id, candidate in candidates.items():
                splits = {}
                for split in SPLITS:
                    ordinary = candidate["splits"][split]
                    gamma = gamma_repeat_only["splits"][split]
                    repeat_se = gamma.get("approx_pooled_log_likelihood_mc_se")
                    combined_se = (float(np.sqrt(ordinary["delta_total_log_likelihood_mc_variance"] + repeat_se**2))
                                   if repeat_se is not None else None)
                    splits[split] = {
                        "total_log_likelihood_gap": gamma["pooled_total_log_likelihood"]
                        - ordinary["total_log_likelihood"],
                        "ordinary_delta_mc_se": ordinary["delta_total_log_likelihood_mc_se"],
                        "gamma_repeat_only_mc_se": repeat_se,
                        "mixed_gap_mc_se": combined_se,
                    }
                mixed_gamma_comparisons.append({
                    "model": gamma_id, "reference": candidate_id,
                    "method": "independent_ordinary_analytic_variance_plus_Gamma_repeat_estimated_variance",
                    "caveat": "Gamma component inherits the uncertainty of a small number of trajectory repetitions.",
                    "splits": splits,
                })
    return {
        "method": "within_history_cached_covariance_delta_log_precision_at_fixed_fitted_parameters",
        "formulas": {
            "repeat_mean_variance": "w.T @ M2 @ w / (samples * (samples - 1))",
            "checkpoint_mean": "sample-count weighted repeat means",
            "checkpoint_mean_variance": "sum(normalized_sample_weight**2 * repeat_mean_variance)",
            "candidate_probability": "equal-weight mean of checkpoint probabilities",
            "candidate_probability_variance": "sum(checkpoint_mean_variance) / checkpoint_count**2",
            "delta_log_variance": "candidate_probability_variance / candidate_probability**2",
            "estimated_log_bias": "-0.5 * delta_log_variance",
            "summed_log_likelihood_variance": "sum(delta_log_variance) over independent trial integrations",
            "distinct_candidate_gap_variance": "sum of the two candidate summed-log variances",
        },
        "scope": "Numerical integration uncertainty conditional on frozen cognitive/stopping/lapse parameters. "
                 "Excludes participant uncertainty, trained-policy-seed variability, and optimization uncertainty.",
        "assumptions": [
            "Independent latent histories within each cache repeat, and independent checkpoint/repeat streams.",
            "Independent numerical integration across problems; same-trial RNG seeds were checked for reuse.",
            "Delta-log and leading log-bias approximations are most reliable at small relative probability MC SE.",
            "Bias-corrected scores are diagnostics and do not replace the fitted or held-out scores.",
        ],
        "cache_rng_rules": streamed["cache_rng_rules"], "split_sha256": fit["split"]["split_sha256"],
        "candidates": candidates, "paired_ordinary_comparisons": comparisons,
        "gamma_repeat_only": gamma_repeat_only, "mixed_gamma_comparisons": mixed_gamma_comparisons,
        "independent_repeat_cross_check": empirical,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache", required=True, nargs="+", type=Path)
    parser.add_argument("--candidate-ids", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tail-count", type=int, default=10)
    parser.add_argument("--omit-trials", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.tail_count < 1:
        parser.error("tail-count must be positive")
    fit, inventory = read_json(args.fit), read_json(args.candidates)
    if file_hash(args.candidates) != fit["provenance"]["candidate_manifest_sha256"]:
        parser.error("candidate inventory hash differs from frozen fit provenance")
    report = precision_report(args.cache, fit, inventory, candidate_ids=args.candidate_ids,
                              include_trials=not args.omit_trials, tail_count=args.tail_count)
    report["provenance"] = {
        "fit": {"path": str(args.fit.resolve()), "sha256": file_hash(args.fit)},
        "candidate_inventory": {"path": str(args.candidates.resolve()), "sha256": file_hash(args.candidates)},
        "cache_manifests": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in args.cache],
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": file_hash(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} candidates={len(report['candidates'])}", flush=True)


if __name__ == "__main__":
    main()
