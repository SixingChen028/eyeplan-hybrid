"""Pooled action-only fitting with frozen participant splits and held-out comparisons.

Checkpoint seeds describe replicates of one cognitive model. Their probabilities
must be averaged before constructing a candidate, rather than averaging log scores.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


MIN_LAPSE_RATE = 0.01


@dataclass
class Candidate:
    candidate_id: str
    family: str
    parameters: dict
    evaluate: Callable[[float], np.ndarray]
    evaluate_subset: Callable | None = None
    evaluate_observed: Callable | None = None

    def probabilities(self, lapse_rate, indices=None):
        if indices is not None and self.evaluate_subset is not None:
            return np.asarray(self.evaluate_subset(float(lapse_rate), indices), dtype=np.float64)
        values = np.asarray(self.evaluate(float(lapse_rate)), dtype=np.float64)
        return values if indices is None else values[indices]


def canonical_hash(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def freeze_participant_split(records, seed=20261011, fractions=(0.6, 0.2, 0.2)):
    """Hash-rank participants once; every trial of a participant stays together."""
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("fractions must contain three positive values summing to one")
    participants = sorted({str(record["pid"]) for record in records})
    if len(participants) < 3:
        raise ValueError("at least three participants are needed")
    ranked = sorted(participants, key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest())
    counts = np.floor(fractions * len(ranked)).astype(int)
    # Largest-remainder allocation preserves the requested proportions and total.
    for index in np.argsort(-(fractions * len(ranked) - counts), kind="stable")[:len(ranked) - counts.sum()]:
        counts[index] += 1
    if np.any(counts == 0):
        raise ValueError("each split needs at least one participant")
    names = ("train", "validation", "test")
    assignment = {}
    start = 0
    for name, count in zip(names, counts):
        assignment.update({pid: name for pid in ranked[start:start + count]})
        start += count
    trial_keys = [[str(r["pid"]), int(r["trial_index"])] for r in records]
    if len({tuple(key) for key in trial_keys}) != len(trial_keys):
        raise ValueError("participant and trial_index must uniquely identify trials")
    splits = {name: [i for i, record in enumerate(records) if assignment[str(record["pid"])] == name] for name in names}
    result = {
        "method": "sha256_ranked_participants_largest_remainder", "seed": seed,
        "fractions": fractions.tolist(), "participant_assignment": assignment,
        "participant_counts": dict(zip(names, counts.tolist())),
        "trial_counts": {name: len(indices) for name, indices in splits.items()},
        "trial_keys_sha256": canonical_hash(trial_keys), "indices": splits,
    }
    result["split_sha256"] = canonical_hash(result)
    return result


def observed_log_probabilities(candidate, lapse_rate, observed_indices, indices=None):
    observed_indices = np.asarray(observed_indices, dtype=int)
    if candidate.evaluate_observed is not None:
        selected = np.asarray(candidate.evaluate_observed(lapse_rate, observed_indices, indices), dtype=float)
        if np.any(~np.isfinite(selected)) or np.any(selected <= 0) or np.any(selected > 1 + 2e-5):
            raise ValueError("observed path probabilities must be finite and in (0,1]")
        return np.log(selected)
    probabilities = candidate.probabilities(lapse_rate)
    if probabilities.ndim != 2 or len(probabilities) != len(observed_indices):
        raise ValueError("candidate probabilities must have shape [trial, path]")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("candidate probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=2e-5):
        raise ValueError("each trial's path probabilities must sum to one")
    if np.any(observed_indices < 0) or np.any(observed_indices >= probabilities.shape[1]):
        raise ValueError("observed path index outside candidate paths")
    rows = np.arange(len(observed_indices)) if indices is None else np.asarray(indices, dtype=int)
    selected = probabilities[rows, observed_indices[rows]]
    if np.any(selected <= 0):
        raise ValueError("observed complete paths must have positive probability under movement lapse")
    return np.log(selected)


def fit_lapse(candidate, observed_indices, indices=None, lapse_bounds=(MIN_LAPSE_RATE, 1.0), profile_points=101):
    """Optimize pooled lapse on supplied trials; include bounds and profile maxima.

    The dense profile guards against the non-unimodal likelihoods possible for
    complete paths. Bounded local refinement is applied around every grid maximum.
    """
    lower, upper = map(float, lapse_bounds)
    if not MIN_LAPSE_RATE <= lower < upper <= 1 or profile_points < 3:
        raise ValueError("invalid lapse bounds or profile size")

    def score(lapse):
        return float(observed_log_probabilities(candidate, lapse, observed_indices, indices).sum())

    grid = np.linspace(lower, upper, profile_points)
    scores = np.asarray([score(lapse) for lapse in grid])
    optima = [(scores[0], lower), (scores[-1], upper)]
    for i in range(1, len(grid) - 1):
        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and (
            scores[i] > scores[i - 1] or scores[i] > scores[i + 1]
        ):
            optimum = minimize_scalar(
                lambda lapse: -score(lapse), bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                options={"xatol": 1e-8},
            )
            optima.append((-float(optimum.fun), float(optimum.x)))
    best_score, best_lapse = max(optima, key=lambda pair: pair[0])
    # Report discrete support, including disconnected regions; finite policy grid
    # and MC error make these diagnostic likelihood intervals, not calibrated CIs.
    if best_lapse not in grid:
        order = np.argsort(np.r_[grid, best_lapse])
        grid = np.r_[grid, best_lapse][order]
        scores = np.r_[scores, best_score][order]
    support = grid[scores >= best_score - 1.920729410347062]
    return {
        "candidate_id": candidate.candidate_id, "family": candidate.family, "parameters": candidate.parameters,
        "lapse_rate": best_lapse, "total_log_likelihood": best_score,
        "trial_count": len(observed_indices) if indices is None else len(indices),
        "lapse_at_boundary": bool(np.isclose(best_lapse, lower, atol=1e-6) or np.isclose(best_lapse, upper, atol=1e-6)),
        "profile": {"lapse_rates": grid.tolist(), "log_likelihoods": scores.tolist(),
                    "likelihood_ratio_95_support": support.tolist(),
                    "approx_95_interval": [float(support.min()), float(support.max())]},
    }


def fit_all_candidates(candidates, observed_indices, split_indices, profile_points=101):
    """Fit train lapse, then select finite cognitive candidates on validation."""
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    fits = []
    for candidate in candidates:
        fit = fit_lapse(candidate, observed_indices, split_indices["train"], profile_points=profile_points)
        fit["validation_log_likelihood"] = float(observed_log_probabilities(
            candidate, fit["lapse_rate"], observed_indices, split_indices["validation"],
        ).sum())
        fits.append(fit)
    families = sorted({candidate.family for candidate in candidates})
    selected = {}
    training_best = {}
    for family in families:
        family_fits = [fit for fit in fits if fit["family"] == family]
        selected[family] = max(family_fits, key=lambda fit: (fit["validation_log_likelihood"], fit["candidate_id"]))[
            "candidate_id"
        ]
        training_best[family] = max(family_fits, key=lambda fit: (fit["total_log_likelihood"], fit["candidate_id"]))[
            "candidate_id"
        ]
    return {"fits": fits, "selected_by_validation": selected, "training_best": training_best}


def participant_bootstrap(log_scores, participant_ids, *, repetitions=2000, seed=20261011):
    """Paired participant resampling, retaining all trials within a participant.

    Means use pooled trial weights in each draw, matching the fitting objective.
    Report an equal-participant sensitivity alongside the primary pooled result.
    """
    scores = np.asarray(log_scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    pids = np.asarray(participant_ids, dtype=str)
    if scores.shape[0] != len(pids) or not len(pids) or repetitions < 1:
        raise ValueError("nonempty trial scores and positive bootstrap repetitions required")
    participants, inverse = np.unique(pids, return_inverse=True)
    totals = np.zeros((len(participants), scores.shape[1]))
    np.add.at(totals, inverse, scores)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(participants), size=(repetitions, len(participants)))
    means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    return {
        "mean_log_score": scores.mean(axis=0).tolist(),
        "bootstrap_probability_above_zero": (means > 0).mean(axis=0).tolist(),
        "mean_log_score_95_interval": np.quantile(means, [.025, .975], axis=0).T.tolist(),
        "equal_participant_mean_log_score": (totals / counts[:, None]).mean(axis=0).tolist(),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "participant_count": len(participants), "trial_count": len(pids),
    }


def evaluate_test(candidates, fitted, observed_indices, split_indices, participant_ids, random_probabilities,
                  *, bootstrap_repetitions=2000, seed=20261011):
    """Score frozen choices on test once and compare with uniform legal movement."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    fits = {fit["candidate_id"]: fit for fit in fitted["fits"]}
    rows = np.asarray(split_indices["test"], dtype=int)
    pids = np.asarray(participant_ids, dtype=str)[rows]
    random_probabilities = np.asarray(random_probabilities, dtype=float)
    if random_probabilities.shape != (len(observed_indices),) or np.any(random_probabilities <= 0):
        raise ValueError("random_probabilities must give the observed uniform-movement path probability")
    random_scores = np.log(random_probabilities[rows])
    scores = {"random": random_scores}
    for family, candidate_id in fitted["selected_by_validation"].items():
        scores[family] = observed_log_probabilities(by_id[candidate_id], fits[candidate_id]["lapse_rate"],
                                                   observed_indices, rows)
    names = list(scores)
    matrix = np.stack([scores[name] for name in names], axis=1)
    report = {
        "selection_rule": "validation_best_at_train_fitted_lapse_without_test_refitting",
        "models": names, "selected_candidates": fitted["selected_by_validation"],
        "scores": participant_bootstrap(matrix, pids, repetitions=bootstrap_repetitions, seed=seed),
        "total_log_likelihoods": dict(zip(names, matrix.sum(axis=0).tolist())), "paired_comparisons": [],
        "participant_scores": [],
    }
    for i in range(len(names)):
        for j in range(i):
            comparison = participant_bootstrap(matrix[:, i] - matrix[:, j], pids,
                                               repetitions=bootstrap_repetitions, seed=seed)
            report["paired_comparisons"].append({"model": names[i], "reference": names[j], **comparison})
    for pid in sorted(set(pids)):
        mask = pids == pid
        report["participant_scores"].append({"pid": pid, "trial_count": int(mask.sum()),
                                             "total_log_scores": dict(zip(names, matrix[mask].sum(axis=0).tolist()))})
    return report


def candidates_from_cache(manifest, caches, candidate_manifest=None, repeat_indices=None):
    """Average latent-MC repeats within checkpoint, then checkpoint seed replicas.

    Explicit candidate manifests have ``id``, ``model``, ``parameters``, and
    ``runs`` (each with ``run_dir``), matching the frozen inventory. A missing
    manifest treats every cached run as a distinct candidate.
    """
    from math import comb
    from pathlib import Path

    trials = manifest["trials"]
    observed = np.asarray([trial["observed_index"] for trial in trials], dtype=int)
    max_paths = max(len(trial["paths"]) for trial in trials)
    max_degree = max(len(path) for trial in trials for path in trial["paths"])
    lengths = np.zeros((len(trials), max_paths), dtype=int)
    binomial = np.zeros((len(trials), max_paths, max_degree + 1))
    for t, trial in enumerate(trials):
        for p, path in enumerate(trial["paths"]):
            lengths[t, p] = len(path)
            binomial[t, p, :len(path) + 1] = [comb(len(path), k) for k in range(len(path) + 1)]
    powers = np.arange(max_degree + 1)[None, None, :]
    remaining = np.maximum(lengths[:, :, None] - powers, 0)
    run_dirs = [str(Path(run["run_dir"])) for run in manifest["runs"]]
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError("each run_dir must occur only once in a combined manifest")
    if candidate_manifest is None:
        candidate_specs = [{"id": Path(path).name, "model": "unspecified", "parameters": {},
                            "cache_run_indices": [i]} for i, path in enumerate(run_dirs)]
    else:
        candidate_specs = candidate_manifest["candidates"]
    candidates = []
    for specification in candidate_specs:
        run_indices = specification.get("cache_run_indices")
        if run_indices is None:
            run_indices = []
            for run in specification["runs"]:
                wanted = str(Path(run["run_dir"]))
                matches = [i for i, path in enumerate(run_dirs) if path == wanted or path.endswith("/" + wanted)]
                if len(matches) != 1:
                    raise ValueError(f"candidate {specification['id']} run {wanted} has {len(matches)} cache matches")
                run_indices.append(matches[0])
        if not run_indices:
            raise ValueError("each candidate requires at least one checkpoint")
        seed_coefficients = []
        for run_index in run_indices:
            coefficients = np.zeros((*lengths.shape, max_degree + 1))
            for t, trial in enumerate(trials):
                repeats = caches[run_index][t]
                selected = repeats if repeat_indices is None else [repeats[index] for index in repeat_indices]
                if not selected:
                    raise ValueError("at least one independent cache repeat is needed")
                if any(tuple(tuple(path) for path in trial["paths"]) != cache.paths for cache in selected):
                    raise ValueError("cache path ordering differs from manifest")
                weights = np.asarray([cache.samples for cache in selected], dtype=float)
                means = np.stack([cache.mean_coefficients for cache in selected])
                averaged = np.average(means, axis=0, weights=weights)
                coefficients[t, :averaged.shape[0], :averaged.shape[1]] = averaged
            seed_coefficients.append(coefficients)
        mean_coefficients = np.mean(seed_coefficients, axis=0)

        def evaluate(lapse, coefficients=mean_coefficients):
            weights = binomial * lapse**powers * (1 - lapse)**remaining
            return np.sum(coefficients * weights, axis=-1)

        candidates.append(Candidate(specification["id"], specification["model"], specification["parameters"], evaluate))
    return candidates, observed


def combine_cache_bundles(bundles):
    """Combine disjoint run/trial shards, checking every overlapping identity.

    Repeated complete bundles for the same run are rejected to avoid silently
    double-counting latent samples. Supply independent repeats in one bundle.
    """
    from pathlib import Path

    trials_by_key = {}
    runs_by_path = {}
    for manifest, caches in bundles:
        keys = []
        for trial in manifest["trials"]:
            key = (str(trial["pid"]), int(trial["trial_index"]))
            if key in trials_by_key and trials_by_key[key] != trial:
                raise ValueError(f"inconsistent trial identity {key}")
            trials_by_key[key] = trial
            keys.append(key)
        for index, run in enumerate(manifest["runs"]):
            path = str(Path(run["run_dir"]))
            if path not in runs_by_path:
                runs_by_path[path] = ({key: value for key, value in run.items() if key != "estimates"}, {})
            metadata, estimates = runs_by_path[path]
            if metadata != {key: value for key, value in run.items() if key != "estimates"}:
                raise ValueError(f"inconsistent run provenance {path}")
            for t, key in enumerate(keys):
                if key in estimates:
                    raise ValueError(f"duplicate cache trial {path} {key}")
                estimates[key] = caches[index][t]
    keys = sorted(trials_by_key)
    combined = {"trials": [trials_by_key[key] for key in keys], "runs": []}
    caches = []
    for path in sorted(runs_by_path):
        run, estimates = runs_by_path[path]
        if set(estimates) != set(keys):
            raise ValueError(f"run {path} is missing {len(set(keys) - set(estimates))} trial caches")
        combined["runs"].append(run)
        caches.append([estimates[key] for key in keys])
    return combined, caches


def cognitive_grid_diagnostics(candidates, fitted):
    """Finite-grid nuisance profiles and boundary flags; these are diagnostics."""
    fits = {fit["candidate_id"]: fit for fit in fitted["fits"]}
    report = {}
    for family, selected_id in fitted["selected_by_validation"].items():
        group = [candidate for candidate in candidates if candidate.family == family]
        selected = next(candidate for candidate in group if candidate.candidate_id == selected_id)
        best_score = max(fits[candidate.candidate_id]["total_log_likelihood"] for candidate in group)
        dimensions = {}
        for parameter in sorted({key for candidate in group for key in candidate.parameters}):
            values = sorted({candidate.parameters[parameter] for candidate in group
                             if isinstance(candidate.parameters.get(parameter), (int, float))})
            if not values:
                continue
            profile = [max(fits[c.candidate_id]["total_log_likelihood"] for c in group
                           if c.parameters.get(parameter) == value) for value in values]
            selected_value = selected.parameters.get(parameter)
            support = [value for value, score in zip(values, profile) if score >= best_score - 1.920729410347062]
            dimensions[parameter] = {
                "values": values, "profile_train_log_likelihoods": profile,
                "likelihood_ratio_95_grid_support": support, "selected_value": selected_value,
                "selected_at_grid_boundary": len(values) > 1 and selected_value in (values[0], values[-1]),
                "fixed_on_available_grid": len(values) == 1,
            }
        report[family] = {"candidate_count": len(group), "parameters": dimensions,
                          "interpretation": "finite_policy_grid_diagnostics_not_continuous_parameter_confidence_intervals"}
    return report


def posterior_predictive_checks(candidates, fitted, observed_indices, records, test_indices,
                                *, bootstrap_repetitions=2000, seed=20261011):
    """Held-out full-path expectations and calibration with participant uncertainty."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    fits = {fit["candidate_id"]: fit for fit in fitted["fits"]}
    rows = np.asarray(test_indices, dtype=int)
    pids = np.asarray([str(record["pid"]) for record in records])[rows]
    report = {}
    for family, candidate_id in fitted["selected_by_validation"].items():
        probabilities = by_id[candidate_id].probabilities(fits[candidate_id]["lapse_rate"])[rows]
        # Enumerate paths in the same left-to-right tree order used by TreeProblem.
        metrics = np.zeros((len(rows), probabilities.shape[1], 3))
        for t, index in enumerate(rows):
            record = records[index]
            paths = []

            def visit(node, path):
                if not record["graph"][node]:
                    paths.append(path)
                for child in record["graph"][node]:
                    visit(child, [*path, child])

            visit(record["start"], [])
            rewards = [sum(record["rewards"][node] for node in path) for path in paths]
            for p, (path, reward) in enumerate(zip(paths, rewards)):
                metrics[t, p] = [reward, reward == max(rewards), len(path)]
        expected = np.sum(probabilities[:, :, None] * metrics, axis=1)
        observed = metrics[np.arange(len(rows)), np.asarray(observed_indices)[rows]]
        checks = {}
        for metric, name in enumerate(("path_reward_raw_units", "optimal_path_rate", "path_length")):
            checks[name] = {
                "columns": ["observed", "model_expected", "expected_minus_observed"],
                **participant_bootstrap(np.stack([observed[:, metric], expected[:, metric],
                                                 expected[:, metric] - observed[:, metric]], axis=1),
                                        pids, repetitions=bootstrap_repetitions, seed=seed),
            }
        actual = np.asarray(observed_indices)[rows]
        chosen = probabilities[np.arange(len(rows)), actual]
        brier = np.square(probabilities).sum(axis=1) - 2 * chosen + 1
        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        calibration = []
        for lower in np.arange(0, 1, .1):
            mask = (confidence >= lower) & (confidence < lower + .1 + (1e-12 if lower > .89 else 0))
            if np.any(mask):
                calibration.append({
                    "lower": float(lower), "upper": float(lower + .1),
                    "columns": ["max_path_probability", "top_path_accuracy"],
                    **participant_bootstrap(np.stack([confidence[mask], (predicted[mask] == actual[mask]).astype(float)], 1),
                                            pids[mask], repetitions=bootstrap_repetitions, seed=seed),
                })
        report[family] = {"candidate_id": candidate_id, "metrics": checks,
                          "probability_scores": {"columns": ["observed_path_probability", "multiclass_brier_score"],
                                                 **participant_bootstrap(np.stack([chosen, brier], 1), pids,
                                                                         repetitions=bootstrap_repetitions, seed=seed)},
                          "top_path_calibration": calibration}
    return report


def independent_repeat_diagnostics(manifest, caches, candidate_manifest, candidates, fitted, observed_indices,
                                   split_indices):
    """Check summed log scores at fixed fitted optima using independent MC repeats.

    These repeats vary numerical integration only. Their spread is distinct from
    participant uncertainty and from variability between trained policy seeds.
    """
    repeats = min(len(trial) for run in caches for trial in run)
    repeat_candidates = [candidates_from_cache(manifest, caches, candidate_manifest, repeat_indices=[r])[0]
                         for r in range(repeats)]
    return repeat_score_diagnostics(candidates, repeat_candidates, fitted, observed_indices, split_indices)


def repeat_score_diagnostics(candidates, repeat_candidates, fitted, observed_indices, split_indices):
    """Compare aggregate ensemble likelihoods across independent numerical repeats."""
    repeats = len(repeat_candidates)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    fits = {fit["candidate_id"]: fit for fit in fitted["fits"]}
    totals = {candidate.candidate_id: {split: [] for split in split_indices} for candidate in candidates}
    for repeated_candidates in repeat_candidates:
        for candidate in repeated_candidates:
            for split, indices in split_indices.items():
                totals[candidate.candidate_id][split].append(float(observed_log_probabilities(
                    candidate, fits[candidate.candidate_id]["lapse_rate"], observed_indices, indices,
                ).sum()))
    report = {"independent_repeat_count": repeats, "candidates": {}, "paired_selected_test_gaps": []}
    for candidate_id, splits in totals.items():
        report["candidates"][candidate_id] = {}
        for split, values in splits.items():
            pooled = float(observed_log_probabilities(by_id[candidate_id], fits[candidate_id]["lapse_rate"],
                                                      observed_indices, split_indices[split]).sum())
            report["candidates"][candidate_id][split] = {
                "repeat_total_log_likelihoods": values, "pooled_total_log_likelihood": pooled,
                "repeat_log_likelihood_sd": float(np.std(values, ddof=1)) if repeats > 1 else None,
                "approx_pooled_log_likelihood_mc_se": float(np.std(values, ddof=1) / np.sqrt(repeats))
                if repeats > 1 else None,
                "pooled_minus_mean_repeat_log_likelihood": pooled - float(np.mean(values)),
            }
    selected = list(fitted["selected_by_validation"].items())
    for i, (family, candidate_id) in enumerate(selected):
        for reference_family, reference_id in selected[:i]:
            gaps = np.asarray(totals[candidate_id]["test"]) - np.asarray(totals[reference_id]["test"])
            report["paired_selected_test_gaps"].append({
                "model": family, "reference": reference_family, "repeat_log_likelihood_gaps": gaps.tolist(),
                "approx_pooled_gap_mc_se": float(np.std(gaps, ddof=1) / np.sqrt(repeats)) if repeats > 1 else None,
            })
    return report
