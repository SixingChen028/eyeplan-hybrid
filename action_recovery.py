"""Finite-grid action-only model and parameter recovery from independent MC caches."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


PROFILE_95_CUTOFF = 1.920729410347062


def validate_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be a finite nonnegative trial-by-path array")
    if not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5, rtol=0):
        raise ValueError("each trial's complete path distribution must sum to one")
    return probabilities


def sample_paths(probabilities, repetitions, rng):
    """Draw complete responses from cached generative distributions, without fitting them."""
    probabilities = validate_probabilities(probabilities)
    cumulative = probabilities.cumsum(axis=-1)
    cumulative[:, -1] = 1
    uniforms = rng.random((repetitions, len(probabilities)))
    return (uniforms[:, :, None] >= cumulative[None, :, :]).sum(axis=-1)


def choose_profiled_maxima(scores, counts, rng, families=None):
    """Choose tied families uniformly, then candidate/lapse cells uniformly."""
    eligible = np.isclose(scores, scores.max(axis=-1, keepdims=True), atol=1e-10, rtol=0)
    if families is not None:
        unique = np.unique(families)
        family_ties = np.asarray([np.any(eligible[:, families == family], axis=-1) for family in unique]).T
        priorities = np.where(family_ties, rng.random(family_ties.shape), -1)
        chosen = unique[priorities.argmax(axis=-1)]
        eligible &= families[None, :] == chosen[:, None]
    cumulative = np.cumsum(np.where(eligible, counts, 0), axis=-1)
    draws = rng.random(len(scores)) * cumulative[:, -1]
    return np.sum(cumulative <= draws[:, None], axis=-1)


def profile_interval(profile, lapse_grid):
    """Grid likelihood-support envelope, profiling over candidate cognitive settings."""
    supported = profile >= profile.max(axis=-1, keepdims=True) - PROFILE_95_CUTOFF
    lower = np.min(np.where(supported, lapse_grid, np.inf), axis=-1)
    upper = np.max(np.where(supported, lapse_grid, -np.inf), axis=-1)
    return lower, upper


def parameter_distance(truth, fitted, candidates):
    """RMS distance across shared numeric cognitive parameters, scaled by family ranges."""
    distances = []
    for key, value in truth.parameters.items():
        values = [candidate.parameters.get(key) for candidate in candidates if candidate.family == truth.family]
        if isinstance(value, (int, float)) and all(isinstance(other, (int, float)) for other in values):
            scale = max(values) - min(values)
            if scale > 0:
                distances.append(((fitted.parameters[key] - value) / scale) ** 2)
    return float(np.sqrt(np.mean(distances))) if distances else None


def summarize_cell(rows):
    errors = np.asarray([row["within_family_lapse"] - row["true_lapse"] for row in rows])
    distances = [row["within_family_parameter_distance"] for row in rows
                 if row["within_family_parameter_distance"] is not None]
    global_errors = np.asarray([row["fitted_lapse"] - row["true_lapse"] for row in rows])
    def proportion_interval(key):
        # Wilson uncertainty describes finite synthetic-dataset replication, not MC-cache uncertainty.
        n = len(rows)
        p = np.mean([row[key] for row in rows])
        center = (p + 1.959963984540054**2 / (2 * n)) / (1 + 1.959963984540054**2 / n)
        radius = 1.959963984540054 * np.sqrt(p * (1 - p) / n + 1.959963984540054**2 / (4 * n**2))
        radius /= 1 + 1.959963984540054**2 / n
        return [float(center - radius), float(center + radius)]

    return {
        "datasets": len(rows),
        "model_recovery_rate": float(np.mean([row["model_recovered"] for row in rows])),
        "model_recovery_wilson_95_interval": proportion_interval("model_recovered"),
        "global_candidate_recovery_rate": float(np.mean([row["candidate_recovered"] for row in rows])),
        "within_family_candidate_recovery_rate": float(np.mean([row["within_family_recovered"] for row in rows])),
        "within_family_candidate_recovery_wilson_95_interval": proportion_interval("within_family_recovered"),
        "within_family_mean_parameter_distance": float(np.mean(distances)) if distances else None,
        "within_family_lapse_bias": float(errors.mean()),
        "within_family_lapse_rmse": float(np.sqrt(np.mean(errors**2))),
        "global_lapse_bias": float(global_errors.mean()),
        "global_lapse_rmse": float(np.sqrt(np.mean(global_errors**2))),
        "within_family_profile_interval_coverage": float(np.mean([row["interval_covers_truth"] for row in rows])),
        "within_family_profile_interval_coverage_wilson_95_interval": proportion_interval("interval_covers_truth"),
        "within_family_mean_interval_width": float(np.mean([row["lapse_interval"][1] - row["lapse_interval"][0]
                                                              for row in rows])),
        "model_counts": dict(Counter(row["fitted_family"] for row in rows)),
        "within_family_candidate_counts": dict(Counter(row["within_family_candidate"] for row in rows)),
        "numerical_tie_rate": float(np.mean([row["numerical_ties"] > 1 for row in rows])),
    }


def summarize_parameters(rows):
    result = []
    groups = sorted({(row["true_family"], row["design"], row["true_lapse"]) for row in rows})
    for family, design, lapse in groups:
        group = [row for row in rows if (row["true_family"], row["design"], row["true_lapse"])
                 == (family, design, lapse)]
        keys = sorted(set.intersection(*(set(row["true_parameters"]) for row in group)))
        parameters = {}
        for key in keys:
            truth = [row["true_parameters"][key] for row in group]
            fitted = [row["within_family_parameters"][key] for row in group]
            if not all(isinstance(value, (int, float)) for value in [*truth, *fitted]):
                continue
            truth, fitted = np.asarray(truth), np.asarray(fitted)
            error = fitted - truth
            correlation = float(np.corrcoef(truth, fitted)[0, 1]) if np.ptp(truth) > 0 and np.ptp(fitted) > 0 else None
            parameters[key] = {"bias": float(error.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
                               "correlation": correlation, "truth_values": sorted(set(truth.tolist()))}
        result.append({"family": family, "design": design, "true_lapse": lapse, "parameters": parameters})
    return result


def recovery(candidates, truth_candidates, *, lapse_grid, true_lapses, trial_sets, repetitions=100, seed=0,
             truth_ids=None, progress=None):
    """Run recovery on fixed task designs with independent generative and fitting MC draws.

    Candidates expose candidate_id, family, parameters, and probabilities(lapse).
    trial_sets maps a design label to the actual trial indices used in every
    synthetic dataset in that cell. Identical design sets across true candidates
    make model comparisons interpretable. Cognitive parameters are recovered
    within the true family as well as in the unrestricted model competition.
    """
    if repetitions < 1 or not candidates or not truth_candidates:
        raise ValueError("repetitions and candidate counts must be positive")
    lapse_grid = np.asarray(lapse_grid, dtype=float)
    if lapse_grid.ndim != 1 or len(lapse_grid) < 2 or np.any(np.diff(lapse_grid) <= 0):
        raise ValueError("lapse grid must be a strictly increasing vector")
    if lapse_grid[0] < 0.01 or lapse_grid[-1] > 1 or any(not lapse_grid[0] <= x <= lapse_grid[-1]
                                                       for x in true_lapses):
        raise ValueError("lapse grid and truth must lie in [0.01, 1]")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate ids must be unique")
    from scipy.sparse import csr_matrix, vstack

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected_truths = [candidate for candidate in truth_candidates
                       if truth_ids is None or candidate.candidate_id in truth_ids]
    if not selected_truths or any(truth.candidate_id not in candidates_by_id for truth in selected_truths):
        raise ValueError("generating candidates must be a nonempty subset of the fitted grid")
    families = np.asarray([candidate.family for candidate in candidates])
    rows, cells, identifiability = [], [], []
    rng = np.random.default_rng(seed)
    total_trials = len(candidates[0].probabilities(lapse_grid[0]))
    for design, indices in trial_sets.items():
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1 or len(indices) < 1 or np.any(indices < 0) or np.any(indices >= total_trials):
            raise ValueError("design trial indices must be a nonempty vector in range")
        support = validate_probabilities(candidates[0].probabilities(lapse_grid[0], indices)) > 0
        shape = support.shape
        scenarios, observations, expected_weights = [], [], []
        for truth in selected_truths:
            for true_lapse in true_lapses:
                generating = validate_probabilities(truth.probabilities(true_lapse, indices))
                if generating.shape != shape or np.any((generating > 0) != support):
                    raise ValueError("all candidate distributions must share the same legal path support")
                seed_text = json.dumps([seed, str(design), truth.candidate_id, float(true_lapse)])
                cell_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
                observed = sample_paths(generating, repetitions, np.random.default_rng(cell_seed))
                columns = (np.arange(len(indices))[None, :] * shape[1] + observed).ravel()
                observations.append(csr_matrix((np.ones(len(columns)), columns,
                                                np.arange(0, len(columns) + 1, len(indices))),
                                               shape=(repetitions, np.prod(shape))))
                expected_weights.append(generating.ravel())
                scenarios.append((truth, float(true_lapse)))
        observation_counts = vstack(observations, format="csr")
        del observations
        expected_weights = np.asarray(expected_weights)
        n_datasets = observation_counts.shape[0]
        best_scores = np.empty((n_datasets, len(candidates)))
        best_lapses = np.empty(best_scores.shape, dtype=np.int32)
        tie_counts = np.empty(best_scores.shape, dtype=np.int32)
        profiles = {family: np.full((n_datasets, len(lapse_grid)), -np.inf) for family in set(families)}
        expected_best = np.empty((len(scenarios), len(candidates)))
        started = time.perf_counter()
        for candidate_index, candidate in enumerate(candidates):
            # Only one candidate's [lapse, trial, path] array is resident at a time.
            logs = np.empty((len(lapse_grid), *shape))
            for lapse_index, lapse in enumerate(lapse_grid):
                probabilities = validate_probabilities(candidate.probabilities(lapse, indices))
                if probabilities.shape != shape or np.any((probabilities > 0) != support):
                    raise ValueError("all candidate distributions must share the same legal path support")
                logs[lapse_index] = 0  # Padded paths are structurally absent from sparse response counts.
                np.log(probabilities, where=support, out=logs[lapse_index])
            flattened = logs.reshape(len(lapse_grid), -1).T
            scores = observation_counts @ flattened
            best_scores[:, candidate_index] = scores.max(axis=-1)
            tied = np.isclose(scores, best_scores[:, candidate_index, None], atol=1e-10, rtol=0)
            tie_counts[:, candidate_index] = tied.sum(axis=-1)
            priorities = np.where(tied, rng.random(tied.shape), -1)
            best_lapses[:, candidate_index] = priorities.argmax(axis=-1)
            np.maximum(profiles[candidate.family], scores, out=profiles[candidate.family])
            expected_best[:, candidate_index] = (expected_weights @ flattened).max(axis=-1)
            if progress is not None and (candidate_index % 50 == 0 or candidate_index + 1 == len(candidates)):
                progress({"design": design, "candidates_scored": candidate_index + 1,
                          "candidates": len(candidates), "datasets": n_datasets,
                          "elapsed_seconds": time.perf_counter() - started})
        global_winners = choose_profiled_maxima(best_scores, tie_counts, rng, families)
        global_ties = np.sum(np.where(np.isclose(best_scores, best_scores.max(axis=-1, keepdims=True),
                                                atol=1e-10, rtol=0), tie_counts, 0), axis=-1)
        for scenario, (truth, true_lapse) in enumerate(scenarios):
            dataset_slice = slice(scenario * repetitions, (scenario + 1) * repetitions)
            family_indices = np.flatnonzero(families == truth.family)
            within_winners = choose_profiled_maxima(best_scores[dataset_slice][:, family_indices],
                                                    tie_counts[dataset_slice][:, family_indices], rng)
            within_winners = family_indices[within_winners]
            lower, upper = profile_interval(profiles[truth.family][dataset_slice], lapse_grid)
            cell_rows = []
            for repetition in range(repetitions):
                dataset = scenario * repetitions + repetition
                fitted_index, within_index = global_winners[dataset], within_winners[repetition]
                fitted, within_family = candidates[fitted_index], candidates[within_index]
                row = {
                    "true_candidate": truth.candidate_id, "true_family": truth.family,
                    "true_parameters": truth.parameters, "true_lapse": true_lapse,
                    "design": str(design), "trials": len(indices), "repetition": repetition,
                    "fitted_candidate": fitted.candidate_id, "fitted_family": fitted.family,
                    "fitted_parameters": fitted.parameters,
                    "fitted_lapse": float(lapse_grid[best_lapses[dataset, fitted_index]]),
                    "model_recovered": fitted.family == truth.family,
                    "candidate_recovered": fitted.candidate_id == truth.candidate_id,
                    "within_family_candidate": within_family.candidate_id,
                    "within_family_parameters": within_family.parameters,
                    "within_family_recovered": within_family.candidate_id == truth.candidate_id,
                    "within_family_parameter_distance": parameter_distance(truth, within_family, candidates),
                    "within_family_lapse": float(lapse_grid[best_lapses[dataset, within_index]]),
                    "lapse_interval": [float(lower[repetition]), float(upper[repetition])],
                    "interval_covers_truth": bool(lower[repetition] - 1e-12 <= true_lapse <= upper[repetition] + 1e-12),
                    "numerical_ties": int(global_ties[dataset]),
                }
                cell_rows.append(row)
            rows.extend(cell_rows)
            cells.append({"true_candidate": truth.candidate_id, "true_family": truth.family,
                          "true_lapse": true_lapse, "design": str(design), "trials": len(indices),
                          **summarize_cell(cell_rows)})
            truth_index = next(i for i, candidate in enumerate(candidates)
                               if candidate.candidate_id == truth.candidate_id)
            same_family = [i for i in family_indices if i != truth_index]
            other_family = np.flatnonzero(families != truth.family)
            expected = expected_best[scenario]
            identifiability.append({
                "true_candidate": truth.candidate_id, "true_lapse": true_lapse, "design": str(design),
                "expected_self_minus_best_other_family_log_likelihood":
                    float(expected[truth_index] - expected[other_family].max()) if len(other_family) else None,
                "expected_self_minus_best_other_cognitive_setting_log_likelihood":
                    float(expected[truth_index] - expected[same_family].max()) if same_family else None,
            })
    return {
        "method": "finite_candidate_and_lapse_grid_recovery", "seed": seed, "repetitions": repetitions,
        "synthetic_seed_rule": "sha256 of JSON [seed, design, true_candidate, true_lapse], "
                               "first eight bytes little endian",
        "lapse_grid": lapse_grid.tolist(), "true_lapses": list(true_lapses),
        "trial_sets": {str(key): list(map(int, values)) for key, values in trial_sets.items()},
        "interval_method": "within_true_family_profile_likelihood_grid_envelope_cutoff_1.920729410347062",
        "tie_rule": "Uniform over numerically tied model families, then uniform over tied candidate/lapse cells",
        "interval_caveat": "Nominal 95% support intervals use an asymptotic cutoff; boundaries and discrete nuisance "
                           "settings can change coverage. Coverage is measured empirically here.",
        "scope": "Recovery conditional on the finite trained cognitive grid and cached generative approximations. "
                 "It does not establish recovery for untrained parameter settings.",
        "cells": cells, "parameter_summary": summarize_parameters(rows),
        "identifiability": identifiability, "datasets": rows,
    }


def assert_independent_repeats(caches, truth_repeat, fit_repeats):
    if truth_repeat in fit_repeats:
        raise ValueError("generative and fitting MC repetitions must be disjoint")
    for run in caches:
        for trial in run:
            if truth_repeat >= len(trial) or any(repeat >= len(trial) for repeat in fit_repeats):
                raise ValueError("requested MC repetition missing from cache")
            if trial[truth_repeat].seed in {trial[repeat].seed for repeat in fit_repeats}:
                raise ValueError("generative and fitting caches share MC seeds for a checkpoint/trial")


def assert_independent_manifest_repeats(paths, truth_repeat, fit_repeats):
    """Check RNG separation directly from metadata, without reading covariance arrays."""
    if truth_repeat in fit_repeats:
        raise ValueError("generative and fitting MC repetitions must be disjoint")
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        for run in manifest["runs"]:
            for entries in run["estimates"]:
                seeds = entries["seeds"] if isinstance(entries, dict) else [entry["seed"] for entry in entries]
                if truth_repeat >= len(seeds) or any(repeat >= len(seeds) for repeat in fit_repeats):
                    raise ValueError("requested MC repetition missing from cache")
                if seeds[truth_repeat] in {seeds[repeat] for repeat in fit_repeats}:
                    raise ValueError("generative and fitting caches share MC seeds for a checkpoint/trial")


def average_repeat_candidates(repeat_candidates, selected):
    from modules.action_fitting import Candidate

    if not selected or max(selected) >= len(repeat_candidates):
        raise ValueError("requested fitting repeat missing from cache")
    if len(selected) == 1:
        return repeat_candidates[selected[0]]
    result = []
    for candidate_group in zip(*(repeat_candidates[repeat] for repeat in selected), strict=True):
        reference = candidate_group[0]
        if any(candidate.candidate_id != reference.candidate_id for candidate in candidate_group):
            raise ValueError("candidate ordering differs between MC repetitions")
        def subset(lapse, indices, group=candidate_group):
            return np.mean([candidate.probabilities(lapse, indices) for candidate in group], axis=0)
        result.append(Candidate(reference.candidate_id, reference.family, reference.parameters,
                                lambda lapse, evaluate=subset: evaluate(lapse, None), subset))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, nargs="*")
    parser.add_argument("--undirected-cache", type=Path, nargs="+")
    parser.add_argument("--gamma-shapes", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    parser.add_argument("--gamma-means", type=float, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--scope", default="primary")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--truth-repeat", default=1, type=int)
    parser.add_argument("--fit-repeats", nargs="+", default=[0], type=int)
    parser.add_argument("--truth-ids", nargs="+")
    parser.add_argument("--true-lapses", nargs="+", default=[0.01, 0.1, 0.3, 0.6, 1.0], type=float)
    parser.add_argument("--lapse-step", default=0.005, type=float)
    parser.add_argument("--trial-counts", nargs="+", default=[50, 2500], type=int)
    parser.add_argument("--participant-designs", default=0, type=int,
                        help="Also use this many representative actual participant trial sets")
    parser.add_argument("--repetitions", default=100, type=int)
    parser.add_argument("--seed", default=20261011, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.lapse_step <= 0 or args.truth_repeat < 0 or any(repeat < 0 for repeat in args.fit_repeats):
        parser.error("lapse step must be positive and repeat indices nonnegative")
    if not args.cache and not args.undirected_cache:
        parser.error("at least one learned or undirected cache is required")
    from modules.action_cache_fitting import (
        align_candidate_trials, stream_cache_candidates, stream_undirected_candidates,
    )

    all_cache_paths = args.cache + (args.undirected_cache or [])
    assert_independent_manifest_repeats(all_cache_paths, args.truth_repeat, args.fit_repeats)
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    candidate_manifest = {**candidate_manifest, "candidates": [
        candidate for candidate in candidate_manifest["candidates"] if candidate.get("scope", "primary") == args.scope
    ]}
    if args.undirected_cache:
        candidate_manifest["candidates"] = [candidate for candidate in candidate_manifest["candidates"]
                                             if candidate["model"] != "undirected"]
    if args.cache:
        manifest, pooled, _, repeat_candidates = stream_cache_candidates(args.cache, candidate_manifest)
        del pooled
    else:
        manifest, repeat_candidates = None, []
    if args.undirected_cache:
        undirected_manifest, pooled, undirected_repeats = stream_undirected_candidates(
            args.undirected_cache, args.gamma_shapes, args.gamma_means,
        )
        del pooled
        if manifest is None:
            manifest, repeat_candidates = undirected_manifest, undirected_repeats
        else:
            if len(repeat_candidates) != len(undirected_repeats):
                raise ValueError("learned and undirected caches need matching independent-repeat counts")
            for learned, undirected in zip(repeat_candidates, undirected_repeats, strict=True):
                learned += align_candidate_trials(undirected, undirected_manifest["trials"], manifest["trials"])
    candidates = average_repeat_candidates(repeat_candidates, args.fit_repeats)
    truth_candidates = repeat_candidates[args.truth_repeat]
    if args.truth_ids is not None and not set(args.truth_ids) <= {c.candidate_id for c in truth_candidates}:
        parser.error("unknown truth candidate id")
    rng = np.random.default_rng(args.seed)
    trial_order = rng.permutation(len(manifest["trials"]))
    trial_sets = {}
    for count in args.trial_counts:
        if not 1 <= count <= len(trial_order):
            parser.error("trial counts must be positive and no larger than the cached task-design pool")
        trial_sets[f"fixed_{count}_trial_design"] = sorted(trial_order[:count].tolist())
    participants = sorted({str(trial["pid"]) for trial in manifest["trials"]})
    if args.participant_designs < 0:
        parser.error("participant-designs must be nonnegative")
    for pid in participants[:args.participant_designs]:
        trial_sets[f"participant_{pid}"] = [i for i, trial in enumerate(manifest["trials"])
                                             if str(trial["pid"]) == pid]
    lapse_grid = np.unique(np.r_[np.arange(0.01, 1, args.lapse_step), 1.0, args.true_lapses])
    result = recovery(candidates, truth_candidates, lapse_grid=lapse_grid, true_lapses=args.true_lapses,
                      trial_sets=trial_sets, repetitions=args.repetitions, seed=args.seed, truth_ids=args.truth_ids,
                      progress=lambda value: print(json.dumps(value), flush=True))
    result["provenance"] = {
        "cache_manifests": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in all_cache_paths],
        "truth_repeat": args.truth_repeat, "fit_repeats": args.fit_repeats, "scope": args.scope,
        "gamma_grid": {"shapes": args.gamma_shapes, "means": args.gamma_means}
        if args.undirected_cache else None,
        "candidate_manifest_sha256": hashlib.sha256(args.candidate_manifest.read_bytes()).hexdigest()
        if args.candidate_manifest else None,
        "independent_generative_and_fitting_mc": True,
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in ["action_recovery.py", "modules/action_fitting.py",
                                       "modules/action_cache_fitting.py"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} datasets={len(result['datasets'])}", flush=True)


if __name__ == "__main__":
    main()
