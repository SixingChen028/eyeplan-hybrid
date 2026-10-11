"""Continuous Gamma stopping and movement-lapse recovery conditional on one WM law."""

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import subprocess
import time

import numpy as np

from action_recovery import assert_independent_manifest_repeats, sample_paths, validate_probabilities


def execution_weights(trials, lapse, maximum_paths, coefficients):
    weights = np.zeros((len(trials), maximum_paths, coefficients))
    for trial_index, trial in enumerate(trials):
        for path_index, path in enumerate(trial["paths"]):
            degree = len(path)
            weights[trial_index, path_index, :degree + 1] = [
                comb(degree, k) * lapse**k * (1 - lapse)**(degree - k) for k in range(degree + 1)
            ]
    return weights


def screened_starts(observed_coefficients, lengths, gamma_shapes, gamma_means, count=3):
    from modules.undirected_likelihood_cache import gamma_stopping_probabilities

    settings = [(float(shape), float(mean)) for shape in gamma_shapes for mean in gamma_means]
    masses = np.asarray([gamma_stopping_probabilities(shape, mean / shape, observed_coefficients.shape[0])
                         for shape, mean in settings])
    integrated = np.einsum("gt,tnk->gnk", masses, observed_coefficients)
    powers = np.arange(observed_coefficients.shape[-1])
    binomial = np.asarray([[comb(int(degree), int(k)) if k <= degree else 0 for k in powers] for degree in lengths])
    lapse_grid = np.asarray([0.01, 0.1, 0.3, 0.6, 1])
    weights = binomial[None] * lapse_grid[:, None, None]**powers
    weights *= (1 - lapse_grid[:, None, None])**np.maximum(lengths[:, None] - powers, 0)[None]
    probabilities = np.einsum("gnk,lnk->gln", integrated, weights)
    if np.any(probabilities <= 0):
        raise ValueError("screened observed-path probabilities must be positive")
    scores = np.log(probabilities).sum(axis=-1)
    lapse_indices = scores.argmax(axis=-1)
    order = np.argsort(-scores[np.arange(len(settings)), lapse_indices], kind="stable")[:count]
    return [{"gamma_shape": settings[index][0], "gamma_mean": settings[index][1],
             "lapse_rate": float(lapse_grid[lapse_indices[index]])} for index in order]


def summarize(rows):
    metrics = ("gamma_shape", "gamma_scale", "gamma_mean", "lapse_rate", "mean_total_fixations",
               "maximum_target_fraction")
    summary = {"datasets": len(rows), "parameters": {}, "boundary_rates": {}}
    for parameter in metrics:
        truth = np.asarray([row["truth"][parameter] for row in rows])
        fitted = np.asarray([row["fit"][parameter] for row in rows])
        error = fitted - truth
        summary["parameters"][parameter] = {
            "bias": float(error.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
            "correlation": float(np.corrcoef(truth, fitted)[0, 1])
            if np.ptp(truth) > 0 and np.ptp(fitted) > 0 else None,
        }
    for parameter in ("gamma_shape", "gamma_mean", "lapse_rate"):
        summary["boundary_rates"][parameter] = {
            boundary: float(np.mean([row["fit"]["boundary_flags"][parameter][boundary] for row in rows]))
            for boundary in ("lower", "upper")
        }
    summary["optimizer_success_rate"] = float(np.mean([row["fit"]["optimizer_success"] for row in rows]))
    summary["weak_local_curvature_rate"] = float(np.mean([row["fit"]["flatness"]["weak_local_curvature"]
                                                          for row in rows]))
    return summary


def continuous_recovery(trials, truth_coefficients, fitting_coefficients, *, true_settings, trial_sets,
                        repetitions=20, seed=20261011, gamma_shapes=(0.5, 1, 2, 4, 8),
                        gamma_means=(2, 4, 8, 16, 32, 64), optimizer_maxiter=300, progress=None):
    """Fit independent synthetic responses with the empirical continuous stopping optimizer.

    Both coefficient arrays have shape [target, trial, path, Bernstein coefficient].
    Their independent MC repetition routing must be checked by the caller.
    """
    from modules.undirected_likelihood_cache import gamma_stopping_probabilities, refine_gamma_coefficients

    truth_coefficients, fitting_coefficients = map(np.asarray, (truth_coefficients, fitting_coefficients))
    if (truth_coefficients.ndim != 4 or truth_coefficients.shape != fitting_coefficients.shape
            or truth_coefficients.shape[1] != len(trials)):
        raise ValueError("independent trajectory arrays must share target/trial/path/coefficient layout")
    if repetitions < 1 or not true_settings:
        raise ValueError("positive repetitions and nonempty true settings are required")
    rng = np.random.default_rng(seed)
    rows, cells = [], []
    for design, indices in trial_sets.items():
        indices = np.asarray(indices, dtype=int)
        if len(indices) < 1 or np.any(indices < 0) or np.any(indices >= len(trials)):
            raise ValueError("design trial indices must be nonempty and in range")
        design_trials = [trials[index] for index in indices]
        generating_coefficients = truth_coefficients[:, indices]
        fitted_coefficients = fitting_coefficients[:, indices]
        for setting in true_settings:
            shape, mean, lapse = (float(setting[key]) for key in ("gamma_shape", "gamma_mean", "lapse_rate"))
            if not 0.1 <= shape <= 100 or not 0.1 <= mean <= 1000 or not 0.01 <= lapse <= 1:
                raise ValueError("true continuous parameters must lie within the fitted optimizer bounds")
            mass = gamma_stopping_probabilities(shape, mean / shape, generating_coefficients.shape[0])
            weights = execution_weights(design_trials, lapse, generating_coefficients.shape[2],
                                        generating_coefficients.shape[3])
            probabilities = np.einsum("t,tnpk,npk->np", mass, generating_coefficients, weights)
            probabilities = validate_probabilities(probabilities)
            observations = sample_paths(probabilities, repetitions, rng)
            truth = {"gamma_shape": shape, "gamma_mean": mean, "gamma_scale": mean / shape, "lapse_rate": lapse,
                     "mean_total_fixations": float(1 + mass @ np.arange(len(mass))),
                     "maximum_target_fraction": float(mass[-1])}
            cell_rows = []
            for repetition, observed in enumerate(observations):
                observed_coefficients = fitted_coefficients[:, np.arange(len(indices)), observed]
                lengths = np.asarray([len(trial["paths"][path]) for trial, path in zip(design_trials, observed)])
                starts = screened_starts(observed_coefficients, lengths, gamma_shapes, gamma_means)
                started = time.perf_counter()
                fit = refine_gamma_coefficients(observed_coefficients, lengths, starts, maxiter=optimizer_maxiter)
                row = {"design": str(design), "trials": len(indices), "repetition": repetition,
                       "truth": truth, "fit": fit, "observed_path_indices": observed.tolist(),
                       "elapsed_seconds": time.perf_counter() - started}
                cell_rows.append(row)
                if progress is not None:
                    progress({"design": design, "truth": truth, "repetition": repetition,
                              "elapsed_seconds": row["elapsed_seconds"]})
            rows.extend(cell_rows)
            cells.append({"design": str(design), "truth": truth, **summarize(cell_rows)})
    return {
        "method": "continuous_gamma_and_lapse_recovery_conditional_on_fixed_wm",
        "scope": "Continuous stopping/lapse recovery at a fixed working-memory law, conditional on approximate "
                 "independent MC-generating distributions. Joint cognitive/model recovery is a separate diagnostic.",
        "seed": seed, "repetitions": repetitions,
        "trial_sets": {str(key): list(map(int, indices)) for key, indices in trial_sets.items()},
        "start_rule": "Three highest-scoring distinct Gamma grid settings on lapse grid [.01,.1,.3,.6,1], "
                      "plus optimizer-generated neighboring starts",
        "screen_gamma_shapes": list(gamma_shapes), "screen_gamma_means": list(gamma_means),
        "interval_caveat": "Local curvature and multistart agreement diagnose flatness; no calibrated Gamma "
                           "parameter confidence intervals are claimed.",
        "cells": cells, "summary_by_design": [{"design": design, **summarize([row for row in rows
                                                if row["design"] == design])} for design in trial_sets],
        "datasets": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", type=Path)
    parser.add_argument("--wm-decay", required=True, type=float)
    parser.add_argument("--wm-neighbor-activation", required=True, type=float)
    parser.add_argument("--truth-gamma", action="append", nargs=2, type=float, metavar=("SHAPE", "MEAN"), required=True)
    parser.add_argument("--true-lapses", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--trial-counts", nargs="+", type=int, default=[50, 2500])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--truth-repeat", type=int, default=1)
    parser.add_argument("--fit-repeats", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--optimizer-maxiter", type=int, default=300)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.truth_repeat < 0 or any(repeat < 0 for repeat in args.fit_repeats):
        parser.error("repeat indices must be nonnegative")
    from modules.undirected_likelihood_cache import load_undirected_trajectory_arrays

    assert_independent_manifest_repeats(args.cache, args.truth_repeat, args.fit_repeats)
    trial_records = {}
    for path in args.cache:
        for trial in json.loads(path.read_text())["trials"]:
            key = (str(trial["pid"]), int(trial["trial_index"]))
            if key in trial_records and trial_records[key] != trial:
                raise ValueError("cache shards have inconsistent trial identities")
            trial_records[key] = trial
    all_keys = sorted(trial_records)
    if any(not 1 <= count <= len(all_keys) for count in args.trial_counts):
        parser.error("trial counts must be positive and no larger than the task-design pool")
    order = np.random.default_rng(args.seed).permutation(len(all_keys))
    selected_keys = [all_keys[index] for index in order[:max(args.trial_counts)]]
    repeats = [args.truth_repeat, *args.fit_repeats]
    trials, arrays, seeds = load_undirected_trajectory_arrays(
        args.cache, args.wm_decay, args.wm_neighbor_activation, trial_keys=selected_keys, repeat_indices=repeats,
    )
    if arrays.shape[0] != len(repeats) or np.any(seeds[0, None] == seeds[1:]):
        raise ValueError("generative and fitting trajectory repetitions overlap")
    true_settings = [{"gamma_shape": shape, "gamma_mean": mean, "lapse_rate": lapse}
                     for shape, mean in args.truth_gamma for lapse in args.true_lapses]
    result = continuous_recovery(
        trials, arrays[0], arrays[1:].mean(axis=0), true_settings=true_settings,
        trial_sets={f"fixed_{count}_trial_design": list(range(count)) for count in args.trial_counts},
        repetitions=args.repetitions, seed=args.seed, optimizer_maxiter=args.optimizer_maxiter,
        progress=lambda value: print(json.dumps(value), flush=True),
    )
    result["working_memory_law"] = {"wm_decay": args.wm_decay, "wm_neighbor_activation": args.wm_neighbor_activation}
    result["provenance"] = {
        "cache_manifests": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in args.cache],
        "truth_repeat": args.truth_repeat, "fit_repeats": args.fit_repeats,
        "selected_trial_keys": [list(key) for key in selected_keys],
        "independent_generative_and_fitting_mc": True,
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in ["continuous_gamma_recovery.py", "modules/undirected_likelihood_cache.py"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"wrote={args.output} datasets={len(result['datasets'])}", flush=True)


if __name__ == "__main__":
    main()
