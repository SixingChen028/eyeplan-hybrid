"""Replay train fitting, validation selection, and frozen test comparison on synthetic actions."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.sparse import csr_matrix

from action_recovery import (
    parameter_distance, sample_paths, summarize_cell, summarize_parameters, validate_probabilities,
)
from modules.action_fitting import fit_lapse, observed_log_probabilities, refine_lapse_profile, select_fitted_candidates


def sparse_responses(observations, indices, path_count):
    columns = (np.asarray(indices)[None, :] * path_count + observations[:, indices]).ravel()
    return csr_matrix((np.ones(len(columns)), columns, np.arange(0, len(columns) + 1, len(indices))),
                      shape=(len(observations), observations.shape[1] * path_count))


def cognitive_match(truth_parameters, fitted_parameters):
    keys = {"wm_decay", "wm_neighbor_activation", "cost"} & truth_parameters.keys() & fitted_parameters.keys()
    return bool(keys) and all(np.isclose(truth_parameters[key], fitted_parameters[key], atol=1e-12, rtol=0)
                              for key in keys)


def coarse_gamma_fits(candidates, observations, split_indices, probabilities_shape, *, profile_points=101,
                      progress=None, continuous_refinement=False):
    """Score the finite Gamma/lapse initialization grid for every synthetic dataset together."""
    matrix = sparse_responses(observations, split_indices["train"], probabilities_shape[1])
    grid = np.linspace(0.01, 1, profile_points)
    records = [[] for _ in observations]
    for candidate_index, candidate in enumerate(candidates):
        logs = np.zeros((len(grid), *probabilities_shape))
        for lapse_index, lapse in enumerate(grid):
            probabilities = validate_probabilities(candidate.probabilities(lapse))
            if probabilities.shape != probabilities_shape:
                raise ValueError("all candidates must share the full actual task-design layout")
            np.log(probabilities, where=probabilities > 0, out=logs[lapse_index])
        flattened = logs.reshape(len(grid), -1).T
        scores = matrix @ flattened
        for dataset, observed in enumerate(observations):
            fit = refine_lapse_profile(candidate, observed, split_indices["train"], grid, scores[dataset])
            fit["profile"] = {"approx_95_interval": fit["profile"]["approx_95_interval"]}
            if not continuous_refinement:
                for name in ("validation", "test"):
                    fit[f"{name}_log_likelihood"] = float(observed_log_probabilities(
                        candidate, fit["lapse_rate"], observed, split_indices[name],
                    ).sum())
            fit["parameter_method"] = "finite_gamma_continuous_lapse_initialization"
            records[dataset].append(fit)
        if progress is not None and (candidate_index % 50 == 0 or candidate_index + 1 == len(candidates)):
            progress({"stage": "gamma_grid", "candidates_scored": candidate_index + 1,
                      "candidates": len(candidates), "datasets": len(observations)})
    return records


def continuous_gamma_fits(paths, trials, observations, split_indices, coarse_records, *, profile_points=101,
                          optimizer_maxiter=300, progress=None, checkpoint_dir=None):
    """Refine every WM law on synthetic training actions with the empirical optimizer."""
    from modules.action_cache_fitting import coefficient_candidates, file_sha256
    from modules.undirected_likelihood_cache import (
        gamma_stopping_probabilities, load_undirected_trajectory_arrays, refine_gamma_coefficients,
    )

    trial_keys = [(str(trial["pid"]), int(trial["trial_index"])) for trial in trials]
    laws = sorted({(record["parameters"]["wm_decay"], record["parameters"]["wm_neighbor_activation"])
                   for record in coarse_records[0]})
    lengths = np.zeros((len(trials), observations.max() + 1), dtype=int)
    for trial_index, trial in enumerate(trials):
        for path_index, path in enumerate(trial["paths"]):
            if path_index < lengths.shape[1]:
                lengths[trial_index, path_index] = len(path)
    results = [[] for _ in observations]
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "observations_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
        "trial_keys": trial_keys, "split_indices": {name: indices.tolist() for name, indices in split_indices.items()},
        "profile_points": profile_points, "optimizer_maxiter": optimizer_maxiter,
        "cache_manifests": [file_sha256(path) for path in paths],
        "source_sha256": {name: file_sha256(Path(__file__).resolve().parent / name) for name in (
            "split_action_recovery.py", "modules/action_fitting.py", "modules/undirected_likelihood_cache.py",
            "modules/action_cache_fitting.py",
        )},
    }
    for law_index, law in enumerate(laws):
        relevant_by_dataset = [[record for record in dataset_records
                                if (record["parameters"]["wm_decay"],
                                    record["parameters"]["wm_neighbor_activation"]) == law]
                               for dataset_records in coarse_records]
        starts_by_dataset = [[{**record["parameters"], "lapse_rate": record["lapse_rate"]}
                              for record in sorted(relevant, key=lambda item: item["total_log_likelihood"],
                                                   reverse=True)[:3]] for relevant in relevant_by_dataset]
        configuration = {**identity, "wm_law": law, "starts": starts_by_dataset}
        configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        checkpoint = checkpoint_dir / f"wm_{law[0]:g}_{law[1]:g}.json" if checkpoint_dir is not None else None
        if checkpoint is not None and checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["configuration_sha256"] != configuration_hash or len(saved["fits"]) != len(observations):
                raise ValueError(f"continuous recovery checkpoint differs from current configuration: {checkpoint}")
            for dataset, fit in enumerate(saved["fits"]):
                results[dataset].append(fit)
            if progress is not None:
                progress({"stage": "continuous_gamma_resumed", "wm_law": list(law),
                          "datasets": len(observations), "checkpoint": str(checkpoint)})
            continue
        _, raw, _ = load_undirected_trajectory_arrays(paths, *law, trial_keys=trial_keys)
        mean = raw.mean(axis=0)
        del raw
        for dataset, observed in enumerate(observations):
            observed_coefficients = mean[:, np.arange(len(trials)), observed]
            observed_lengths = lengths[np.arange(len(trials)), observed]
            train = split_indices["train"]
            started = time.perf_counter()
            refinement = refine_gamma_coefficients(observed_coefficients[:, train], observed_lengths[train],
                                                   starts_by_dataset[dataset],
                                                   maxiter=optimizer_maxiter)
            mass = gamma_stopping_probabilities(refinement["gamma_shape"], refinement["gamma_scale"], mean.shape[0])
            integrated = np.einsum("t,tnpk->npk", mass, mean)
            specification = {
                "id": f"undirected_decay{law[0]:g}_neighbor{law[1]:g}_continuous", "model": "undirected",
                "parameters": {"wm_decay": law[0], "wm_neighbor_activation": law[1],
                               **{key: refinement[key] for key in ("gamma_shape", "gamma_scale", "gamma_mean")}},
            }
            candidate = coefficient_candidates([specification], [integrated], trials)[0]
            fit = fit_lapse(candidate, observed, train, profile_points=profile_points)
            for name in ("validation", "test"):
                fit[f"{name}_log_likelihood"] = float(observed_log_probabilities(
                    candidate, fit["lapse_rate"], observed, split_indices[name],
                ).sum())
            fit["parameter_method"] = "continuous_gamma_and_lapse_train_then_fixed_gamma_lapse_profile"
            fit["gamma_refinement"] = refinement
            results[dataset].append(fit)
            if progress is not None:
                progress({"stage": "continuous_gamma", "wm_law": list(law), "wm_laws_completed": law_index,
                          "wm_laws": len(laws), "dataset": dataset, "datasets": len(observations),
                          "elapsed_seconds": time.perf_counter() - started})
        if checkpoint is not None:
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps({"configuration_sha256": configuration_hash,
                                             "fits": [dataset_records[-1] for dataset_records in results]},
                                            allow_nan=False))
            temporary.replace(checkpoint)
    return results


def split_recovery(candidates, truth_candidates, trials, split, *, truth_lapses, repetitions=20, seed=20261011,
                   profile_points=101, continuous_gamma_paths=None, optimizer_maxiter=300, progress=None,
                   checkpoint_dir=None):
    """Generate full datasets and replay the frozen participant-split fitting procedure.

    Fit candidates use the complete pooled MC precision. Truth candidates must
    come from separately sampled histories, audited by the calling orchestration.
    Every truth uses its supplied fitted lapse; no cross-product is generated.
    """
    if not candidates or not truth_candidates or repetitions < 1 or profile_points < 3:
        raise ValueError("nonempty candidate sets, positive repetitions, and a sufficient lapse profile are required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("fitted candidate IDs must be unique")
    assignment = split["participant_assignment"]
    split_indices = {name: np.asarray([i for i, trial in enumerate(trials)
                                      if assignment[str(trial["pid"])] == name], dtype=int)
                     for name in ("train", "validation", "test")}
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise ValueError("every frozen participant split needs actual task designs")
    if sum(map(len, split_indices.values())) != len(trials):
        raise ValueError("the frozen split must assign every cached trial exactly once")
    rows, observations, scenarios = [], [], []
    probabilities_shape = validate_probabilities(candidates[0].probabilities(0.01)).shape
    if probabilities_shape[0] != len(trials):
        raise ValueError("candidate trial count differs from frozen actual designs")
    for truth in truth_candidates:
        true_lapse = float(truth_lapses[truth.candidate_id])
        if not 0.01 <= true_lapse <= 1:
            raise ValueError("generating lapse must lie in [.01,1]")
        generating = validate_probabilities(truth.probabilities(true_lapse))
        if generating.shape != probabilities_shape:
            raise ValueError("generative and fitted trial/path layouts differ")
        seed_text = json.dumps([seed, "full_frozen_participant_split", truth.candidate_id, true_lapse])
        cell_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
        observations.extend(sample_paths(generating, repetitions, np.random.default_rng(cell_seed)))
        scenarios.extend([(truth, true_lapse, repetition) for repetition in range(repetitions)])
    observations = np.asarray(observations, dtype=int)
    gamma_candidates = [candidate for candidate in candidates if candidate.family == "undirected"]
    ordinary_candidates = [candidate for candidate in candidates if candidate.family != "undirected"]
    coarse_records = coarse_gamma_fits(gamma_candidates, observations, split_indices, probabilities_shape,
                                       profile_points=profile_points, progress=progress,
                                       continuous_refinement=continuous_gamma_paths is not None)
    if continuous_gamma_paths is not None:
        if not gamma_candidates:
            raise ValueError("continuous Gamma replay requires the finite initialization candidates")
        records = continuous_gamma_fits(continuous_gamma_paths, trials, observations, split_indices, coarse_records,
                                        profile_points=profile_points, optimizer_maxiter=optimizer_maxiter,
                                        progress=progress, checkpoint_dir=checkpoint_dir)
    else:
        records = coarse_records
    for dataset, observed in enumerate(observations):
        for candidate in ordinary_candidates:
            fit = fit_lapse(candidate, observed, split_indices["train"], profile_points=profile_points)
            for name in ("validation", "test"):
                fit[f"{name}_log_likelihood"] = float(observed_log_probabilities(
                    candidate, fit["lapse_rate"], observed, split_indices[name],
                ).sum())
            fit["parameter_method"] = "empirical_continuous_lapse_optimizer"
            records[dataset].append(fit)
        # Independent uniform legal children yield 2**(-depth), including unequal path depths.
        random_logs = np.asarray([-len(trial["paths"][path]) * np.log(2)
                                  for trial, path in zip(trials, observed, strict=True)])
        records[dataset].append({
            "candidate_id": "random", "family": "random", "parameters": {}, "lapse_rate": 1.0,
            "total_log_likelihood": float(random_logs[split_indices["train"]].sum()),
            "validation_log_likelihood": float(random_logs[split_indices["validation"]].sum()),
            "test_log_likelihood": float(random_logs[split_indices["test"]].sum()),
            "profile": {"approx_95_interval": [0.01, 1.0]}, "parameter_method": "exact_uniform_legal_children",
        })
        if progress is not None:
            progress({"stage": "ordinary_continuous_lapse", "datasets_completed": dataset + 1,
                      "datasets": len(observations), "ordinary_candidates": len(ordinary_candidates)})
    rng = np.random.default_rng(seed)
    for dataset, (truth, true_lapse, repetition) in enumerate(scenarios):
        selected = select_fitted_candidates(records[dataset])["selected_by_validation"]
        by_id = {record["candidate_id"]: record for record in records[dataset]}
        family_fits = {family: by_id[candidate_id] for family, candidate_id in selected.items()}
        test_scores = {family: fit["test_log_likelihood"] for family, fit in family_fits.items()}
        best_score = max(test_scores.values())
        tied = [family for family, score in test_scores.items() if np.isclose(score, best_score, atol=1e-10, rtol=0)]
        fitted_family = str(rng.choice(tied))
        fitted, within = family_fits[fitted_family], family_fits[truth.family]
        interval = within["profile"]["approx_95_interval"]
        within_candidate = SimpleNamespace(family=truth.family, parameters=within["parameters"])
        rows.append({
            "design": "full_frozen_participant_split", "trials": len(trials), "repetition": repetition,
            "true_candidate": truth.candidate_id, "true_family": truth.family,
            "true_parameters": truth.parameters, "true_lapse": true_lapse,
            "fitted_candidate": fitted["candidate_id"], "fitted_family": fitted_family,
            "fitted_parameters": fitted["parameters"], "fitted_lapse": fitted["lapse_rate"],
            "model_recovered": fitted_family == truth.family,
            "candidate_recovered": fitted_family == truth.family and cognitive_match(truth.parameters,
                                                                                     fitted["parameters"]),
            "within_family_candidate": within["candidate_id"], "within_family_parameters": within["parameters"],
            "within_family_recovered": cognitive_match(truth.parameters, within["parameters"]),
            "within_family_parameter_distance": parameter_distance(truth, within_candidate, candidates),
            "within_family_lapse": within["lapse_rate"], "lapse_interval": interval,
            "interval_covers_truth": bool(interval[0] - 1e-12 <= true_lapse <= interval[1] + 1e-12),
            "numerical_ties": len(tied), "uniform_boundary_identity_degenerate": true_lapse == 1,
            "selected_by_validation": selected, "family_test_log_likelihoods": test_scores,
            "test_gap_to_best_competitor": test_scores[truth.family]
                - max(score for family, score in test_scores.items() if family != truth.family),
            "selected_family_fits": family_fits,
            "observed_path_indices": observations[dataset].tolist(),
        })
    cells = []
    for truth in truth_candidates:
        cell_rows = [row for row in rows if row["true_candidate"] == truth.candidate_id]
        cells.append({"true_candidate": truth.candidate_id, "true_family": truth.family,
                      "true_lapse": truth_lapses[truth.candidate_id], "design": "full_frozen_participant_split",
                      "trials": len(trials), **summarize_cell(cell_rows)})
    return {
        "method": "frozen_participant_split_model_and_parameter_recovery",
        "procedure": "train_continuous_lapse_and_gamma_validation_cognitive_selection_test_family_comparison"
                     if continuous_gamma_paths is not None else
                     "train_continuous_lapse_finite_gamma_approximation_validation_selection_test_comparison",
        "gamma_recovery": "continuous_at_every_working_memory_law" if continuous_gamma_paths is not None
                          else "finite_gamma_initialization_grid_approximation",
        "split": split, "actual_trial_counts": {name: len(indices) for name, indices in split_indices.items()},
        "seed": seed, "repetitions": repetitions, "profile_points": profile_points,
        "cognitive_exact_match": "Shared wm_decay, wm_neighbor_activation, and cost parameters; continuous Gamma "
                                 "parameters are summarized by bias/RMSE rather than exact equality.",
        "interval_method": "Conditional selected-candidate training lapse profile; Gamma is held at its fitted value",
        "interval_caveat": "Likelihood support is not a joint cognitive/Gamma confidence region; empirical coverage "
                           "includes validation-selection effects.",
        "scope": "Homogeneous full-dataset recovery on the frozen participant designs, conditional on the finite "
                 "trained cognitive grid and approximate independent MC-generating distributions.",
        "cells": cells, "parameter_summary": summarize_parameters(rows), "datasets": rows,
    }
