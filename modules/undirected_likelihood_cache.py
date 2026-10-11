"""Reuse uniform planning trajectories for arbitrary Gamma stopping and movement lapse rates."""

from dataclasses import dataclass
from hashlib import sha256
from math import comb
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammainc, gammaincc

from modules.path_likelihood import MIN_LAPSE_RATE, PathLikelihoodEstimator, conditional_path_coefficients
from modules.random_search import RANDOM_SEARCH_STOP_MAX_FIXATIONS


CACHE_VERSION = 1


def gamma_stopping_probabilities(shape, scale, max_fixations):
    """Mass of max(min(floor(Gamma(shape, scale)), M) - 1, 0), including both boundary atoms."""
    if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
        raise ValueError("Gamma shape and scale must be positive and finite")
    if not isinstance(max_fixations, (int, np.integer)) or max_fixations < 1:
        raise ValueError("max_fixations must be a positive integer")
    if max_fixations == 1:
        return np.ones(1)
    boundaries = np.arange(2, max_fixations + 1, dtype=np.float64) / scale
    cdf = gammainc(shape, boundaries)
    sf = gammaincc(shape, boundaries)
    interior = np.where(cdf[:-1] < 0.5, np.diff(cdf), -np.diff(sf))
    probabilities = np.concatenate(([cdf[0]], interior, [sf[-1]]))
    return np.maximum(probabilities, 0) / probabilities.sum()


def lapse_weights(paths, lapse_rate):
    if not MIN_LAPSE_RATE <= lapse_rate <= 1:
        raise ValueError(f"lapse_rate must be in [{MIN_LAPSE_RATE}, 1]")
    degree = max(map(len, paths))
    powers = np.arange(degree + 1)
    lengths = np.asarray([len(path) for path in paths])[:, None]
    binomial = np.asarray([[comb(len(path), k) if k <= len(path) else 0 for k in powers] for path in paths])
    return binomial * lapse_rate**powers * (1 - lapse_rate)**np.maximum(lengths - powers, 0)


@dataclass(frozen=True)
class UndirectedLikelihoodCache:
    paths: tuple[tuple[int, ...], ...]
    mean_coefficients: np.ndarray  # independent repeat x stopping target x path x Bernstein coefficient
    samples_per_repeat: int
    seeds: tuple[int, ...]

    def evaluate_repeats(self, shape, scale, lapse_rate):
        mass = gamma_stopping_probabilities(shape, scale, self.mean_coefficients.shape[1])
        return np.einsum("rtpk,t,pk->rp", self.mean_coefficients, mass, lapse_weights(self.paths, lapse_rate))

    def evaluate(self, shape, scale, lapse_rate):
        """Integrate stopping exactly; estimate MC uncertainty from independent trajectory repeats."""
        repeats = self.evaluate_repeats(shape, scale, lapse_rate)
        mass = gamma_stopping_probabilities(shape, scale, self.mean_coefficients.shape[1])
        probabilities = repeats.mean(axis=0)
        standard_errors = repeats.std(axis=0, ddof=1) / np.sqrt(len(repeats))
        return {
            "gamma_shape": float(shape), "gamma_scale": float(scale), "lapse_rate": float(lapse_rate),
            "paths": [list(path) for path in self.paths], "probabilities": probabilities.tolist(),
            "standard_errors": standard_errors.tolist(), "repeat_probabilities": repeats.tolist(),
            "samples": self.samples_per_repeat * len(repeats), "repeats": len(repeats),
            "mean_latent_fixations": float(mass @ np.arange(len(mass))),
            "maximum_target_fraction": float(mass[-1]), "probability_sum": float(probabilities.sum()),
        }


class UndirectedTrajectoryEstimator:
    def __init__(self, env):
        self.env = env
        self.estimator = PathLikelihoodEstimator(env, undirected=True)
        self.max_fixations = min(env.t_max, RANDOM_SEARCH_STOP_MAX_FIXATIONS)
        self._trajectory_batch = jax.jit(jax.vmap(self._trajectory_coefficients, in_axes=(None,) * 7 + (0,)))

    def _trajectory_coefficients(self, env_params, root, points, children, parents, depths, paths, key):
        policy_key, memory_key, _ = jax.random.split(key, 3)
        state = self.estimator.initial_state(root, points, children, parents, memory_key, env_params)

        def step(carry, target):
            state, policy_key = carry
            coefficients = conditional_path_coefficients(state, env_params, depths, paths)

            def look(carry):
                state, policy_key = carry
                policy_key, action_key = jax.random.split(policy_key)
                fixation = self.estimator.random_search._sample_random_fixation(
                    action_key, self.env._get_action_mask(state), state.fixation_node,
                )
                state = self.env._look(state._replace(time_elapsed=state.time_elapsed + 1), fixation, env_params)
                return state, policy_key

            carry = jax.lax.cond(target < self.max_fixations - 1, look, lambda carry: carry, carry)
            return carry, coefficients

        _, coefficients = jax.lax.scan(step, (state, policy_key), jnp.arange(self.max_fixations))
        return coefficients

    def build_cache(self, problem, env_params, *, samples, repeats=4, batch_size=256, seed=0):
        if not 2 <= samples < 2**32 or batch_size < 1 or repeats < 2:
            raise ValueError("samples must be in [2, 2**32), batch_size positive, and repeats at least two")
        if len(problem.points) != self.env.num_nodes:
            raise ValueError("problem size does not match checkpoint environment")
        if not np.all(np.isin(np.delete(problem.points, problem.root), np.asarray(self.env.point_set))):
            raise ValueError("problem rewards must use the checkpoint's raw point units")
        paths = problem.padded_paths()[:, :max(map(len, problem.paths))]
        args = (env_params, problem.root, problem.points, problem.children, problem.parents, problem.depths, paths)
        means = []
        seeds = tuple(seed + repeat for repeat in range(repeats))
        for repeat_seed in seeds:
            master_key = jax.random.PRNGKey(repeat_seed)
            total = np.zeros((self.max_fixations, len(problem.paths), paths.shape[1] + 1), dtype=np.float64)
            for start in range(0, samples, batch_size):
                keys = jax.vmap(lambda i: jax.random.fold_in(master_key, i))(
                    jnp.arange(start, start + batch_size, dtype=jnp.uint32),
                )
                values = np.asarray(self._trajectory_batch(*args, keys), dtype=np.float64)
                total += values[:min(batch_size, samples - start)].sum(axis=0)
            means.append(total / samples)
        return UndirectedLikelihoodCache(problem.paths, np.stack(means), samples, seeds)


def load_undirected_cache_bundle(manifest_path):
    """Return manifest and caches[run][trial], checking the NPZ hash and loading without pickle."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["cache_version"] != CACHE_VERSION or manifest["method"] != "undirected_gamma_trajectory_cache":
        raise ValueError("unsupported undirected cache format")
    array_path = manifest_path.parent / manifest["arrays_file"]
    if sha256(array_path.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("cache array hash does not match manifest")
    with np.load(array_path, allow_pickle=False) as arrays:
        caches = [[UndirectedLikelihoodCache(
            tuple(tuple(path) for path in trial["paths"]), arrays[entry["key"]],
            entry["samples_per_repeat"], tuple(entry["seeds"]),
        ) for trial, entry in zip(manifest["trials"], run["estimates"], strict=True)] for run in manifest["runs"]]
    return manifest, caches


def load_undirected_candidates(manifest_path, candidate_specs=None, repeat_indices=None):
    """Build fitting candidates from one bundle, optionally selecting independent MC repetition groups."""
    from modules.action_cache_fitting import coefficient_candidates

    manifest, caches = load_undirected_cache_bundle(manifest_path)
    if candidate_specs is None:
        candidate_specs = [{"gamma_shape": shape, "gamma_scale": mean / shape}
                           for shape in (0.5, 1, 2, 4, 8) for mean in (2, 4, 8, 16, 32, 64)]
    specifications, candidate_coefficients = [], []
    maximum_paths = max(len(trial["paths"]) for trial in manifest["trials"])
    maximum_degree = max(len(path) for trial in manifest["trials"] for path in trial["paths"])
    for run, trial_caches in zip(manifest["runs"], caches, strict=True):
        for spec in candidate_specs:
            shape, scale = spec["gamma_shape"], spec["gamma_scale"]
            coefficients = np.zeros((len(trial_caches), maximum_paths, maximum_degree + 1))
            for trial_index, cache in enumerate(trial_caches):
                integrated = integrate_undirected_coefficients(cache, shape, scale, repeat_indices)
                coefficients[trial_index, :integrated.shape[0], :integrated.shape[1]] = integrated
            parameters = {"wm_decay": run["wm_decay"], "wm_neighbor_activation": run["wm_neighbor_activation"],
                          "gamma_shape": shape, "gamma_scale": scale, "gamma_mean": shape * scale}
            identifier = (f"undirected_decay{run['wm_decay']:g}_neighbor{run['wm_neighbor_activation']:g}"
                          f"_shape{shape:g}_scale{scale:g}")
            specifications.append({"id": identifier, "model": "undirected", "parameters": parameters})
            candidate_coefficients.append(coefficients)
    return coefficient_candidates(specifications, candidate_coefficients, manifest["trials"]), manifest["trials"]


def integrate_undirected_coefficients(cache, shape, scale, repeat_indices=None):
    """Collapse a trajectory cache to mean path Bernstein coefficients for one stopping distribution."""
    selected = (cache.mean_coefficients if repeat_indices is None
                else cache.mean_coefficients[list(repeat_indices)])
    if selected.ndim != 4 or selected.shape[0] == 0:
        raise ValueError("repeat_indices must select a nonempty sequence of independent repetitions")
    mass = gamma_stopping_probabilities(shape, scale, selected.shape[1])
    return np.einsum("tpk,t->pk", selected.mean(axis=0), mass)


def load_undirected_trajectory_arrays(
    manifest_paths, wm_decay, wm_neighbor_activation, *, trial_keys=None, repeat_indices=None,
):
    """Stream one WM law into [repeat, target, trial, path, coefficient] arrays.

    Requested participant/trial keys preserve their supplied order; otherwise
    keys are sorted. Returning seeds[repeat, trial] permits explicit checks of
    independent truth and fitting repetition groups during recovery.
    """
    selected_runs = []
    trials_by_key = {}
    for path in manifest_paths:
        path = Path(path)
        manifest = json.loads(path.read_text())
        if (manifest.get("cache_version") != CACHE_VERSION
                or manifest.get("method") != "undirected_gamma_trajectory_cache"):
            raise ValueError("unsupported undirected cache format")
        for run in manifest["runs"]:
            if (not np.isclose(run["wm_decay"], wm_decay, atol=1e-12, rtol=0)
                    or not np.isclose(run["wm_neighbor_activation"], wm_neighbor_activation, atol=1e-12, rtol=0)):
                continue
            selected_runs.append((path, manifest, run))
            for trial in manifest["trials"]:
                key = (str(trial["pid"]), int(trial["trial_index"]))
                if key in trials_by_key and trial != trials_by_key[key]:
                    raise ValueError(f"inconsistent undirected trial metadata: {key}")
                trials_by_key[key] = trial
    if not selected_runs:
        raise ValueError("no cache run matches the requested working-memory parameters")
    keys = (sorted(trials_by_key) if trial_keys is None
            else [(str(pid), int(index)) for pid, index in trial_keys])
    if not keys or len(set(keys)) != len(keys) or any(key not in trials_by_key for key in keys):
        raise ValueError("requested trial keys must be unique, nonempty, and present in the selected WM cache")
    trials = [trials_by_key[key] for key in keys]
    trial_indices = {key: i for i, key in enumerate(keys)}
    maximum_paths = max(len(trial["paths"]) for trial in trials)
    maximum_degree = max(len(path) for trial in trials for path in trial["paths"])
    result = seeds = None
    seen = np.zeros(len(keys), dtype=bool)
    checked_files = set()
    for path, manifest, run in selected_runs:
        array_path = path.parent / manifest["arrays_file"]
        if array_path not in checked_files:
            digest = sha256()
            with array_path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != manifest["arrays_sha256"]:
                raise ValueError(f"cache array hash does not match manifest: {array_path}")
            checked_files.add(array_path)
        with np.load(array_path, allow_pickle=False) as arrays:
            for trial, entry in zip(manifest["trials"], run["estimates"], strict=True):
                key = (str(trial["pid"]), int(trial["trial_index"]))
                if key not in trial_indices:
                    continue
                t = trial_indices[key]
                if seen[t]:
                    raise ValueError(f"duplicate working-memory law and trial cache: {key}")
                raw = arrays[entry["key"]]
                selected = raw if repeat_indices is None else raw[list(repeat_indices)]
                selected_seeds = np.asarray(entry["seeds"], dtype=np.uint64)
                if repeat_indices is not None:
                    selected_seeds = selected_seeds[list(repeat_indices)]
                if selected.ndim != 4 or selected.shape[0] < 1:
                    raise ValueError("repeat_indices must select a nonempty sequence of repetitions")
                if result is None:
                    result = np.zeros((selected.shape[0], selected.shape[1], len(keys), maximum_paths,
                                       maximum_degree + 1), dtype=np.float64)
                    seeds = np.zeros((selected.shape[0], len(keys)), dtype=np.uint64)
                if (selected.shape[:2] != result.shape[:2] or selected.shape[2] != len(trial["paths"])
                        or selected.shape[3] > result.shape[-1] or len(selected_seeds) != result.shape[0]):
                    raise ValueError("inconsistent trajectory dimensions or repetition counts")
                result[:, :, t, :selected.shape[2], :selected.shape[3]] = selected
                seeds[:, t] = selected_seeds
                seen[t] = True
    if not np.all(seen):
        raise ValueError("selected working-memory law is missing requested trial caches")
    return trials, result, seeds


def refine_gamma_coefficients(
    observed_coefficients, path_lengths, starts, *, shape_bounds=(0.1, 100), mean_bounds=(0.1, 1000),
    lapse_bounds=(MIN_LAPSE_RATE, 1), maxiter=300,
):
    """Refine Gamma stopping and lapse from training-only observed-path trajectory coefficients.

    Input shape is [stopping target, training trial, Bernstein coefficient]. Each
    trial's polynomial has its own path length. No validation/test data are
    selected here: the caller must pass only the intended fitting observations.
    Local curvature and multistart agreement diagnose flatness; they are not
    calibrated parameter confidence intervals.
    """
    coefficients = np.asarray(observed_coefficients, dtype=np.float64)
    lengths = np.asarray(path_lengths, dtype=np.int32)
    if coefficients.ndim != 3 or coefficients.shape[0] < 1 or coefficients.shape[1] < 1:
        raise ValueError("observed_coefficients must have shape [target, nonempty trial, coefficient]")
    if lengths.shape != (coefficients.shape[1],) or np.any(lengths < 1) or np.any(lengths >= coefficients.shape[2]):
        raise ValueError("path_lengths must match trials and fit the Bernstein coefficient degree")
    if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0):
        raise ValueError("observed coefficients must be finite and nonnegative")
    bounds = np.asarray([shape_bounds, mean_bounds, lapse_bounds], dtype=float)
    if (bounds.shape != (3, 2) or np.any(~np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(bounds[:2, 0] <= 0) or bounds[2, 0] < MIN_LAPSE_RATE or bounds[2, 1] > 1):
        raise ValueError("invalid Gamma shape, mean, or movement lapse bounds")
    if not starts or maxiter < 1:
        raise ValueError("at least one starting value and positive maxiter are required")
    transformed_bounds = bounds.copy()
    transformed_bounds[:2] = np.log(transformed_bounds[:2])
    powers = np.arange(coefficients.shape[-1])
    binomial = np.asarray([[comb(int(d), int(k)) if k <= d else 0 for k in powers] for d in lengths])
    remaining = np.maximum(lengths[:, None] - powers, 0)

    def objective(point):
        shape, mean = np.exp(point[:2])
        mass = gamma_stopping_probabilities(shape, mean / shape, coefficients.shape[0])
        lapse = point[2]
        weights = binomial * lapse**powers * (1 - lapse)**remaining
        probabilities = mass @ np.einsum("tnk,nk->tn", coefficients, weights)
        if np.any(probabilities <= 0) or np.any(~np.isfinite(probabilities)):
            raise ValueError("observed-path coefficients must imply positive finite probabilities")
        return -float(np.log(probabilities).sum())

    points = []
    for start in starts:
        shape = float(start["gamma_shape"])
        mean = float(start["gamma_mean"] if "gamma_mean" in start else shape * start["gamma_scale"])
        lapse = float(start["lapse_rate"])
        if not np.all(np.isfinite([shape, mean, lapse])) or shape <= 0 or mean <= 0:
            raise ValueError("starting Gamma parameters must be positive and finite, and lapse finite")
        points.append(np.clip([np.log(shape), np.log(mean), lapse], *transformed_bounds.T))
    best_start = min(points, key=objective)
    for dimension in (0, 1):
        for direction in (-1, 1):
            point = best_start.copy()
            point[dimension] += direction * np.log(2)
            points.append(np.clip(point, *transformed_bounds.T))
    points = np.unique(np.round(points, 12), axis=0)
    solutions = []
    for point in points:
        solution = minimize(objective, point, method="L-BFGS-B", bounds=transformed_bounds,
                            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-6})
        # Preserve the initialization if a local optimizer terminates at an inferior value.
        initial_loss = objective(point)
        if not np.isfinite(solution.fun) or solution.fun > initial_loss:
            fitted_point, loss = point, initial_loss
        else:
            fitted_point, loss = solution.x, float(solution.fun)
        solutions.append({"point": fitted_point, "loss": loss, "success": bool(solution.success),
                          "message": str(solution.message), "evaluations": int(solution.nfev)})
    best = min(solutions, key=lambda result: result["loss"])
    point = best["point"]
    shape, mean = np.exp(point[:2])
    mass = gamma_stopping_probabilities(shape, mean / shape, coefficients.shape[0])

    # Use an interior point for the lapse derivative at either optimization bound.
    steps = np.asarray([1e-3, 1e-3, min(1e-4, (bounds[2, 1] - bounds[2, 0]) / 4)])
    center = point.copy()
    center[2] = np.clip(center[2], bounds[2, 0] + steps[2], bounds[2, 1] - steps[2])
    basis = np.diag(steps)
    hessian = np.empty((3, 3))
    center_loss = objective(center)
    for i in range(3):
        hessian[i, i] = (objective(center + basis[i]) - 2 * center_loss + objective(center - basis[i])) / steps[i]**2
        for j in range(i):
            hessian[i, j] = hessian[j, i] = (
                objective(center + basis[i] + basis[j]) - objective(center + basis[i] - basis[j])
                - objective(center - basis[i] + basis[j]) + objective(center - basis[i] - basis[j])
            ) / (4 * steps[i] * steps[j])
    eigenvalues = np.linalg.eigvalsh(hessian)
    names = ("gamma_shape", "gamma_mean", "lapse_rate")
    boundary_flags = {name: {"lower": bool(np.isclose(point[i], transformed_bounds[i, 0], atol=1e-5, rtol=0)),
                             "upper": bool(np.isclose(point[i], transformed_bounds[i, 1], atol=1e-5, rtol=0))}
                      for i, name in enumerate(names)}
    near_best = [solution for solution in solutions if solution["loss"] - best["loss"] <= 1e-3]
    near_best_points = np.stack([solution["point"] for solution in near_best])
    flat = bool(eigenvalues[0] < 1e-3 or eigenvalues[0] / max(eigenvalues[-1], 1e-12) < 1e-6)
    return {
        "gamma_shape": float(shape), "gamma_scale": float(mean / shape), "gamma_mean": float(mean),
        "lapse_rate": float(point[2]), "train_log_likelihood": -best["loss"], "training_trials": len(lengths),
        "bounds": {name: bounds[i].tolist() for i, name in enumerate(names)}, "boundary_flags": boundary_flags,
        "mean_extra_fixations": float(mass @ np.arange(len(mass))),
        "mean_total_fixations": float(1 + mass @ np.arange(len(mass))),
        "maximum_target_fraction": float(mass[-1]), "initial_target_fraction": float(mass[0]),
        "optimizer_success": best["success"], "optimizer_message": best["message"],
        "flatness": {"local_hessian_coordinates": ["log_gamma_shape", "log_gamma_mean", "lapse_rate"],
                     "local_hessian_eigenvalues": eigenvalues.tolist(), "weak_local_curvature": flat,
                     "near_best_start_count": len(near_best),
                     "near_best_log_parameter_spread": np.ptp(near_best_points, axis=0).tolist(),
                     "likelihood_tolerance": 1e-3},
        "starts": [{"gamma_shape": float(np.exp(solution["point"][0])),
                    "gamma_mean": float(np.exp(solution["point"][1])), "lapse_rate": float(solution["point"][2]),
                    "train_log_likelihood": -solution["loss"], "success": solution["success"],
                    "message": solution["message"], "evaluations": solution["evaluations"]}
                   for solution in solutions],
    }
