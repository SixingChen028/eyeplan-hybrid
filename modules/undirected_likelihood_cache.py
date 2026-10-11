"""Reuse uniform planning trajectories for arbitrary Gamma stopping and movement lapse rates."""

from dataclasses import dataclass
from hashlib import sha256
from math import comb
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
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
    from modules.action_fitting import Candidate

    manifest, caches = load_undirected_cache_bundle(manifest_path)
    if candidate_specs is None:
        candidate_specs = [{"gamma_shape": shape, "gamma_scale": mean / shape}
                           for shape in (0.5, 1, 2, 4, 8) for mean in (2, 4, 8, 16, 32, 64)]
    candidates = []
    maximum_paths = max(len(trial["paths"]) for trial in manifest["trials"])
    maximum_degree = max(len(path) for trial in manifest["trials"] for path in trial["paths"])
    path_lengths = np.zeros((len(manifest["trials"]), maximum_paths), dtype=np.int32)
    valid_paths = np.zeros_like(path_lengths, dtype=bool)
    for trial_index, trial in enumerate(manifest["trials"]):
        lengths = [len(path) for path in trial["paths"]]
        path_lengths[trial_index, :len(lengths)] = lengths
        valid_paths[trial_index, :len(lengths)] = True
    powers = np.arange(maximum_degree + 1)
    binomial = np.asarray([[[comb(int(length), int(k)) if k <= length else 0 for k in powers]
                            for length in lengths] for lengths in path_lengths])

    def evaluator(coefficients):
        def evaluate(lapse_rate):
            if not MIN_LAPSE_RATE <= lapse_rate <= 1:
                raise ValueError(f"lapse_rate must be in [{MIN_LAPSE_RATE}, 1]")
            weights = binomial * lapse_rate**powers * (1 - lapse_rate)**np.maximum(
                path_lengths[..., None] - powers, 0,
            )
            return np.sum(coefficients * weights, axis=-1) * valid_paths
        return evaluate

    for run_index, (run, trial_caches) in enumerate(zip(manifest["runs"], caches, strict=True)):
        for spec in candidate_specs:
            shape, scale = spec["gamma_shape"], spec["gamma_scale"]
            coefficients = np.zeros((len(trial_caches), maximum_paths, maximum_degree + 1))
            for trial_index, cache in enumerate(trial_caches):
                selected = (cache.mean_coefficients if repeat_indices is None
                            else cache.mean_coefficients[list(repeat_indices)])
                if selected.ndim != 4 or selected.shape[0] == 0:
                    raise ValueError("repeat_indices must select a nonempty sequence of independent repetitions")
                mass = gamma_stopping_probabilities(shape, scale, selected.shape[1])
                integrated = np.einsum("tpk,t->pk", selected.mean(axis=0), mass)
                coefficients[trial_index, :integrated.shape[0], :integrated.shape[1]] = integrated
            parameters = {"wm_decay": run["wm_decay"], "wm_neighbor_activation": run["wm_neighbor_activation"],
                          "gamma_shape": shape, "gamma_scale": scale, "gamma_mean": shape * scale}
            identifier = (f"undirected_decay{run['wm_decay']:g}_neighbor{run['wm_neighbor_activation']:g}"
                          f"_shape{shape:g}_scale{scale:g}")
            candidates.append(Candidate(identifier, "undirected", parameters, evaluator(coefficients)))
    return candidates, manifest["trials"]
