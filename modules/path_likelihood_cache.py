"""Compact planning-MC sufficient statistics for arbitrary movement lapse rates."""

from dataclasses import dataclass
from math import comb
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from modules.path_likelihood import MIN_LAPSE_RATE, conditional_path_coefficients


CACHE_VERSION = 1


@dataclass(frozen=True)
class LapseLikelihoodCache:
    paths: tuple[tuple[int, ...], ...]
    mean_coefficients: np.ndarray
    coefficient_m2: np.ndarray
    samples: int
    seed: int
    mean_latent_fixations: float
    horizon_fraction: float

    @property
    def path_lengths(self):
        return np.asarray([len(path) for path in self.paths], dtype=np.int32)

    def weights(self, lapse_rate):
        if not MIN_LAPSE_RATE <= lapse_rate <= 1:
            raise ValueError(f"lapse_rate must be in [{MIN_LAPSE_RATE}, 1]")
        k = np.arange(self.mean_coefficients.shape[-1])
        degrees = self.path_lengths[:, None]
        binomial = np.asarray([[comb(int(d), int(i)) if i <= d else 0 for i in k]
                               for d in self.path_lengths], dtype=np.float64)
        return binomial * lapse_rate**k * (1 - lapse_rate)**np.maximum(degrees - k, 0)

    def evaluate(self, lapse_rate):
        """Return all leaf probabilities and independent-sample Monte Carlo SEs."""
        weights = self.weights(lapse_rate)
        probabilities = np.sum(self.mean_coefficients * weights, axis=-1)
        variance = np.einsum("pi,pij,pj->p", weights, self.coefficient_m2, weights)
        standard_errors = np.sqrt(np.maximum(variance, 0) / ((self.samples - 1) * self.samples))
        return {
            "lapse_rate": float(lapse_rate), "paths": [list(path) for path in self.paths],
            "probabilities": probabilities.tolist(), "standard_errors": standard_errors.tolist(),
            "samples": self.samples, "seed": self.seed,
            "mean_latent_fixations": self.mean_latent_fixations, "horizon_fraction": self.horizon_fraction,
            "probability_sum": float(probabilities.sum()),
        }


def build_lapse_cache(estimator, problem, policy_params, env_params, *, samples, batch_size=512, seed=0):
    if not 2 <= samples < 2**32 or batch_size < 1:
        raise ValueError("samples must be in [2, 2**32) and batch_size must be positive")
    if len(problem.points) != estimator.env.num_nodes:
        raise ValueError("problem size does not match checkpoint environment")
    if not np.all(np.isin(np.delete(problem.points, problem.root), np.asarray(estimator.env.point_set))):
        raise ValueError("problem rewards must use the checkpoint's raw point units")
    degree = max(map(len, problem.paths))
    # Full binary trees with n nodes have maximum possible depth (n-1)/2.
    # Fixed JIT input width avoids recompilation as the supplied graph changes.
    paths = problem.padded_paths()[:, :(estimator.env.num_nodes - 1) // 2]
    shape = (len(problem.paths), degree + 1)
    mean = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros((*shape, degree + 1), dtype=np.float64)
    # Reuse compilation across problems and repetitions with the same shape.
    if not hasattr(estimator, "_coefficient_batch"):
        def batch(policy_params, env_params, root, points, children, parents, depths, paths, keys):
            states = estimator._sample_batch(policy_params, env_params, root, points, children, parents, keys)
            coefficients = jax.vmap(conditional_path_coefficients, in_axes=(0, None, None, None))(
                states, env_params, depths, paths,
            )
            return coefficients, states.time_elapsed
        estimator._coefficient_batch = jax.jit(batch)
    count = fixation_total = horizon_total = 0
    master_key = jax.random.PRNGKey(seed)
    args = (problem.root, problem.points, problem.children, problem.parents, problem.depths, paths)
    for start in range(0, samples, batch_size):
        keys = jax.vmap(lambda i: jax.random.fold_in(master_key, i))(
            jnp.arange(start, start + batch_size, dtype=jnp.uint32),
        )
        coefficients, fixations = estimator._coefficient_batch(policy_params, env_params, *args, keys)
        n = min(batch_size, samples - start)
        coefficients = np.asarray(coefficients, dtype=np.float64)[:n, :, :degree + 1]
        fixations = np.asarray(fixations)[:n]
        batch_mean = coefficients.mean(axis=0)
        centered = coefficients - batch_mean
        delta = batch_mean - mean
        m2 += np.einsum("npi,npj->pij", centered, centered)
        m2 += np.einsum("pi,pj->pij", delta, delta) * count * n / (count + n)
        count += n
        mean += delta * n / count
        fixation_total += int(fixations.sum())
        horizon_total += int((fixations == estimator.env.t_max - 1).sum())
    return LapseLikelihoodCache(problem.paths, mean, m2, count, seed,
                                fixation_total / count, horizon_total / count)


def load_cache_bundle(manifest_path):
    """Load a manifest and its hash-verified, pickle-free sufficient-statistic arrays.

    Returns (manifest, caches), where caches[run_index][trial_index][repeat_index]
    is a LapseLikelihoodCache. Trial records live in manifest['trials']; run metadata
    and checkpoint identities live in manifest['runs'].
    """
    from hashlib import sha256

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["cache_version"] != CACHE_VERSION:
        raise ValueError("unsupported lapse cache version")
    array_path = manifest_path.parent / manifest["arrays_file"]
    if sha256(array_path.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("cache array hash does not match manifest")
    caches = []
    with np.load(array_path, allow_pickle=False) as arrays:
        for run in manifest["runs"]:
            trials = []
            for trial, repeats in zip(manifest["trials"], run["estimates"], strict=True):
                trials.append([
                    LapseLikelihoodCache(
                        tuple(tuple(path) for path in trial["paths"]),
                        arrays[entry["key"] + "_mean"], arrays[entry["key"] + "_m2"],
                        entry["samples"], entry["seed"], entry["mean_latent_fixations"], entry["horizon_fraction"],
                    ) for entry in repeats
                ])
            caches.append(trials)
    return manifest, caches
