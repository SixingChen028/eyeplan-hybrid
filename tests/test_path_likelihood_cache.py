"""Check reusable lapse statistics against independently evaluated conditional paths."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.config import ENV_STATIC_PARAM_KEYS, load_canonical_defaults
from modules.compat import COMPAT_VERSION
from modules.environment import DecisionTreeEnv
from modules.path_likelihood import PathLikelihoodEstimator, TreeProblem, conditional_path_coefficients
from modules.path_likelihood_cache import LapseLikelihoodCache, load_cache_bundle


def setup_problem():
    _, defaults = load_canonical_defaults()
    defaults.update(num_nodes=5, t_max=4)
    env = DecisionTreeEnv(**{key: defaults[key] for key in ENV_STATIC_PARAM_KEYS})
    problem = TreeProblem.from_record({
        "graph": [[], [0, 4], [1, 3], [], []], "start": 2, "rewards": [-8, 4, 0, 2, -4],
    })
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    return env, problem, params


def test_cached_means_and_covariances_match_direct_mc_for_all_lapses():
    env, problem, params = setup_problem()
    estimator = PathLikelihoodEstimator(env, undirected=True)
    cache = estimator.estimate_lapse_cache(problem, {}, params, samples=97, batch_size=32, seed=12)
    other_batch = estimator.estimate_lapse_cache(problem, {}, params, samples=97, batch_size=23, seed=12)
    np.testing.assert_allclose(cache.mean_coefficients, other_batch.mean_coefficients, atol=1e-12)
    np.testing.assert_allclose(cache.coefficient_m2, other_batch.coefficient_m2, atol=1e-12)
    for lapse in [0.01, 0.13, 0.5, 0.99, 1]:
        direct = PathLikelihoodEstimator(env, undirected=True, lapse_rate=lapse).estimate(
            problem, {}, params, samples=97, batch_size=32, seed=12,
        )
        actual = cache.evaluate(lapse)
        np.testing.assert_allclose(actual["probabilities"], direct["probabilities"], atol=5e-8)
        np.testing.assert_allclose(actual["standard_errors"], direct["standard_errors"], atol=5e-9)
        assert actual["probability_sum"] == pytest.approx(1, abs=1e-7)
    uniform = cache.evaluate(1)
    np.testing.assert_array_equal(uniform["probabilities"], [0.25, 0.25, 0.5])
    np.testing.assert_array_equal(uniform["standard_errors"], 0)


def test_bernstein_coefficients_match_independent_degree_seven_product():
    # Comb tree includes degrees 1 through 7, exposing padded-path errors.
    # Make internal nodes 0,1,3,5,7,9,11 with each right sibling a leaf.
    graph = [[] for _ in range(15)]
    for depth, node in enumerate([0, 1, 3, 5, 7, 9, 11]):
        graph[node] = [2 * depth + 1, 2 * depth + 2]
    problem = TreeProblem.from_record({"graph": graph, "start": 0, "rewards": [0] + [8, -8] * 7})
    _, defaults = load_canonical_defaults()
    defaults.update(num_nodes=15, t_max=4)
    env = DecisionTreeEnv(**{key: defaults[key] for key in ENV_STATIC_PARAM_KEYS})
    params = env.make_params(wm_decay=1, wm_neighbor_activation=1, cost=0)
    state = PathLikelihoodEstimator(env).initial_state(
        problem.root, jnp.asarray(problem.points), jnp.asarray(problem.children), jnp.asarray(problem.parents),
        jax.random.PRNGKey(0), params,
    )._replace(activation=jnp.ones(15), q_values=jnp.asarray(problem.points))
    coeffs = np.asarray(conditional_path_coefficients(state, params, problem.depths, problem.padded_paths()[:, :7]))
    cache = LapseLikelihoodCache(problem.paths, coeffs, np.zeros((8, 8, 8)), 2, 0, 0, 0)
    assert np.all(coeffs >= 0)
    for lapse in [0.01, 0.333, 0.999, 1]:
        expected = []
        for path in problem.paths:
            expected.append(np.prod([1 - lapse / 2 if node % 2 else lapse / 2 for node in path]))
        np.testing.assert_allclose(cache.evaluate(lapse)["probabilities"], expected, atol=1e-8)
        assert cache.evaluate(lapse)["probability_sum"] == pytest.approx(1, abs=1e-8)


def test_cache_cli_shards_keep_rng_identity_and_hash_protection(tmp_path):
    env, problem, params = setup_problem()
    _, defaults = load_canonical_defaults()
    run_args = {key: defaults[key] for key in ENV_STATIC_PARAM_KEYS}
    run_args.update(num_nodes=5, t_max=4, wm_decay=0.8, wm_neighbor_activation=0.4, cost=0,
                    lesion_policy="random_search_gamma_stopping")
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"args": run_args, "compat_version": COMPAT_VERSION}))
    record = {"graph": [[], [0, 4], [1, 3], [], []], "start": 2,
              "rewards": [-8, 4, 0, 2, -4], "moves": [1, 0], "pid": "p1"}
    data = tmp_path / "data.json"
    data.write_text(json.dumps([record, record]))
    output = tmp_path / "cache.json"
    subprocess.run([
        sys.executable, "lapse_likelihood_cache.py", str(run), "--data", str(data), "--output", str(output),
        "--samples", "8", "--batch-size", "4", "--repeats", "2", "--trial-start", "1", "--limit", "1",
    ], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    manifest, caches = load_cache_bundle(output)
    assert manifest["trials"][0]["record_index"] == 1
    assert manifest["trials"][0]["observed_index"] == 0
    assert manifest["trials"][0]["pid"] == "p1"
    identity = manifest["runs"][0]["checkpoint_identity"]
    expected_seeds = [int.from_bytes(hashlib.sha256(f"0:{identity}:1:{r}".encode()).digest()[:4], "little")
                      for r in range(2)]
    assert [cache.seed for cache in caches[0][0]] == expected_seeds
    assert caches[0][0][0].evaluate(0.2)["probability_sum"] == pytest.approx(1, abs=1e-7)
    assert manifest["source_sha256"]["modules/path_likelihood_cache.py"]
    with output.with_suffix(".npz").open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_cache_bundle(output)


def test_fixed_jit_width_preserves_draws_and_compact_statistics_across_tree_depths():
    _, defaults = load_canonical_defaults()
    defaults.update(num_nodes=15, t_max=3)
    env = DecisionTreeEnv(**{key: defaults[key] for key in ENV_STATIC_PARAM_KEYS})
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    estimator = PathLikelihoodEstimator(env, undirected=True)
    graphs = [[[2 * i + 1, 2 * i + 2] if i < 7 else [] for i in range(15)], [[] for _ in range(15)]]
    for depth, node in enumerate([0, 1, 3, 5, 7, 9, 11]):
        graphs[1][node] = [2 * depth + 1, 2 * depth + 2]
    for graph in graphs:
        problem = TreeProblem.from_record({"graph": graph, "start": 0, "rewards": [0] + [8, -8] * 7})
        cache = estimator.estimate_lapse_cache(problem, {}, params, samples=17, batch_size=8, seed=3)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(3), i))(jnp.arange(17, dtype=jnp.uint32))
        states = estimator._sample_batch({}, params, problem.root, problem.points,
                                         problem.children, problem.parents, keys)
        degree = max(map(len, problem.paths))
        compact = np.asarray(jax.jit(jax.vmap(conditional_path_coefficients, in_axes=(0, None, None, None)))(
            states, params, problem.depths, problem.padded_paths()[:, :degree],
        ), dtype=np.float64)
        centered = compact - compact.mean(axis=0)
        assert cache.mean_coefficients.shape == (8, degree + 1)
        np.testing.assert_allclose(cache.mean_coefficients, compact.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(cache.coefficient_m2, np.einsum("npi,npj->pij", centered, centered), atol=1e-12)
    assert estimator._coefficient_batch._cache_size() == 1


def test_estimator_reuse_keeps_dynamic_parameters_and_checkpoints_out_of_static_key():
    from lapse_likelihood_cache import estimator_for_run

    _, args = load_canonical_defaults()
    args.update(num_nodes=5, t_max=4)
    estimators = {}
    first = estimator_for_run(args, estimators)
    other_fit = {**args, "wm_decay": 0.3, "cost": 0.02, "seed": 12}
    assert estimator_for_run(other_fit, estimators) is first
    for key, value in [("t_max", 5), ("hidden_size", args["hidden_size"] + 1),
                       ("lesion_policy", "random_search_gamma_stopping")]:
        assert estimator_for_run({**args, key: value}, estimators) is not first
