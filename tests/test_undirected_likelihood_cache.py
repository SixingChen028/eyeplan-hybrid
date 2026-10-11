import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma

from modules.compat import COMPAT_VERSION
from modules.config import ENV_STATIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import DecisionTreeEnv
from modules.path_likelihood import PathLikelihoodEstimator, TreeProblem, conditional_path_probabilities
from modules.undirected_likelihood_cache import (
    UndirectedTrajectoryEstimator, gamma_stopping_probabilities, lapse_weights, load_undirected_cache_bundle,
    load_undirected_candidates,
)


def environment(n=5, horizon=5):
    _, defaults = load_canonical_defaults()
    defaults.update(num_nodes=n, t_max=horizon)
    return DecisionTreeEnv(**{key: defaults[key] for key in ENV_STATIC_PARAM_KEYS})


def problem():
    return TreeProblem.from_record({
        "graph": [[], [0, 4], [1, 3], [], []], "start": 2, "rewards": [-8, 4, 0, 2, -4],
    })


@pytest.mark.parametrize("shape,scale", [(0.2, 0.1), (1.6619370843815855, 9.364300515304453), (8, 20)])
@pytest.mark.parametrize("maximum", [1, 2, 5, 100])
def test_gamma_mass_matches_rounded_capped_distribution(shape, scale, maximum):
    actual = gamma_stopping_probabilities(shape, scale, maximum)
    assert len(actual) == maximum
    assert actual.sum() == pytest.approx(1)
    assert np.all(actual >= 0)
    if maximum == 1:
        np.testing.assert_array_equal(actual, [1])
    else:
        expected = [gamma.cdf(2, shape, scale=scale)]
        expected += [gamma.cdf(k + 2, shape, scale=scale) - gamma.cdf(k + 1, shape, scale=scale)
                     for k in range(1, maximum - 1)]
        expected += [gamma.sf(maximum, shape, scale=scale)]
        np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_trajectory_snapshots_match_ordinary_gamma_fixation_law():
    env = environment()
    tree = problem()
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    estimator = UndirectedTrajectoryEstimator(env)
    paths = tree.padded_paths()[:, :max(map(len, tree.paths))]
    keys = jax.random.split(jax.random.PRNGKey(7), 16)
    coefficients = np.asarray(estimator._trajectory_batch(
        params, tree.root, tree.points, tree.children, tree.parents, tree.depths, paths, keys,
    ))
    for target in range(env.t_max):
        ordinary = PathLikelihoodEstimator(env, undirected=True)
        ordinary.random_search._fixation_target = lambda key: jnp.int32(target)
        states = ordinary._sample_batch({}, params, tree.root, tree.points, tree.children, tree.parents, keys)
        expected = np.asarray(jax.vmap(conditional_path_probabilities, in_axes=(0, None, None, None, None))(
            states, params, tree.depths, paths, 0.2,
        ))
        actual = np.einsum("npk,pk->np", coefficients[:, target], lapse_weights(tree.paths, 0.2))
        np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_cache_normalization_mc_accuracy_and_batch_invariance():
    env = environment(3, 2)
    tree = TreeProblem.from_record({"graph": [[1, 2], [], []], "start": 0, "rewards": [0, 8, -8]})
    params = env.make_params(wm_decay=1, wm_neighbor_activation=1, cost=0)
    estimator = UndirectedTrajectoryEstimator(env)
    a = estimator.build_cache(tree, params, samples=1024, repeats=4, batch_size=256, seed=9)
    b = estimator.build_cache(tree, params, samples=1024, repeats=4, batch_size=127, seed=9)
    np.testing.assert_allclose(a.mean_coefficients, b.mean_coefficients, atol=1e-13)
    result = a.evaluate(2, 8, 0.1)
    mass = gamma_stopping_probabilities(2, 8, 2)
    # Uniform legal fixation includes root repetitions; either child reveals the preferred branch.
    expected_left = mass[0] * 0.5 + mass[1] * (0.9 * (5 / 6) + 0.1 / 2)
    assert abs(result["probabilities"][0] - expected_left) < 5 * result["standard_errors"][0]
    assert result["probability_sum"] == pytest.approx(1, abs=1e-7)
    np.testing.assert_allclose(a.evaluate_repeats(8, 100, 1), 0.5, atol=1e-8)
    assert a.evaluate(8, 100, 1)["standard_errors"] == [0, 0]


def test_cli_cache_roundtrip_and_hash_check(tmp_path):
    _, defaults = load_canonical_defaults()
    run_args = {key: defaults[key] for key in ENV_STATIC_PARAM_KEYS}
    run_args.update(num_nodes=3, t_max=3, wm_decay=0.8, wm_neighbor_activation=0.4, cost=0,
                    lesion_policy="random_search_gamma_stopping")
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"args": run_args, "compat_version": COMPAT_VERSION}))
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"graph": [[1, 2], [], []], "start": 0, "rewards": [0, 8, -8], "moves": [1],
                                "looks": [999], "look_times": "ignored"}]))
    output = tmp_path / "cache.json"
    subprocess.run([sys.executable, "undirected_likelihood_cache.py", str(run), "--data", str(data),
                    "--output", str(output), "--samples", "8", "--repeats", "2", "--batch-size", "4",
                    "--wm-decay", "0.7", "0.9"], cwd=Path(__file__).resolve().parents[1],
                   check=True, capture_output=True, text=True)
    manifest, caches = load_undirected_cache_bundle(output)
    assert len(caches) == 2
    assert manifest["trials"][0]["observed_index"] == 0
    assert [run["wm_decay"] for run in manifest["runs"]] == [0.7, 0.9]
    assert caches[0][0].evaluate(2, 8, 0.1)["probability_sum"] == pytest.approx(1, abs=1e-7)
    candidates, trials = load_undirected_candidates(
        output, [{"gamma_shape": 2, "gamma_scale": 8}], repeat_indices=(0,),
    )
    assert len(candidates) == 2
    assert candidates[0].candidate_id != candidates[1].candidate_id
    assert trials == manifest["trials"]
    np.testing.assert_allclose(candidates[0].probabilities(0.2)[0], caches[0][0].evaluate_repeats(2, 8, 0.2)[0])
    with output.with_suffix(".npz").open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_undirected_cache_bundle(output)
