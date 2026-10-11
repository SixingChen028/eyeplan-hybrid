import itertools
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.config import ENV_STATIC_PARAM_KEYS, load_canonical_defaults
from modules.a2c import save_jax_params
from modules.compat import COMPAT_VERSION
from modules.environment import DecisionTreeEnv
from modules.network import actor_critic_forward, flatten_observation, init_mlp_actor_critic_params, sample_actions
from modules.path_likelihood import PathLikelihoodEstimator, TreeProblem, conditional_path_probabilities


def make_env(n=5, t_max=4):
    _, defaults = load_canonical_defaults()
    defaults.update(num_nodes=n, t_max=t_max)
    return DecisionTreeEnv(**{key: defaults[key] for key in ENV_STATIC_PARAM_KEYS})


def make_problem():
    # Nonzero root label, unequal path lengths, positive and negative rewards.
    return TreeProblem.from_record({
        "graph": [[], [0, 4], [1, 3], [], []], "start": 2, "rewards": [-8, 4, 0, 2, -4],
    })


def enumerate_execution(problem, activation, q, decay, neighbor):
    """Independent reference: enumerate whole retention masks and movement ties."""
    result = np.zeros(len(problem.paths))

    def walk(node, activation, q, path, mass):
        decayed = np.clip(activation * decay, 0, 1)
        for keep in itertools.product((False, True), repeat=len(activation)):
            keep = np.asarray(keep)
            probability = np.prod(np.where(keep, decayed, 1 - decayed))
            if probability == 0:
                continue
            refreshed = np.where(keep, decayed, 0)
            refreshed[node] = 1
            neighbors = [problem.parents[node], *problem.children[node]]
            for other in neighbors:
                if other >= 0:
                    refreshed[other] = max(refreshed[other], neighbor)
            cleared = np.where(refreshed > 0, q, 0)
            children = problem.children[node]
            if children[0] < 0:
                result[problem.paths.index(tuple(path))] += mass * probability
            else:
                winners = children[cleared[children] == max(cleared[children])]
                for child in winners:
                    walk(int(child), refreshed, cleared, [*path, int(child)], mass * probability / len(winners))

    walk(problem.root, np.asarray(activation, dtype=float), np.asarray(q, dtype=float), [], 1.0)
    return result


@pytest.mark.parametrize("decay, neighbor", [(0.0, 0.4), (0.7, 0.3), (1.0, 0.4), (1.0, 1.0)])
def test_exact_execution_matches_retention_enumeration(decay, neighbor):
    env = make_env()
    problem = make_problem()
    params = env.make_params(wm_decay=decay, wm_neighbor_activation=neighbor, cost=0)
    estimator = PathLikelihoodEstimator(env)
    state = estimator.initial_state(
        problem.root, jnp.asarray(problem.points), jnp.asarray(problem.children), jnp.asarray(problem.parents),
        jax.random.PRNGKey(0), params,
    )
    for q in ([8, -4, 0, -2, 4], [-8, 4, 0, -2, -4], [0, 0, 0, 0, 0]):
        activation = jnp.asarray([0.5, 0.9, 1, 0.6, 0.75])
        state = state._replace(activation=activation, q_values=jnp.asarray(q, dtype=jnp.float32))
        actual = conditional_path_probabilities(state, params, problem.depths, problem.padded_paths())
        expected = enumerate_execution(problem, activation, q, decay, neighbor)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_allclose(np.sum(actual), 1, atol=1e-7)


def test_fixed_problem_reset_matches_environment():
    env = make_env()
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    before = env._sample_initial_state(jax.random.PRNGKey(42))
    expected = env._look(before, before.root_node, params)
    actual = PathLikelihoodEstimator(env).initial_state(
        before.root_node, before.points, before.child_nodes, before.parent_nodes, before.rng_key, params,
    )
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a, b)


def constant_policy(env, stop):
    params = init_mlp_actor_critic_params(
        jax.random.PRNGKey(0), feature_size=len(flatten_observation(env.observation_template)),
        action_size=env.action_size, hidden_size=4,
    )
    params = jax.tree_util.tree_map(jnp.zeros_like, params)
    params["policy"]["b"] = params["policy"]["b"].at[-1].set(1000 if stop else -1000)
    return params


@pytest.mark.parametrize("stop", [False, True])
def test_stopping_sampler_respects_initial_and_forced_stop(stop):
    env = make_env()
    problem = make_problem()
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    estimator = PathLikelihoodEstimator(env)
    states = estimator._sample_batch(
        constant_policy(env, stop), params, problem.root, problem.points, problem.children, problem.parents,
        jax.random.split(jax.random.PRNGKey(3), 32),
    )
    np.testing.assert_array_equal(states.time_elapsed, 0 if stop else env.t_max - 1)
    if stop:
        # Root reward is zero and no descendant has been visited, so choices are uniform.
        probabilities = jax.vmap(conditional_path_probabilities, in_axes=(0, None, None, None))(
            states, params, problem.depths, problem.padded_paths(),
        )
        np.testing.assert_allclose(probabilities, np.broadcast_to([0.25, 0.25, 0.5], probabilities.shape))


def test_mc_is_batch_invariant_and_includes_horizon_mass():
    env = make_env(t_max=3)
    problem = make_problem()
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    estimator = PathLikelihoodEstimator(env, undirected=True)
    a = estimator.estimate(problem, {}, params, samples=64, batch_size=16, seed=123)
    b = estimator.estimate(problem, {}, params, samples=64, batch_size=23, seed=123)
    np.testing.assert_allclose(a["probabilities"], b["probabilities"], atol=1e-12)
    np.testing.assert_allclose(a["standard_errors"], b["standard_errors"], atol=1e-12)
    assert a["horizon_fraction"] > 0
    assert a["probability_sum"] == pytest.approx(1, abs=1e-6)


def test_mc_recovers_exact_tiny_planning_likelihood():
    # At most one voluntary fixation; uniform policy over root, both children, stop.
    env = make_env(n=3, t_max=2)
    problem = TreeProblem.from_record({"graph": [[1, 2], [], []], "start": 0, "rewards": [0, 8, -8]})
    params = env.make_params(wm_decay=0.7, wm_neighbor_activation=0.3, cost=0)
    policy = constant_policy(env, True)
    policy["policy"]["b"] = jnp.zeros(env.action_size)
    result = PathLikelihoodEstimator(env).estimate(problem, policy, params, samples=8192, batch_size=512, seed=7)
    # Root/stop -> tie; looking at either child -> left. Each case has mass 1/4.
    expected = np.array([0.75, 0.25])
    error = np.abs(np.asarray(result["probabilities"]) - expected)
    assert np.all(error < 5 * np.asarray(result["standard_errors"]))


def test_stopping_states_match_original_environment_steps():
    env = make_env()
    problem = make_problem()
    params = env.make_params(wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    policy = constant_policy(env, True)
    policy["policy"]["b"] = jnp.zeros(env.action_size)
    estimator = PathLikelihoodEstimator(env)

    def original(key):
        policy_key, memory_key, _ = jax.random.split(key, 3)
        state = estimator.initial_state(
            problem.root, jnp.asarray(problem.points), jnp.asarray(problem.children), jnp.asarray(problem.parents),
            memory_key, params,
        )

        def step(carry):
            state, policy_key, _, _ = carry
            policy_key, action_key = jax.random.split(policy_key)
            mask = env._get_action_mask(state)
            obs = jax.tree_util.tree_map(
                lambda x: None if x is None else x[None, ...], env._get_obs(state),
            )
            logits, _ = actor_critic_forward(policy, obs, mask[None, :], env._get_observation_mask(state)[None, :])
            actions, _, _ = sample_actions(action_key, logits, mask[None, :])
            before_step = state
            state, _, _, done, _ = env.step(state, actions[0], params)
            return state, policy_key, done, before_step

        return jax.lax.while_loop(
            lambda carry: ~carry[2], step, (state, policy_key, jnp.bool_(False), state),
        )[3]

    keys = jax.random.split(jax.random.PRNGKey(12), 64)
    actual = estimator._sample_batch(
        policy, params, problem.root, problem.points, problem.children, problem.parents, keys,
    )
    expected = jax.jit(jax.vmap(original))(keys)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a, b)


def test_problem_rejects_invalid_graph_and_reward_units():
    with pytest.raises(ValueError, match="unique parents"):
        TreeProblem.from_record({"graph": [[1, 1], [], []], "start": 0, "rewards": [0, 1, 1]})
    env = make_env()
    problem = TreeProblem.from_record({"graph": [[], [0, 4], [1, 3], [], []], "start": 2,
                                       "rewards": [-80, 40, 0, 20, -40]})
    params = env.make_params(wm_decay=1, wm_neighbor_activation=1, cost=0)
    with pytest.raises(ValueError, match="raw point units"):
        PathLikelihoodEstimator(env).estimate(problem, constant_policy(env, True), params, samples=2)


def test_exact_execution_matches_unchanged_deep_tree_simulator():
    env = make_env(n=9)
    problem = TreeProblem.from_record({
        "graph": [[1, 2], [3, 4], [], [5, 6], [], [7, 8], [], [], []],
        "start": 0, "rewards": [0, 8, -4, 4, -8, 2, -2, 1, -1],
    })
    params = env.make_params(wm_decay=0.85, wm_neighbor_activation=0.4, cost=0)
    state = PathLikelihoodEstimator(env).initial_state(
        problem.root, jnp.asarray(problem.points), jnp.asarray(problem.children), jnp.asarray(problem.parents),
        jax.random.PRNGKey(0), params,
    )._replace(
        activation=jnp.asarray([1, 0.9, 0.7, 0.8, 0.5, 0.9, 0.7, 0.6, 0.8]),
        q_values=jnp.asarray([0, 8, -4, 4, -8, -2, -4, 8, 2], dtype=jnp.float32),
    )
    expected = np.asarray(conditional_path_probabilities(state, params, problem.depths, problem.padded_paths()))
    n = 32768

    def simulate(key):
        _, path, _ = env._sample_move_path(state._replace(rng_key=key), params)
        return path[jnp.sum(path >= 0) - 1]

    leaves = np.asarray(jax.jit(jax.vmap(simulate))(jax.random.split(jax.random.PRNGKey(15), n)))
    actual = np.array([(leaves == path[-1]).mean() for path in problem.paths])
    assert np.all(np.abs(actual - expected) < 6 * np.sqrt(expected * (1 - expected) / n) + 1 / n)


def test_cli_scores_moves_and_ignores_looks(tmp_path):
    env = make_env(n=3)
    _, defaults = load_canonical_defaults()
    run_args = {key: defaults[key] for key in ENV_STATIC_PARAM_KEYS}
    run_args.update(num_nodes=3, t_max=4, wm_decay=0.8, wm_neighbor_activation=0.4, cost=0)
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"args": run_args, "compat_version": COMPAT_VERSION}))
    save_jax_params(constant_policy(env, True), str(run / "net_jax.p"))
    record = {"graph": [[1, 2], [], []], "start": 0, "rewards": [0, 8, -8], "moves": [1]}
    data = tmp_path / "data.json"
    # Deliberately nonsensical look fields must not enter evaluation.
    data.write_text(json.dumps([record, {**record, "looks": [999], "look_times": "ignored"}]))
    output = tmp_path / "output.json"
    subprocess.run([
        sys.executable, "path_likelihood.py", str(run), "--data", str(data), "--output", str(output),
        "--samples", "8", "--batch-size", "4", "--repeats", "1",
    ], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    for trial in result["runs"][0]["trials"]:
        estimate = trial["estimates"][0]
        assert estimate["observed_probability"] == 0.5
        assert estimate["observed_standard_error"] == 0
