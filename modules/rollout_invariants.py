from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from modules.environment import DecisionTreeObs, JaxDecisionTreeEnv, JaxDecisionTreeParams, JaxDecisionTreeState


class FixationRolloutTrace(NamedTuple):
    states: JaxDecisionTreeState
    observations: DecisionTreeObs
    action_masks: jax.Array
    observation_masks: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array


def collect_random_fixation_rollouts(
    env: JaxDecisionTreeEnv,
    params: JaxDecisionTreeParams,
    *,
    seed: int,
    num_rollouts: int,
    num_steps: int,
) -> FixationRolloutTrace:
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if num_steps > env.t_max - 1:
        raise ValueError("num_steps must be at most env.t_max - 1 for legal fixation-only rollouts.")

    def append_initial(initial, scanned):
        if initial is None:
            return None
        return jnp.concatenate([initial[None, ...], scanned], axis=0)

    def run_one(key):
        state, obs, info = env.reset(key, params)
        action_mask = info["mask"]
        observation_mask = info["observation_mask"]

        def body_fn(carry, _):
            state, obs, action_mask, observation_mask, key = carry
            key, action_key = jax.random.split(key)
            fixation_mask = action_mask[: env.num_nodes]
            probs = fixation_mask.astype(jnp.float32) / jnp.sum(fixation_mask)
            action = jax.random.choice(action_key, env.num_nodes, p=probs)

            next_state, next_obs, reward, done, next_info = env.step(state, action, params)
            next_action_mask = next_info["mask"]
            next_observation_mask = next_info["observation_mask"]

            next_carry = (
                next_state,
                next_obs,
                next_action_mask,
                next_observation_mask,
                key,
            )
            output = (
                action,
                reward,
                done,
                next_state,
                next_obs,
                next_action_mask,
                next_observation_mask,
            )
            return next_carry, output

        _, outputs = jax.lax.scan(body_fn, (state, obs, action_mask, observation_mask, key), None, length=num_steps)
        actions, rewards, dones, step_states, step_obs, step_action_masks, step_observation_masks = outputs

        return FixationRolloutTrace(
            states=jax.tree_util.tree_map(append_initial, state, step_states),
            observations=jax.tree_util.tree_map(append_initial, obs, step_obs),
            action_masks=jnp.concatenate([action_mask[None, ...], step_action_masks], axis=0),
            observation_masks=jnp.concatenate([observation_mask[None, ...], step_observation_masks], axis=0),
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

    keys = jax.random.split(jax.random.PRNGKey(seed), num_rollouts)
    return jax.jit(jax.vmap(run_one))(keys)


def assert_fixation_rollout_invariants(
    env: JaxDecisionTreeEnv,
    params: JaxDecisionTreeParams,
    trace: FixationRolloutTrace,
    *,
    expect_max_consistent_q: bool = False,
    atol: float = 1e-5,
) -> None:
    trace = jax.device_get(trace)
    num_rollouts = int(np.asarray(trace.actions).shape[0])
    num_steps = int(np.asarray(trace.actions).shape[1])

    for rollout_idx in range(num_rollouts):
        _assert_tree_invariants(env, trace.states, rollout_idx, atol)
        for step_idx in range(num_steps + 1):
            _assert_state_obs_info_invariants(env, trace, rollout_idx, step_idx, atol)
            if expect_max_consistent_q:
                _assert_max_consistent_q(trace.states, rollout_idx, step_idx, atol)

        for step_idx in range(num_steps):
            _assert_action_invariants(env, params, trace, rollout_idx, step_idx, atol)


def _state_at(states: JaxDecisionTreeState, rollout_idx: int, step_idx: int) -> JaxDecisionTreeState:
    return jax.tree_util.tree_map(lambda value: np.asarray(value[rollout_idx, step_idx]), states)


def _obs_at(observations: DecisionTreeObs, rollout_idx: int, step_idx: int) -> DecisionTreeObs:
    return jax.tree_util.tree_map(
        lambda value: None if value is None else np.asarray(value[rollout_idx, step_idx]),
        observations,
    )


def _context(rollout_idx: int, step_idx: int, message: str) -> str:
    return f"rollout={rollout_idx}, step={step_idx}: {message}"


def _assert_tree_invariants(
    env: JaxDecisionTreeEnv,
    states: JaxDecisionTreeState,
    rollout_idx: int,
    atol: float,
) -> None:
    state = _state_at(states, rollout_idx, 0)
    root = int(state.root_node)
    points = np.asarray(state.points)
    child_nodes = np.asarray(state.child_nodes)
    parent_nodes = np.asarray(state.parent_nodes)

    assert 0 <= root < env.num_nodes, f"rollout={rollout_idx}: root_node is invalid."
    assert parent_nodes[root] == -1, f"rollout={rollout_idx}: root has a parent."
    np.testing.assert_allclose(points[root], 0.0, atol=atol)

    parent_counts = np.zeros(env.num_nodes, dtype=np.int32)
    for node, children in enumerate(child_nodes):
        both_missing = np.all(children < 0)
        both_present = np.all((0 <= children) & (children < env.num_nodes))
        assert both_missing or both_present, f"rollout={rollout_idx}: node {node} has malformed children."

        for child in children:
            if child < 0:
                continue
            assert int(parent_nodes[child]) == node, f"rollout={rollout_idx}: child/parent mismatch."
            parent_counts[child] += 1

    assert parent_counts[root] == 0, f"rollout={rollout_idx}: root has counted parents."
    assert np.all(parent_counts[np.arange(env.num_nodes) != root] == 1), (
        f"rollout={rollout_idx}: non-root parent counts are invalid."
    )

    expected_g = _path_values(child_nodes, points, root)
    for step_idx in range(np.asarray(states.g_values).shape[1]):
        g_values = np.asarray(states.g_values[rollout_idx, step_idx])
        is_discovered = np.asarray(states.is_discovered[rollout_idx, step_idx])
        np.testing.assert_allclose(g_values[is_discovered], expected_g[is_discovered], atol=atol)


def _path_values(child_nodes: np.ndarray, points: np.ndarray, root: int) -> np.ndarray:
    g_values = np.zeros(points.shape, dtype=np.float32)
    stack = [int(root)]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            raise AssertionError("tree contains a cycle.")
        seen.add(node)

        for child in child_nodes[node]:
            if child < 0:
                continue
            g_values[child] = g_values[node] + points[node]
            stack.append(int(child))

    if len(seen) != points.shape[0]:
        raise AssertionError("tree is not fully reachable from root.")
    return g_values


def _assert_state_obs_info_invariants(
    env: JaxDecisionTreeEnv,
    trace: FixationRolloutTrace,
    rollout_idx: int,
    step_idx: int,
    atol: float,
) -> None:
    state = _state_at(trace.states, rollout_idx, step_idx)
    obs = _obs_at(trace.observations, rollout_idx, step_idx)
    action_mask = np.asarray(trace.action_masks[rollout_idx, step_idx])
    observation_mask = np.asarray(trace.observation_masks[rollout_idx, step_idx])

    root = int(state.root_node)
    fixation = int(state.fixation_node)
    activation = np.asarray(state.activation)
    is_discovered = np.asarray(state.is_discovered)
    child_nodes = np.asarray(state.child_nodes)
    parent_nodes = np.asarray(state.parent_nodes)
    q_values = np.asarray(state.q_values)
    n_visits = np.asarray(state.n_visits)
    fixation_recency = np.asarray(state.fixation_recency)
    is_terminal = np.asarray(state.is_terminal)
    time_elapsed = int(state.time_elapsed)

    assert time_elapsed == step_idx, _context(rollout_idx, step_idx, "time_elapsed does not match trace step.")
    assert 0 <= fixation < env.num_nodes, _context(rollout_idx, step_idx, "fixation_node is invalid.")
    assert bool(is_discovered[fixation]), _context(rollout_idx, step_idx, "fixation node is undiscovered.")
    np.testing.assert_allclose(activation[fixation], 1.0, atol=atol)
    assert np.all((0.0 <= activation) & (activation <= 1.0 + atol)), (
        _context(rollout_idx, step_idx, "activation is outside [0, 1].")
    )
    assert np.all((activation <= 0.0) | is_discovered), (
        _context(rollout_idx, step_idx, "active nodes must be discovered.")
    )

    leaf_nodes = child_nodes[:, 0] < 0
    assert np.all(is_terminal <= leaf_nodes), _context(rollout_idx, step_idx, "non-leaf terminal marker.")
    assert np.all(n_visits >= 0), _context(rollout_idx, step_idx, "negative visit count.")
    assert np.all(np.isfinite(q_values)), _context(rollout_idx, step_idx, "non-finite q value.")
    assert np.all((0.0 <= fixation_recency) & (fixation_recency <= 1.0 + atol)), (
        _context(rollout_idx, step_idx, "fixation recency is outside [0, 1].")
    )
    np.testing.assert_allclose(fixation_recency[fixation], 1.0, atol=atol)
    assert np.all(n_visits[fixation] >= 1), _context(rollout_idx, step_idx, "unvisited fixation node.")
    assert np.all(n_visits <= step_idx + 1), _context(rollout_idx, step_idx, "visit count exceeds look count.")
    assert np.all(n_visits[~is_discovered] == 0), _context(rollout_idx, step_idx, "undiscovered node was visited.")
    np.testing.assert_allclose(q_values[~is_discovered], 0.0, atol=atol)
    np.testing.assert_allclose(fixation_recency[~is_discovered], 0.0, atol=atol)
    assert np.all(~is_terminal[~is_discovered]), _context(rollout_idx, step_idx, "undiscovered terminal marker.")

    expected_observation_mask = activation > 0.0 if env.activation_masks_observation else is_discovered
    np.testing.assert_array_equal(observation_mask, expected_observation_mask)

    fixation_allowed = time_elapsed != env.t_max - 1
    expected_action_mask = np.zeros(env.action_size, dtype=bool)
    if env.activation_masks_actions:
        expected_action_mask[: env.num_nodes] = (activation > 0.0) & fixation_allowed
    else:
        expected_action_mask[: env.num_nodes] = fixation_allowed
    expected_action_mask[root] = fixation_allowed
    expected_action_mask[-1] = True
    np.testing.assert_array_equal(action_mask, expected_action_mask)

    _assert_observation_matches_state(env, state, obs, observation_mask, atol, rollout_idx, step_idx)


def _assert_observation_matches_state(
    env: JaxDecisionTreeEnv,
    state: JaxDecisionTreeState,
    obs: DecisionTreeObs,
    observation_mask: np.ndarray,
    atol: float,
    rollout_idx: int,
    step_idx: int,
) -> None:
    fixation = int(state.fixation_node)
    parent = int(state.parent_nodes[fixation])
    children = np.asarray(state.child_nodes[fixation])
    expected_fixation = np.eye(env.num_nodes, dtype=np.float32)[fixation]
    expected_root = np.eye(env.num_nodes, dtype=np.float32)[int(state.root_node)]
    expected_parent = np.zeros(env.num_nodes, dtype=np.float32)
    expected_child = np.zeros(env.num_nodes, dtype=np.float32)

    if parent >= 0:
        expected_parent[parent] = 1.0
    for child in children:
        if child >= 0:
            expected_child[child] = 1.0

    np.testing.assert_array_equal(obs.fixation, expected_fixation)
    np.testing.assert_allclose(obs.fixation_point, np.array([state.points[fixation]], dtype=np.float32), atol=atol)
    np.testing.assert_array_equal(obs.parent, expected_parent)
    np.testing.assert_array_equal(obs.child, expected_child)
    np.testing.assert_array_equal(obs.root, expected_root)

    if obs.g_values is not None:
        np.testing.assert_allclose(obs.g_values, np.where(observation_mask, state.g_values, 0.0), atol=atol)
    if obs.q_values is not None:
        np.testing.assert_allclose(obs.q_values, np.where(observation_mask, state.q_values, 0.0), atol=atol)
    if obs.n_visits is not None:
        np.testing.assert_allclose(obs.n_visits, np.where(observation_mask, state.n_visits, 0), atol=atol)
    if obs.is_terminal is not None:
        np.testing.assert_allclose(obs.is_terminal, state.is_terminal & observation_mask, atol=atol)
    if obs.recency is not None:
        np.testing.assert_allclose(obs.recency, np.where(observation_mask, state.fixation_recency, 0.0), atol=atol)
    if obs.time_elapsed is not None:
        np.testing.assert_allclose(obs.time_elapsed, np.array([state.time_elapsed], dtype=np.float32), atol=atol)


def _assert_action_invariants(
    env: JaxDecisionTreeEnv,
    params: JaxDecisionTreeParams,
    trace: FixationRolloutTrace,
    rollout_idx: int,
    step_idx: int,
    atol: float,
) -> None:
    action = int(np.asarray(trace.actions[rollout_idx, step_idx]))
    action_mask = np.asarray(trace.action_masks[rollout_idx, step_idx])
    reward = float(np.asarray(trace.rewards[rollout_idx, step_idx]))
    done = bool(np.asarray(trace.dones[rollout_idx, step_idx]))

    assert 0 <= action < env.num_nodes, _context(rollout_idx, step_idx, "sampled non-fixation action.")
    assert bool(action_mask[action]), _context(rollout_idx, step_idx, "sampled illegal fixation action.")
    np.testing.assert_allclose(reward, -float(params.cost), atol=atol)
    assert not done, _context(rollout_idx, step_idx, "fixation-only rollout ended early.")


def _assert_max_consistent_q(
    states: JaxDecisionTreeState,
    rollout_idx: int,
    step_idx: int,
    atol: float,
) -> None:
    state = _state_at(states, rollout_idx, step_idx)
    points = np.asarray(state.points)
    child_nodes = np.asarray(state.child_nodes)
    q_values = np.asarray(state.q_values)
    visited = np.asarray(state.n_visits) > 0

    np.testing.assert_array_equal(np.asarray(state.activation) > 0.0, np.asarray(state.is_discovered))

    for node in np.flatnonzero(visited):
        children = child_nodes[node]
        if children[0] < 0:
            expected = points[node]
        else:
            expected = points[node] + np.max(q_values[children])
        np.testing.assert_allclose(
            q_values[node],
            expected,
            atol=atol,
            err_msg=_context(rollout_idx, step_idx, f"q value for node {node} is not max-consistent."),
        )
