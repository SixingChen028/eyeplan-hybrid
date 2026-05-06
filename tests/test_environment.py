import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import JaxDecisionTreeEnv, make_decision_tree_params

_, _DEFAULT_PARAMS = load_canonical_defaults()


def _env(**overrides):
    params = dict(_DEFAULT_PARAMS)
    params.update(overrides)
    return JaxDecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        shuffle_nodes=bool(params["shuffle_nodes"]),
        use_recency_obs=bool(params.get("use_recency_obs", False)),
        point_set=params.get("point_set"),
    )


def _env_params(env, **overrides):
    params = {key: _DEFAULT_PARAMS[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    params.update(overrides)
    return make_decision_tree_params(env, **params)


def _bfs_visit_order(child_nodes: np.ndarray, root: int) -> list[int]:
    order: list[int] = []
    queue: list[int] = [int(root)]
    seen: set[int] = set()

    while queue:
        node = queue.pop(0)
        if node in seen:
            continue

        seen.add(node)
        order.append(node)

        left = int(child_nodes[node, 0])
        right = int(child_nodes[node, 1])
        if left >= 0:
            queue.append(left)
        if right >= 0:
            queue.append(right)

    return order


def _optimal_path_reward_raw(child_nodes: np.ndarray, points: np.ndarray, root: int) -> float:
    def dfs(node: int) -> float:
        left = int(child_nodes[node, 0])
        if left < 0:
            return float(points[node])

        right = int(child_nodes[node, 1])
        return float(points[node]) + max(dfs(left), dfs(right))

    return dfs(int(root))


def _first_child_path(child_nodes: np.ndarray, root: int) -> list[int]:
    path: list[int] = []
    node = int(root)

    while True:
        child = int(child_nodes[node, 0])
        if child < 0:
            return path

        path.append(child)
        node = child


def _jax_action(action: int):
    return jnp.asarray(action, dtype=jnp.int32)




def test_reset_uses_raw_node_ids_by_default():
    env = _env(num_nodes=15, shuffle_nodes=True)

    state, obs, info = env.reset_with_params(jax.random.PRNGKey(23), _env_params(env))
    root = int(state.root_node)

    fixation_slice = slice(0, env.num_nodes)
    root_slice = slice(1 + env.num_nodes * 3, 1 + env.num_nodes * 4)
    np.testing.assert_array_equal(np.asarray(obs[fixation_slice]), np.eye(env.num_nodes)[root])
    np.testing.assert_array_equal(np.asarray(obs[root_slice]), np.eye(env.num_nodes)[root])

    expected_mask = np.zeros(env.action_size, dtype=bool)
    expected_mask[: env.num_nodes] = np.asarray(state.activation) > 0
    expected_mask[root] = True
    expected_mask[-1] = True
    np.testing.assert_array_equal(np.asarray(info["mask"]), expected_mask)


def test_recency_observation_tracks_direct_fixations():
    num_nodes = 7
    jax_env = _env(
        num_nodes=num_nodes,
        t_max=20,
        shuffle_nodes=False,
        use_recency_obs=True,
    )
    jax_params = _env_params(jax_env, wm_decay=0.5, recency_decay=0.5)
    default_env = _env(num_nodes=num_nodes, use_recency_obs=False)

    assert jax_env.observation_shape[0] == default_env.observation_shape[0] + num_nodes

    state, obs_jax, _ = jax_env.reset_with_params(jax.random.PRNGKey(8), jax_params)

    recency_slice = slice(-num_nodes - 1, -1)
    reset_recency = np.zeros(num_nodes)
    reset_recency[int(state.root_node)] = 1.0
    np.testing.assert_allclose(np.asarray(obs_jax)[recency_slice], reset_recency, atol=1e-6)

    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, obs_jax, _, _, _, _ = jax_env.step_with_params(state, _jax_action(action), jax_params)

    expected_recency = reset_recency * 0.5
    expected_recency[action] = 1.0
    np.testing.assert_allclose(np.asarray(obs_jax)[recency_slice], expected_recency, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.fixation_recency), expected_recency, atol=1e-6)


def test_zero_recency_decay_keeps_only_current_fixation():
    env = _env(
        num_nodes=7,
        t_max=20,
        shuffle_nodes=False,
        use_recency_obs=True,
    )
    params = _env_params(env, wm_decay=0.5, recency_decay=0.0)
    state, obs, _ = env.reset_with_params(jax.random.PRNGKey(9), params)
    recency_slice = slice(-env.num_nodes - 1, -1)

    expected_reset = np.zeros(env.num_nodes)
    expected_reset[int(state.root_node)] = 1.0
    np.testing.assert_allclose(np.asarray(obs)[recency_slice], expected_reset, atol=1e-6)

    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, obs, _, _, _, _ = env.step_with_params(state, _jax_action(action), params)

    expected_step = np.zeros(env.num_nodes)
    expected_step[action] = 1.0
    np.testing.assert_allclose(np.asarray(obs)[recency_slice], expected_step, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.fixation_recency), expected_step, atol=1e-6)


def test_recency_decay_one_means_no_decay():
    env = _env(
        num_nodes=7,
        t_max=20,
        shuffle_nodes=False,
        use_recency_obs=True,
    )
    params = _env_params(env, wm_decay=1.0, recency_decay=1.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(10), params)
    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, obs, _, _, _, _ = env.step_with_params(state, _jax_action(action), params)

    recency = np.asarray(obs)[-env.num_nodes - 1 : -1]
    expected = np.zeros(env.num_nodes)
    expected[int(state.root_node)] = 1.0
    expected[action] = 1.0
    np.testing.assert_allclose(recency, expected, atol=1e-6)


def test_q_drop_rate_resets_inactive_q_values():
    env = _env(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = _env_params(env, wm_decay=0.0, q_drop_rate=1.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(12), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    activation = jnp.zeros((env.num_nodes,), dtype=jnp.float32)
    state = state._replace(q_values=q_values, activation=activation)

    state = env._update_activation(state, state.root_node, params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(np.asarray(state.q_values)[inactive_mask], 0.0, atol=1e-6)


def test_q_drop_rate_zero_preserves_inactive_q_values():
    env = _env(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = _env_params(env, wm_decay=0.0, q_drop_rate=0.0, q_decay=1.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(13), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    activation = jnp.zeros((env.num_nodes,), dtype=jnp.float32)
    state = state._replace(q_values=q_values, activation=activation)

    state = env._update_activation(state, state.root_node, params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(np.asarray(state.q_values)[inactive_mask], np.asarray(q_values)[inactive_mask], atol=1e-6)


def test_q_decay_scales_inactive_q_values():
    env = _env(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = _env_params(env, wm_decay=0.0, q_decay=0.75)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(14), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    activation = jnp.zeros((env.num_nodes,), dtype=jnp.float32)
    state = state._replace(q_values=q_values, activation=activation)

    state = env._update_activation(state, state.root_node, params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(
        np.asarray(state.q_values)[inactive_mask],
        np.asarray(q_values * 0.75)[inactive_mask],
        atol=1e-6,
    )


def test_q_drift_adds_noise_to_inactive_q_values_only():
    env = _env(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = _env_params(env, wm_decay=0.0, q_drift=0.5)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(15), params)
    state = state._replace(
        q_values=jnp.zeros((env.num_nodes,), dtype=jnp.float32),
        activation=jnp.zeros((env.num_nodes,), dtype=jnp.float32),
    )

    state = env._update_activation(state, state.root_node, params)

    active_mask = np.asarray(state.activation) > 0.0
    inactive_mask = ~active_mask
    np.testing.assert_allclose(np.asarray(state.q_values)[active_mask], 0.0, atol=1e-6)
    assert np.any(np.abs(np.asarray(state.q_values)[inactive_mask]) > 1e-6)


def test_q_decay_one_means_no_decay():
    env = _env(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = _env_params(env, wm_decay=0.0, q_decay=1.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(16), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    activation = jnp.zeros((env.num_nodes,), dtype=jnp.float32)
    state = state._replace(q_values=q_values, activation=activation)

    state = env._update_activation(state, state.root_node, params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(
        np.asarray(state.q_values)[inactive_mask],
        np.asarray(q_values)[inactive_mask],
        atol=1e-6,
    )


def test_move_reward_marginalizes_over_possible_paths():
    env = _env(
        num_nodes=3,
        t_max=3,
        scale_factor=1.0,
        shuffle_nodes=False,
    )
    params = _env_params(env, beta_move=0.0, eps_move=0.0, cost=0.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(101), params)
    state = state._replace(
        points=jnp.array([0.0, 2.0, 6.0], dtype=jnp.float32),
        q_values=jnp.zeros((3,), dtype=jnp.float32),
    )

    state, _, reward, done, truncated, _ = env.step_with_params(state, _jax_action(env.num_nodes), params)

    assert bool(done)
    assert not bool(truncated)
    np.testing.assert_allclose(float(reward), 4.0, atol=1e-6)
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


def test_compiled_rollout_matches_eager_rollout():
    env = _env(
        num_nodes=7,
        t_max=20,
        scale_factor=1 / 8,
        shuffle_nodes=False,
        point_set=jnp.array([1.0], dtype=jnp.float32),
    )
    params = _env_params(env, beta_move=50.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=0.5,
        wm_decay=1.0,
        cost=0.01,
    )
    key = jax.random.PRNGKey(6)
    reset_state, _, _ = env.reset_with_params(key, params)
    path_actions = _first_child_path(np.asarray(reset_state.child_nodes), int(reset_state.root_node))
    actions = jnp.array(path_actions + [env.num_nodes], dtype=jnp.int32)

    def rollout(k, action_seq):
        state, reset_obs, reset_info = env.reset_with_params(k, params)

        def body_fn(carry, action):
            state = carry
            state, obs, reward, done, truncated, info = env.step_with_params(state, action, params)
            output = (obs, reward, done, truncated, info["mask"])
            return state, output

        state, (obses, rewards, dones, truncateds, masks) = jax.lax.scan(body_fn, state, action_seq)
        return state, reset_obs, reset_info["mask"], obses, rewards, dones, truncateds, masks

    eager = rollout(key, actions)
    compiled = jax.jit(rollout)(key, actions)

    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager[0]),
        jax.tree_util.tree_leaves(compiled[0]),
    ):
        np.testing.assert_allclose(np.asarray(compiled_leaf), np.asarray(eager_leaf), atol=1e-6)

    for eager_item, compiled_item in zip(eager[1:], compiled[1:]):
        np.testing.assert_allclose(np.asarray(compiled_item), np.asarray(eager_item), atol=1e-6)

def test_move_path_is_not_stored_in_environment_state():
    env = _env(num_nodes=7, t_max=3, shuffle_nodes=False)
    params = _env_params(env)
    key = jax.random.PRNGKey(7)
    state, _, _ = env.reset_with_params(key, params)
    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, _, _, _, _, _ = env.step_with_params(state, _jax_action(action), params)
    state, _, _, _, _, _ = env.step_with_params(state, _jax_action(env.num_nodes), params)

    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")

    state, _, _ = env.reset_with_params(key, params)
    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    done_jax = False
    for _ in range(env.t_max - 1):
        state, _, _, done_jax, _, _ = env.step_with_params(state, _jax_action(action), params)
    assert not bool(done_jax)
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


def test_timeout_masks_to_move_action():
    env = _env(num_nodes=7, t_max=3, shuffle_nodes=False)
    params = _env_params(env)
    state, _, info_jax = env.reset_with_params(jax.random.PRNGKey(11), params)

    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    for _ in range(env.t_max - 1):
        state, _, _, _, _, info_jax = env.step_with_params(state, _jax_action(action), params)

    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), np.eye(env.action_size, dtype=bool)[env.num_nodes])
    state, _, reward_jax, done_jax, _, _ = env.step_with_params(state, _jax_action(env.num_nodes), params)

    assert bool(done_jax)
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


def test_visit_all_once_then_terminate_is_optimal_jax():
    env = _env(
        num_nodes=7,
        t_max=8,
        shuffle_nodes=True,
    )
    params = _env_params(env, beta_move=100.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=1.0,
        wm_decay=1.0,
        cost=0.0,
    )

    state, _, info = env.reset_with_params(jax.random.PRNGKey(19), params)
    child_nodes = np.asarray(state.child_nodes)
    points = np.asarray(state.points)
    root = int(state.root_node)

    visit_order = _bfs_visit_order(child_nodes, root)
    assert len(visit_order) == env.num_nodes
    assert len(set(visit_order)) == env.num_nodes

    for raw_action in visit_order:
        action = raw_action
        assert bool(np.asarray(info["mask"])[action])
        state, _, _, done, truncated, info = env.step_with_params(state, _jax_action(action), params)
        assert not bool(done)
        assert not bool(truncated)

    state, _, reward, done, truncated, _ = env.step_with_params(state, _jax_action(env.num_nodes), params)
    assert bool(done)
    assert not bool(truncated)

    optimal_scaled = _optimal_path_reward_raw(child_nodes, points, root) * env.scale_factor
    np.testing.assert_allclose(float(reward), optimal_scaled, atol=1e-6)


def test_backup_steps_zero_disables_ancestor_backup():
    env = _env(num_nodes=3, shuffle_nodes=False)
    params = _env_params(env, learning_rate=1.0, lamda_backup=1.0, backup_steps=0)
    q_values = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    child_nodes = jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32)
    parent_nodes = jnp.array([-1, 0, 0], dtype=jnp.int32)
    points = jnp.array([0.0, 3.0, 1.0], dtype=jnp.float32)

    updated = env._update_q(
        q_values=q_values,
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        root_node=jnp.asarray(0, dtype=jnp.int32),
        points=points,
        node=jnp.asarray(1, dtype=jnp.int32),
        activation=jnp.ones((3,), dtype=jnp.float32),
        params=params,
    )

    np.testing.assert_allclose(np.asarray(updated), np.array([0.0, 3.0, 0.0], dtype=np.float32), atol=1e-6)


def test_backup_steps_limits_ancestor_depth():
    env = _env(num_nodes=5, shuffle_nodes=False)
    params = _env_params(env, learning_rate=1.0, lamda_backup=1.0, backup_steps=1)
    q_values = jnp.zeros((5,), dtype=jnp.float32)
    child_nodes = jnp.array([[-1, -1], [0, -1], [1, -1], [2, -1], [3, -1]], dtype=jnp.int32)
    parent_nodes = jnp.array([1, 2, 3, 4, -1], dtype=jnp.int32)
    points = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=jnp.float32)

    updated = env._update_q(
        q_values=q_values,
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        root_node=jnp.asarray(4, dtype=jnp.int32),
        points=points,
        node=jnp.asarray(0, dtype=jnp.int32),
        activation=jnp.ones((5,), dtype=jnp.float32),
        params=params,
    )

    np.testing.assert_allclose(np.asarray(updated), np.array([1.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_wm_backup_ignores_inactive_child_q_during_ancestor_backup():
    child_nodes = jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32)
    parent_nodes = jnp.array([-1, 0, 0], dtype=jnp.int32)
    points = jnp.array([0.0, 5.0, 0.0], dtype=jnp.float32)
    activation = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)

    env_no_wm_backup = _env(
        num_nodes=3,
        shuffle_nodes=False,
    )
    no_wm_params = _env_params(
        env_no_wm_backup, learning_rate=1.0,
        lamda_backup=1.0,
        backup_steps=1,
        wm_backup=False,
    )
    updated_no_wm_backup = env_no_wm_backup._update_q(
        q_values=jnp.array([0.0, 0.0, 10.0], dtype=jnp.float32),
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        root_node=jnp.asarray(0, dtype=jnp.int32),
        points=points,
        node=jnp.asarray(1, dtype=jnp.int32),
        activation=activation,
        params=no_wm_params,
    )
    np.testing.assert_allclose(np.asarray(updated_no_wm_backup), np.array([10.0, 5.0, 10.0], dtype=np.float32), atol=1e-6)

    env_wm_backup = _env(
        num_nodes=3,
        shuffle_nodes=False,
    )
    wm_params = _env_params(
        env_wm_backup, learning_rate=1.0,
        lamda_backup=1.0,
        backup_steps=1,
        wm_backup=True,
    )
    updated_wm_backup = env_wm_backup._update_q(
        q_values=jnp.array([0.0, 0.0, 10.0], dtype=jnp.float32),
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        root_node=jnp.asarray(0, dtype=jnp.int32),
        points=points,
        node=jnp.asarray(1, dtype=jnp.int32),
        activation=activation,
        params=wm_params,
    )
    np.testing.assert_allclose(np.asarray(updated_wm_backup), np.array([5.0, 5.0, 10.0], dtype=np.float32), atol=1e-6)
