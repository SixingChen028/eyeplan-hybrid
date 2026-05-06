import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.environment import JaxDecisionTreeEnv
from modules.reference_environment import ReferenceDecisionTreeEnv


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


def _reference_children_array(env: ReferenceDecisionTreeEnv) -> np.ndarray:
    return env.child_nodes


def _reference_parent_array(env: ReferenceDecisionTreeEnv) -> np.ndarray:
    return env.parent_nodes


def _sync_reference_from_state(env: ReferenceDecisionTreeEnv, state) -> None:
    env.time_elapsed = int(state.time_elapsed)
    env.fixation_node = int(state.fixation_node)
    env.root_node = int(state.root_node)

    env.child_nodes = np.asarray(state.child_nodes).copy()
    env.parent_nodes = np.asarray(state.parent_nodes).copy()
    env.points = np.asarray(state.points).copy()

    env.q_values = np.asarray(state.q_values).copy()
    env.g_values = np.asarray(state.g_values).copy()
    env.n_visits = np.asarray(state.n_visits).copy()
    env.fixation_recency = np.asarray(state.fixation_recency).copy()

    env.activation = np.asarray(state.activation).copy()


def _reset_synced_envs(seed: int = 0, t_max: int = 20):
    reference_env, jax_env, jax_params, key = _make_envs(seed=seed, t_max=t_max)

    reference_env.reset()
    state, obs_jax, info_jax = jax_env.reset_with_params(key, jax_params)
    _sync_reference_from_state(reference_env, state)

    return reference_env, jax_env, jax_params, key, state, obs_jax, info_jax


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




def _assert_state_matches_reference(env: ReferenceDecisionTreeEnv, state) -> None:
    np.testing.assert_equal(int(state.time_elapsed), env.time_elapsed)
    np.testing.assert_equal(int(state.fixation_node), env.fixation_node)
    np.testing.assert_equal(int(state.root_node), env.root_node)

    np.testing.assert_array_equal(np.asarray(state.child_nodes), _reference_children_array(env))
    np.testing.assert_array_equal(np.asarray(state.parent_nodes), _reference_parent_array(env))
    np.testing.assert_array_equal(np.asarray(state.points), env.points)

    np.testing.assert_allclose(np.asarray(state.q_values), env.q_values, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.g_values), env.g_values, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.n_visits), env.n_visits)
    np.testing.assert_allclose(np.asarray(state.fixation_recency), env.fixation_recency, atol=1e-6)

    np.testing.assert_allclose(np.asarray(state.activation), env.activation, atol=1e-6)


def _make_envs(seed: int = 0, t_max: int = 20):
    num_nodes = 7
    beta_move = 50.0
    eps_move = 0.0
    learning_rate = 1.0
    lamda_backup = 0.5
    wm_decay = 1.0
    cost = 0.01
    scale_factor = 1 / 8
    shuffle_nodes = False

    reference_env = ReferenceDecisionTreeEnv(
        num_nodes=num_nodes,
        beta_move=beta_move,
        eps_move=eps_move,
        learning_rate=learning_rate,
        lamda_backup=lamda_backup,
        wm_decay=wm_decay,
        t_max=t_max,
        cost=cost,
        scale_factor=scale_factor,
        shuffle_nodes=shuffle_nodes,
        seed=seed,
    )
    reference_env.point_set = np.array([1.0], dtype=np.float32)

    jax_env = JaxDecisionTreeEnv(
        num_nodes=num_nodes,
        t_max=t_max,
        scale_factor=scale_factor,
        shuffle_nodes=shuffle_nodes,
        point_set=jnp.array([1.0], dtype=jnp.float32),
    )
    jax_params = jax_env.params(
        beta_move=beta_move,
        eps_move=eps_move,
        learning_rate=learning_rate,
        lamda_backup=lamda_backup,
        wm_decay=wm_decay,
        cost=cost,
    )

    key = jax.random.PRNGKey(seed)
    return reference_env, jax_env, jax_params, key


def test_reset_matches_reference_environment():
    reference_env, _, _, _, state, obs_jax, info_jax = _reset_synced_envs(seed=3)

    obs_reference = reference_env.get_obs()
    info_reference = {"mask": reference_env.get_action_mask()}

    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])
    _assert_state_matches_reference(reference_env, state)


def test_reset_uses_raw_node_ids_by_default():
    env = JaxDecisionTreeEnv(num_nodes=15, shuffle_nodes=True)

    state, obs, info = env.reset_with_params(jax.random.PRNGKey(23), env.params())
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


def test_fixation_step_matches_reference_environment():
    reference_env, jax_env, jax_params, _, state, _, _ = _reset_synced_envs(seed=4)

    raw_action = _first_child_path(_reference_children_array(reference_env), reference_env.root_node)[0]
    action = raw_action
    obs_reference, reward_reference, done_reference, truncated_reference, info_reference = reference_env.step(action)
    state, obs_jax, reward_jax, done_jax, truncated_jax, info_jax = jax_env.step_with_params(
        state,
        _jax_action(action),
        jax_params,
    )

    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_allclose(float(reward_jax), reward_reference, atol=1e-6)
    np.testing.assert_equal(bool(done_jax), done_reference)
    np.testing.assert_equal(bool(truncated_jax), truncated_reference)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])

    _assert_state_matches_reference(reference_env, state)


def test_recency_observation_tracks_direct_fixations():
    num_nodes = 7
    reference_env = ReferenceDecisionTreeEnv(
        num_nodes=num_nodes,
        wm_decay=0.5,
        t_max=20,
        shuffle_nodes=False,
        recency_decay="auto",
        seed=8,
    )
    jax_env = JaxDecisionTreeEnv(
        num_nodes=num_nodes,
        t_max=20,
        shuffle_nodes=False,
        use_recency_obs=True,
    )
    jax_params = jax_env.params(wm_decay=0.5, recency_decay="auto")
    default_env = JaxDecisionTreeEnv(num_nodes=num_nodes, use_recency_obs=False)

    assert jax_env.observation_shape[0] == default_env.observation_shape[0] + num_nodes

    reference_env.reset()
    state, obs_jax, _ = jax_env.reset_with_params(jax.random.PRNGKey(8), jax_params)
    _sync_reference_from_state(reference_env, state)

    recency_slice = slice(-num_nodes - 1, -1)
    reset_recency = np.zeros(num_nodes)
    reset_recency[int(state.root_node)] = 1.0
    np.testing.assert_allclose(np.asarray(obs_jax)[recency_slice], reset_recency, atol=1e-6)

    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    obs_reference, _, _, _, _ = reference_env.step(action)
    state, obs_jax, _, _, _, _ = jax_env.step_with_params(state, _jax_action(action), jax_params)

    expected_recency = reset_recency * 0.5
    expected_recency[action] = 1.0
    np.testing.assert_allclose(np.asarray(obs_jax)[recency_slice], expected_recency, atol=1e-6)
    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.fixation_recency), reference_env.fixation_recency, atol=1e-6)


def test_zero_recency_decay_observation_stays_zero():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        t_max=20,
        shuffle_nodes=False,
        use_recency_obs=True,
    )
    params = env.params(wm_decay=0.5, recency_decay=0.0)
    state, obs, _ = env.reset_with_params(jax.random.PRNGKey(9), params)
    recency_slice = slice(-env.num_nodes - 1, -1)

    np.testing.assert_allclose(np.asarray(obs)[recency_slice], np.zeros(env.num_nodes), atol=1e-6)

    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, obs, _, _, _, _ = env.step_with_params(state, _jax_action(action), params)

    np.testing.assert_allclose(np.asarray(obs)[recency_slice], np.zeros(env.num_nodes), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.fixation_recency), np.zeros(env.num_nodes), atol=1e-6)


def test_auto_recency_decay_uses_half_when_wm_decay_is_one():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        t_max=20,
        shuffle_nodes=False,
        use_recency_obs=True,
    )
    params = env.params(wm_decay=1.0, recency_decay="auto")
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(10), params)
    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, obs, _, _, _, _ = env.step_with_params(state, _jax_action(action), params)

    recency = np.asarray(obs)[-env.num_nodes - 1 : -1]
    expected = np.zeros(env.num_nodes)
    expected[int(state.root_node)] = 0.5
    expected[action] = 1.0
    np.testing.assert_allclose(recency, expected, atol=1e-6)


def test_q_drop_rate_resets_inactive_q_values():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = env.params(wm_decay=0.0, q_drop_rate=1.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(12), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    activation = jnp.zeros((env.num_nodes,), dtype=jnp.float32)
    state = state._replace(q_values=q_values, activation=activation)

    state = env._update_activation(state, state.root_node, params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(np.asarray(state.q_values)[inactive_mask], 0.0, atol=1e-6)


def test_q_drop_rate_zero_preserves_inactive_q_values():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = env.params(wm_decay=0.0, q_drop_rate=0.0)
    state, _, _ = env.reset_with_params(jax.random.PRNGKey(13), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    activation = jnp.zeros((env.num_nodes,), dtype=jnp.float32)
    state = state._replace(q_values=q_values, activation=activation)

    state = env._update_activation(state, state.root_node, params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(np.asarray(state.q_values)[inactive_mask], np.asarray(q_values)[inactive_mask], atol=1e-6)


def test_q_decay_shrinks_inactive_q_values():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = env.params(wm_decay=0.0, q_decay=0.25)
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
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        shuffle_nodes=False,
    )
    params = env.params(wm_decay=0.0, q_drift=0.5)
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


def test_q_decay_auto_uses_reward_scale_prior_variance():
    env = JaxDecisionTreeEnv(
        scale_factor=0.25,
        point_set=jnp.array([-2.0, 2.0], dtype=jnp.float32),
    )

    expected = 0.5**2 / (0.5**2 + np.var(np.array([-2.0, 2.0], dtype=np.float32) * 0.25))
    np.testing.assert_allclose(np.asarray(env.params(q_drift=0.5, q_decay="auto").q_decay), expected, atol=1e-6)


def test_move_step_matches_reference_environment():
    reference_env, jax_env, jax_params, _, state, _, _ = _reset_synced_envs(seed=5)

    for raw_action in _first_child_path(_reference_children_array(reference_env), reference_env.root_node):
        action = raw_action
        reference_env.step(action)
        state, _, _, _, _, _ = jax_env.step_with_params(state, _jax_action(action), jax_params)

    move_action = reference_env.num_nodes
    obs_reference, reward_reference, done_reference, truncated_reference, info_reference = reference_env.step(move_action)
    state, obs_jax, reward_jax, done_jax, truncated_jax, info_jax = jax_env.step_with_params(
        state,
        _jax_action(move_action),
        jax_params,
    )

    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_allclose(float(reward_jax), reward_reference, atol=1e-6)
    np.testing.assert_equal(bool(done_jax), done_reference)
    np.testing.assert_equal(bool(truncated_jax), truncated_reference)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])


def test_move_reward_marginalizes_over_possible_paths():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=3,
        scale_factor=1.0,
        shuffle_nodes=False,
    )
    params = env.params(beta_move=0.0, eps_move=0.0, cost=0.0)
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


def test_compiled_rollout_matches_reference_environment():
    reference_env, jax_env, jax_params, key = _make_envs(seed=6)

    reset_state, _, _ = jax_env.reset_with_params(key, jax_params)
    _sync_reference_from_state(reference_env, reset_state)
    action_list = []
    for raw_action in _first_child_path(_reference_children_array(reference_env), reference_env.root_node):
        action = raw_action
        action_list.append(action)
        reference_env.step(action)
    action_list.append(reference_env.num_nodes)
    actions = jnp.array(action_list, dtype=jnp.int32)

    def rollout(k, action_seq):
        state, reset_obs, reset_info = jax_env.reset_with_params(k, jax_params)

        def body_fn(carry, action):
            state = carry
            state, obs, reward, done, truncated, info = jax_env.step_with_params(state, action, jax_params)
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

    reference_env.reset()
    _sync_reference_from_state(reference_env, reset_state)
    obs_reference_reset = reference_env.get_obs()
    info_reference_reset = {"mask": reference_env.get_action_mask()}
    reference_step_obs = []
    reference_step_rewards = []
    reference_step_dones = []
    reference_step_truncateds = []
    reference_step_masks = []

    for action in np.asarray(actions):
        obs, reward, done, truncated, info = reference_env.step(int(action))
        reference_step_obs.append(obs)
        reference_step_rewards.append(reward)
        reference_step_dones.append(done)
        reference_step_truncateds.append(truncated)
        reference_step_masks.append(info["mask"])

    (
        _,
        obs_reset_jax,
        mask_reset_jax,
        obses_jax,
        rewards_jax,
        dones_jax,
        truncateds_jax,
        masks_jax,
    ) = compiled

    np.testing.assert_allclose(np.asarray(obs_reset_jax), obs_reference_reset, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(mask_reset_jax), info_reference_reset["mask"])
    np.testing.assert_allclose(np.asarray(obses_jax), np.asarray(reference_step_obs), atol=1e-6)
    np.testing.assert_allclose(np.asarray(rewards_jax), np.asarray(reference_step_rewards), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(dones_jax), np.asarray(reference_step_dones))
    np.testing.assert_array_equal(np.asarray(truncateds_jax), np.asarray(reference_step_truncateds))
    np.testing.assert_array_equal(np.asarray(masks_jax), np.asarray(reference_step_masks))


def test_move_path_is_not_stored_in_environment_state():
    reference_env, jax_env, jax_params, key, state, _, _ = _reset_synced_envs(seed=7, t_max=3)

    raw_action = _first_child_path(_reference_children_array(reference_env), reference_env.root_node)[0]
    action = raw_action
    reference_env.step(action)
    state, _, _, _, _, _ = jax_env.step_with_params(state, _jax_action(action), jax_params)
    move_action = reference_env.num_nodes
    reference_env.step(move_action)
    state, _, _, _, _, _ = jax_env.step_with_params(state, _jax_action(move_action), jax_params)

    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")

    reference_env.reset()
    state, _, _ = jax_env.reset_with_params(key, jax_params)
    _sync_reference_from_state(reference_env, state)
    raw_action = _first_child_path(_reference_children_array(reference_env), reference_env.root_node)[0]
    action = raw_action

    done_reference = False
    done_jax = False
    for _ in range(reference_env.t_max - 1):
        _, _, done_reference, _, _ = reference_env.step(action)
        state, _, _, done_jax, _, _ = jax_env.step_with_params(state, _jax_action(action), jax_params)

    assert not done_reference
    assert not bool(done_jax)
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


def test_timeout_masks_to_move_action():
    reference_env, jax_env, jax_params, _, state, _, _ = _reset_synced_envs(seed=11, t_max=3)

    raw_action = _first_child_path(_reference_children_array(reference_env), reference_env.root_node)[0]
    action = raw_action
    info_reference = {"mask": reference_env.get_action_mask()}
    info_jax = {"mask": jax_env.get_action_mask(state)}
    for _ in range(reference_env.t_max - 1):
        _, _, _, _, info_reference = reference_env.step(action)
        state, _, _, _, _, info_jax = jax_env.step_with_params(state, _jax_action(action), jax_params)

    np.testing.assert_array_equal(
        info_reference["mask"],
        np.eye(reference_env.action_size, dtype=bool)[reference_env.num_nodes],
    )
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])

    with pytest.raises(ValueError, match=f"Invalid action {action}"):
        reference_env.step(action)

    move_action = reference_env.num_nodes
    _, reference_reward, reference_done, _, _ = reference_env.step(move_action)
    state, _, reward_jax, done_jax, _, _ = jax_env.step_with_params(state, _jax_action(move_action), jax_params)

    assert reference_done
    assert bool(done_jax)
    assert reference_reward > 0.0
    assert float(reward_jax) > 0.0
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


def test_visit_all_once_then_terminate_is_optimal_reference():
    env = ReferenceDecisionTreeEnv(
        num_nodes=7,
        beta_move=100.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=1.0,
        wm_decay=1.0,
        t_max=8,
        cost=0.0,
        shuffle_nodes=True,
        seed=19,
    )

    _, info = env.reset()
    child_nodes = _reference_children_array(env)
    points = env.points
    root = int(env.root_node)

    visit_order = _bfs_visit_order(child_nodes, root)
    assert len(visit_order) == env.num_nodes
    assert len(set(visit_order)) == env.num_nodes

    for raw_action in visit_order:
        action = raw_action
        assert bool(info["mask"][action])
        _, _, done, truncated, info = env.step(action)
        assert not done
        assert not truncated

    _, reward, done, truncated, _ = env.step(env.num_nodes)
    assert done
    assert not truncated

    optimal_scaled = _optimal_path_reward_raw(child_nodes, points, root) * env.scale_factor
    np.testing.assert_allclose(reward, optimal_scaled, atol=1e-6)


def test_visit_all_once_then_terminate_is_optimal_jax():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        t_max=8,
        shuffle_nodes=True,
    )
    params = env.params(
        beta_move=100.0,
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
    env = JaxDecisionTreeEnv(num_nodes=3, shuffle_nodes=False)
    params = env.params(learning_rate=1.0, lamda_backup=1.0, backup_steps=0)
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
    env = JaxDecisionTreeEnv(num_nodes=5, shuffle_nodes=False)
    params = env.params(learning_rate=1.0, lamda_backup=1.0, backup_steps=1)
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

    env_no_wm_backup = JaxDecisionTreeEnv(
        num_nodes=3,
        shuffle_nodes=False,
    )
    no_wm_params = env_no_wm_backup.params(
        learning_rate=1.0,
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

    env_wm_backup = JaxDecisionTreeEnv(
        num_nodes=3,
        shuffle_nodes=False,
    )
    wm_params = env_wm_backup.params(
        learning_rate=1.0,
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
