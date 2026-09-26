import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import DecisionTreeEnv
from modules.network import flatten_observation

_, _DEFAULT_PARAMS = load_canonical_defaults()


def _env(**overrides):
    params = dict(_DEFAULT_PARAMS)
    params.update(overrides)
    return DecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        use_g_values_obs=bool(params["use_g_values_obs"]),
        use_q_values_obs=bool(params["use_q_values_obs"]),
        use_n_visits_obs=bool(params["use_n_visits_obs"]),
        use_is_terminal_obs=bool(params["use_is_terminal_obs"]),
        use_time_elapsed_obs=bool(params["use_time_elapsed_obs"]),
        point_set=params["point_set"],
    )


def _env_params(env, **overrides):
    params = {key: _DEFAULT_PARAMS[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    params.update(overrides)
    return env.make_params(**params)


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


def _descendant_count(child_nodes: np.ndarray, node: int) -> int:
    left = int(child_nodes[node, 0])
    if left < 0:
        return 1

    right = int(child_nodes[node, 1])
    return 1 + _descendant_count(child_nodes, left) + _descendant_count(child_nodes, right)


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


def _obs_size(env: DecisionTreeEnv) -> int:
    return int(flatten_observation(env.observation_template).shape[0])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"wm_decay": 1.1}, "wm_decay"),
        ({"wm_neighbor_activation": 0.0}, "wm_neighbor_activation"),
        ({"wm_neighbor_activation": 1.1}, "wm_neighbor_activation"),
        ({"cost": -0.1}, "cost"),
    ],
)
def test_make_params_validates_dynamic_ranges(overrides, message):
    env = _env(num_nodes=3)
    params = {key: _DEFAULT_PARAMS[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    params.update(overrides)

    with pytest.raises(AssertionError, match=message):
        env.make_params(**params)


@pytest.mark.slow
def test_reset_uses_raw_node_ids_by_default():
    env = _env(num_nodes=15)

    state, obs, info = env.reset(jax.random.PRNGKey(23), _env_params(env))
    root = int(state.root_node)

    np.testing.assert_array_equal(np.asarray(obs.fixation), np.eye(env.num_nodes)[root])
    np.testing.assert_array_equal(np.asarray(obs.root), np.eye(env.num_nodes)[root])

    expected_mask = np.zeros(env.action_size, dtype=bool)
    expected_mask[: env.num_nodes] = np.asarray(state.activation) > 0
    expected_mask[root] = True
    expected_mask[-1] = True
    np.testing.assert_array_equal(np.asarray(info["mask"]), expected_mask)


@pytest.mark.slow
def test_shuffle_nodes_randomizes_sibling_order():
    env = _env(num_nodes=15)
    params = _env_params(env)

    descendant_diffs: list[int] = []
    for seed in range(128):
        state, _, _ = env.reset(jax.random.PRNGKey(seed), params)
        child_nodes = np.asarray(state.child_nodes)

        for children in child_nodes:
            left = int(children[0])
            if left < 0:
                continue

            right = int(children[1])
            diff = _descendant_count(child_nodes, left) - _descendant_count(child_nodes, right)
            if diff != 0:
                descendant_diffs.append(diff)

    assert any(diff < 0 for diff in descendant_diffs)
    assert any(diff > 0 for diff in descendant_diffs)


@pytest.mark.parametrize(
    "flag, field, expected_delta",
    [
        ("use_g_values_obs", "g_values", 7),
        ("use_q_values_obs", "q_values", 7),
        ("use_n_visits_obs", "n_visits", 7),
        ("use_is_terminal_obs", "is_terminal", 7),
        ("use_time_elapsed_obs", "time_elapsed", 1),
    ],
)
def test_static_observation_flags_control_feature_size(flag, field, expected_delta):
    enabled_env = _env(num_nodes=7, **{flag: True})
    disabled_env = _env(num_nodes=7, **{flag: False})

    assert _obs_size(enabled_env) == _obs_size(disabled_env) + expected_delta
    assert getattr(disabled_env.observation_template, field) is None


def test_look_forgets_inactive_node_memory():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=0.0, wm_neighbor_activation=0.25)
    state = env._sample_initial_state(jax.random.PRNGKey(11))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        q_values=jnp.arange(7, dtype=jnp.float32),
        n_visits=jnp.ones((7,), dtype=jnp.int32),
        g_values=jnp.arange(1, 8, dtype=jnp.float32),
        activation=jnp.ones((7,), dtype=jnp.float32),
    )

    state = env._look(state, jnp.asarray(1, dtype=jnp.int32), params)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(np.asarray(state.q_values)[inactive_mask], 0.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.n_visits)[inactive_mask], np.zeros(np.sum(inactive_mask)))
    np.testing.assert_allclose(
        np.asarray(state.g_values)[inactive_mask & (np.arange(env.num_nodes) != int(state.root_node))],
        env.min_path_value, atol=1e-6
    )
    assert float(state.g_values[int(state.root_node)]) == 0.0


def test_backup_excludes_reward_of_unobserved_node():
    # Fixating child C activates its parent P as a neighbor, so the backup walks into P.
    # P has never been fixated (n_visits == 0), so its reward must not enter the target:
    # Q(P) should back up only the child value, not points[P].
    env = _env(num_nodes=3)
    params = _env_params(env, wm_decay=1.0)
    state = env._sample_initial_state(jax.random.PRNGKey(7))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        child_nodes=jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32),
        parent_nodes=jnp.array([-1, 0, 0], dtype=jnp.int32),
        points=jnp.array([5.0, 3.0, 0.0], dtype=jnp.float32),
        q_values=jnp.zeros((3,), dtype=jnp.float32),
        n_visits=jnp.zeros((3,), dtype=jnp.int32),
        activation=jnp.ones((3,), dtype=jnp.float32),
    )

    # Fixate child C=1. C becomes observed (n_visits -> 1) and backs up its own reward.
    # Parent P=0 stays unobserved (n_visits == 0).
    state = env._look(state, jnp.asarray(1, dtype=jnp.int32), params)

    assert int(state.n_visits[0]) == 0
    # Q(C) = points[C]; Q(P) = Q(C) only (points[P] excluded because P is unobserved).
    np.testing.assert_allclose(float(state.q_values[1]), 3.0, atol=1e-6)
    np.testing.assert_allclose(float(state.q_values[0]), 3.0, atol=1e-6)


def test_backup_includes_reward_of_observed_node():
    # Same structure, but P has been observed (n_visits > 0), so its reward is available.
    env = _env(num_nodes=3)
    params = _env_params(env, wm_decay=1.0)
    state = env._sample_initial_state(jax.random.PRNGKey(7))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        child_nodes=jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32),
        parent_nodes=jnp.array([-1, 0, 0], dtype=jnp.int32),
        points=jnp.array([5.0, 3.0, 0.0], dtype=jnp.float32),
        q_values=jnp.zeros((3,), dtype=jnp.float32),
        n_visits=jnp.array([1, 0, 0], dtype=jnp.int32),
        activation=jnp.ones((3,), dtype=jnp.float32),
    )

    state = env._look(state, jnp.asarray(1, dtype=jnp.int32), params)

    # Q(P) = points[P] + Q(C) = 5 + 3.
    np.testing.assert_allclose(float(state.q_values[0]), 8.0, atol=1e-6)


def test_terminal_backup_has_zero_continuation():
    env = _env(num_nodes=3)
    params = _env_params(env, wm_decay=1.0)
    state = env._sample_initial_state(jax.random.PRNGKey(71))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        child_nodes=jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32),
        parent_nodes=jnp.array([-1, 0, 0], dtype=jnp.int32),
        points=jnp.array([0.0, 3.0, 0.0], dtype=jnp.float32),
        q_values=jnp.zeros((3,), dtype=jnp.float32),
        n_visits=jnp.zeros((3,), dtype=jnp.int32),
        activation=jnp.ones((3,), dtype=jnp.float32),
    )

    state = env._look(state, jnp.asarray(1, dtype=jnp.int32), params)

    np.testing.assert_allclose(float(state.q_values[1]), 3.0, atol=1e-6)


def test_activation_masks_observation_keeps_active_g_values():
    env = _env(num_nodes=7)
    state = env._sample_initial_state(jax.random.PRNGKey(12))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(3, dtype=jnp.int32),
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        points=jnp.zeros((7,), dtype=jnp.float32),
        g_values=jnp.array([0.0, 4.0, 2.0, 12.0, 5.0, 6.0, 7.0], dtype=jnp.float32),
        n_visits=jnp.array([1, 1, 0, 1, 0, 0, 0], dtype=jnp.int32),
        activation=jnp.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        is_terminal=jnp.array([False, False, False, True, False, False, False], dtype=jnp.bool_),
    )

    obs = env._get_obs(state)

    np.testing.assert_array_equal(np.asarray(obs.child), np.zeros(env.num_nodes))
    assert np.asarray(obs.g_values)[3] == 12.0
    assert np.asarray(obs.n_visits)[3] == 1.0
    assert np.asarray(obs.is_terminal)[3] == 1.0


def test_activation_masks_observation_hides_inactive_root_values():
    env = _env(num_nodes=3)
    state = env._sample_initial_state(jax.random.PRNGKey(121))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(1, dtype=jnp.int32),
        child_nodes=jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32),
        parent_nodes=jnp.array([-1, 0, 0], dtype=jnp.int32),
        g_values=jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
        q_values=jnp.array([7.0, 8.0, 9.0], dtype=jnp.float32),
        n_visits=jnp.array([1, 2, 3], dtype=jnp.int32),
        activation=jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
    )

    obs = env._get_obs(state)

    assert np.asarray(obs.g_values)[0] == 0.0
    assert np.asarray(obs.q_values)[0] == 0.0
    assert np.asarray(obs.n_visits)[0] == 0.0


def test_observation_masking_decouples_known_path_from_activation():
    # With persistence enabled and activation_masks_observation, g_values uses the
    # observation mask only: active nodes are shown even if unknown.
    env = _env(num_nodes=7)
    state = env._sample_initial_state(jax.random.PRNGKey(13))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(2, dtype=jnp.int32),
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        points=jnp.zeros((7,), dtype=jnp.float32),
        g_values=jnp.array([0.0, 1.0, 2.0, 100.0, 5.0, 90.0, 3.0], dtype=jnp.float32),
        n_visits=jnp.array([1, 0, 1, 0, 0, 0, 0], dtype=jnp.int32),
        activation=jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
        is_terminal=jnp.array([False, False, False, True, False, False, False], dtype=jnp.bool_),
    )

    obs = env._get_obs(state)

    # node 4 is active but unknown (parent unvisited): shown in g_values via the obs mask.
    assert np.asarray(obs.g_values)[4] == 5.0
    # node 5 is inactive: hidden from g_values regardless of being known.
    assert np.asarray(obs.g_values)[5] == 0.0


def test_clear_inactive_memory_always_clears_inactive_node_memory():
    env = _env(num_nodes=7)
    state = env._sample_initial_state(jax.random.PRNGKey(14))
    state = state._replace(
        q_values=jnp.arange(env.num_nodes, dtype=jnp.float32),
        n_visits=jnp.ones((env.num_nodes,), dtype=jnp.int32),
        g_values=jnp.arange(1, env.num_nodes + 1, dtype=jnp.float32),
        is_terminal=jnp.array([False, False, False, True, False, False, True], dtype=jnp.bool_),
        activation=jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
    )

    state = env._clear_inactive_memory(state)

    inactive_mask = np.asarray(state.activation) == 0.0
    np.testing.assert_allclose(np.asarray(state.q_values)[inactive_mask], 0.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.n_visits)[inactive_mask], np.zeros(np.sum(inactive_mask)))
    np.testing.assert_allclose(
        np.asarray(state.g_values)[inactive_mask & (np.arange(env.num_nodes) != int(state.root_node))],
        env.min_path_value, atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(state.is_terminal),
        np.array([False, False, False, False, False, False, True]),
    )


def test_update_activation_preserves_q_values():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=0.0)
    state, _, _ = env.reset(jax.random.PRNGKey(17), params)
    q_values = jnp.arange(env.num_nodes, dtype=jnp.float32)
    state = state._replace(
        q_values=q_values,
        activation=jnp.zeros((env.num_nodes,), dtype=jnp.float32),
    )

    state = env._update_activation(state, params)

    np.testing.assert_allclose(np.asarray(state.q_values), np.asarray(q_values), atol=1e-6)


def test_update_activation_refreshes_fixation_neighborhood_after_drop():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=0.0)
    state = env._sample_initial_state(jax.random.PRNGKey(18))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(1, dtype=jnp.int32),
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, -1, -1], dtype=jnp.int32),
        activation=jnp.ones((7,), dtype=jnp.float32),
    )

    state = env._update_activation(state, params)

    expected_activation = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(state.activation), expected_activation)


def test_update_activation_does_not_refresh_root_outside_fixation_neighborhood():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=0.0)
    state = env._sample_initial_state(jax.random.PRNGKey(181))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(3, dtype=jnp.int32),
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, -1, -1], dtype=jnp.int32),
        activation=jnp.ones((7,), dtype=jnp.float32),
    )

    state = env._update_activation(state, params)

    expected_activation = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(state.activation), expected_activation)


def test_update_activation_uses_neighbor_activation_without_reducing_activation():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=1.0, wm_neighbor_activation=0.25)
    state = env._sample_initial_state(jax.random.PRNGKey(19))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(1, dtype=jnp.int32),
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, -1, -1], dtype=jnp.int32),
        activation=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    state = env._update_activation(state, params)

    expected_activation = np.array([0.25, 1.0, 0.0, 1.0, 0.25, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(state.activation), expected_activation)


def test_update_activation_tracks_consumed_rng_key():
    env = _env(num_nodes=7)
    params = _env_params(env)
    state, _, _ = env.reset(jax.random.PRNGKey(17), params)
    expected_key, _ = jax.random.split(state.rng_key)

    state = env._update_activation(state, params)

    np.testing.assert_array_equal(np.asarray(state.rng_key), np.asarray(expected_key))


def test_root_action_is_legal_when_root_is_inactive():
    env = _env(num_nodes=3)
    state = env._sample_initial_state(jax.random.PRNGKey(182))
    state = state._replace(
        root_node=jnp.asarray(0, dtype=jnp.int32),
        time_elapsed=jnp.asarray(0, dtype=jnp.int32),
        activation=jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
    )

    mask = np.asarray(env._get_action_mask(state))

    np.testing.assert_array_equal(mask, np.array([True, True, False, True]))


def test_move_reward_samples_one_path_and_reports_choice_path():
    env = _env(num_nodes=3, t_max=3, scale_factor=1.0)
    params = _env_params(env, cost=0.0)
    state, _, _ = env.reset(jax.random.PRNGKey(101), params)
    state = state._replace(
        points=jnp.array([0.0, 2.0, 6.0], dtype=jnp.float32),
        q_values=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )

    state, _, reward, done, info = env.step(state, _jax_action(env.num_nodes), params)

    assert bool(done)
    np.testing.assert_allclose(float(reward), 6.0, atol=1e-6)
    np.testing.assert_allclose(float(info["move_reward"]), 6.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info["choice_path"]), np.array([2, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(state.fixation_node), np.asarray(2, dtype=np.int32))
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


def test_detailed_move_trace_preserves_step_behavior_and_records_each_movement():
    env = _env(num_nodes=7, scale_factor=1.0)
    params = _env_params(env, wm_decay=1.0, cost=0.0)
    state, _, _ = env.reset(jax.random.PRNGKey(104), params)
    state = state._replace(
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        points=jnp.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], dtype=jnp.float32),
        q_values=jnp.array([0.0, 5.0, 0.0, 1.0, 10.0, 0.0, 0.0], dtype=jnp.float32),
        activation=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    action = _jax_action(env.num_nodes)

    expected_state, expected_obs, expected_reward, expected_done, expected_info = env.step(state, action, params)
    traced_state, traced_obs, traced_reward, traced_done, traced_info = env.step_with_move_trace(
        state,
        action,
        params,
    )

    jax.tree_util.tree_map(np.testing.assert_array_equal, traced_state, expected_state)
    jax.tree_util.tree_map(np.testing.assert_array_equal, traced_obs, expected_obs)
    np.testing.assert_array_equal(traced_reward, expected_reward)
    np.testing.assert_array_equal(traced_done, expected_done)
    np.testing.assert_array_equal(traced_info["choice_path"], expected_info["choice_path"])

    trace = traced_info["move_trace"]
    trace_len = int(trace.length)
    path = np.asarray(expected_info["choice_path"])
    path = path[path >= 0]
    np.testing.assert_array_equal(np.asarray(trace.actions[:trace_len]), np.concatenate([[0], path]))
    assert float(trace.qs[0, 4]) == 0.0
    np.testing.assert_array_equal(np.asarray(trace.qs[trace_len - 1]), np.asarray(traced_state.q_values))
    np.testing.assert_array_equal(np.asarray(trace.counts[trace_len - 1]), np.asarray(traced_state.n_visits))


def test_movement_look_skips_q_update_but_clears_inactive_memory():
    env = _env(num_nodes=3)
    params = _env_params(env, wm_decay=0.0)
    state, _, _ = env.reset(jax.random.PRNGKey(102), params)
    state = state._replace(
        q_values=jnp.array([5.0, 7.0, 9.0], dtype=jnp.float32),
        n_visits=jnp.ones((env.num_nodes,), dtype=jnp.int32),
        activation=jnp.zeros((env.num_nodes,), dtype=jnp.float32),
    )

    state = env._look(state, _jax_action(1), params, skip_q_update=True)

    np.testing.assert_allclose(float(state.q_values[1]), 7.0, atol=1e-6)
    np.testing.assert_allclose(float(state.q_values[2]), 0.0, atol=1e-6)
    assert int(state.n_visits[2]) == 0


def test_movement_forgetting_changes_downstream_choice():
    env = _env(num_nodes=7, scale_factor=1.0)
    params = _env_params(env, wm_decay=1.0, cost=0.0)
    state, _, _ = env.reset(jax.random.PRNGKey(103), params)
    state = state._replace(
        child_nodes=jnp.array(
            [[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            dtype=jnp.int32,
        ),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32),
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        points=jnp.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], dtype=jnp.float32),
        q_values=jnp.array([0.0, 5.0, 0.0, 1.0, 10.0, 0.0, 0.0], dtype=jnp.float32),
        activation=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    _, _, reward, _, info = env.step(state, _jax_action(env.num_nodes), params)

    np.testing.assert_allclose(float(reward), 3.0, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(info["choice_path"]),
        np.array([1, 3, -1, -1, -1, -1, -1], dtype=np.int32),
    )


@pytest.mark.slow
def test_compiled_rollout_matches_eager_rollout():
    env = _env(num_nodes=7, t_max=20, scale_factor=1 / 8, point_set=jnp.array([1.0], dtype=jnp.float32))
    params = _env_params(env, wm_decay=1.0, cost=0.01)
    key = jax.random.PRNGKey(6)
    reset_state, _, _ = env.reset(key, params)
    path_actions = _first_child_path(np.asarray(reset_state.child_nodes), int(reset_state.root_node))
    actions = jnp.array(path_actions + [env.num_nodes], dtype=jnp.int32)

    def rollout(k, action_seq):
        state, reset_obs, reset_info = env.reset(k, params)

        def body_fn(carry, action):
            state = carry
            state, obs, reward, done, info = env.step(state, action, params)
            output = (obs, reward, done, info["mask"])
            return state, output

        state, (obses, rewards, dones, masks) = jax.lax.scan(body_fn, state, action_seq)
        return state, reset_obs, reset_info["mask"], obses, rewards, dones, masks

    eager = rollout(key, actions)
    compiled = jax.jit(rollout)(key, actions)

    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager[0]),
        jax.tree_util.tree_leaves(compiled[0]),
    ):
        np.testing.assert_allclose(np.asarray(compiled_leaf), np.asarray(eager_leaf), atol=1e-6)

    for eager_item, compiled_item in zip(eager[1:], compiled[1:]):
        for eager_leaf, compiled_leaf in zip(
            jax.tree_util.tree_leaves(eager_item),
            jax.tree_util.tree_leaves(compiled_item),
        ):
            np.testing.assert_allclose(np.asarray(compiled_leaf), np.asarray(eager_leaf), atol=1e-6)

@pytest.mark.slow
def test_move_path_is_not_stored_in_environment_state():
    env = _env(num_nodes=7, t_max=3)
    params = _env_params(env)
    key = jax.random.PRNGKey(7)
    state, _, _ = env.reset(key, params)
    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    state, _, _, _, _ = env.step(state, _jax_action(action), params)
    state, _, _, _, _ = env.step(state, _jax_action(env.num_nodes), params)

    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")

    state, _, _ = env.reset(key, params)
    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    done_jax = False
    for _ in range(env.t_max - 1):
        state, _, _, done_jax, _ = env.step(state, _jax_action(action), params)
    assert not bool(done_jax)
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


@pytest.mark.slow
def test_timeout_masks_to_move_action():
    env = _env(num_nodes=7, t_max=3)
    params = _env_params(env)
    state, _, info_jax = env.reset(jax.random.PRNGKey(11), params)

    action = _first_child_path(np.asarray(state.child_nodes), int(state.root_node))[0]
    for _ in range(env.t_max - 1):
        state, _, _, _, info_jax = env.step(state, _jax_action(action), params)

    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), np.eye(env.action_size, dtype=bool)[env.num_nodes])
    state, _, reward_jax, done_jax, _ = env.step(state, _jax_action(env.num_nodes), params)

    assert bool(done_jax)
    assert not hasattr(state, "chosen_path")
    assert not hasattr(state, "chosen_path_len")


@pytest.mark.slow
def test_visit_all_once_then_terminate_is_optimal_jax():
    env = _env(num_nodes=7, t_max=8)
    params = _env_params(env, wm_decay=1.0, cost=0.0)

    state, _, info = env.reset(jax.random.PRNGKey(19), params)
    child_nodes = np.asarray(state.child_nodes)
    points = np.asarray(state.points)
    root = int(state.root_node)

    visit_order = _bfs_visit_order(child_nodes, root)
    assert len(visit_order) == env.num_nodes
    assert len(set(visit_order)) == env.num_nodes

    for raw_action in visit_order:
        action = raw_action
        assert bool(np.asarray(info["mask"])[action])
        state, _, _, done, info = env.step(state, _jax_action(action), params)
        assert not bool(done)

    state, _, reward, done, _ = env.step(state, _jax_action(env.num_nodes), params)
    assert bool(done)

    optimal_scaled = _optimal_path_reward_raw(child_nodes, points, root) * env.scale_factor
    np.testing.assert_allclose(float(reward), optimal_scaled, atol=1e-6)


def test_wm_decay_zero_multi_step_backup_uses_refreshed_activation():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=0.0)
    child_nodes = jnp.array(
        [[1, -1], [2, -1], [3, -1], [4, -1], [5, -1], [-1, -1], [-1, -1]],
        dtype=jnp.int32,
    )
    parent_nodes = jnp.array([-1, 0, 1, 2, 3, 4, -1], dtype=jnp.int32)
    points = jnp.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 0.0], dtype=jnp.float32)

    state = env._sample_initial_state(jax.random.PRNGKey(0))
    state = state._replace(
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        points=points,
        q_values=jnp.zeros((7,), dtype=jnp.float32),
        activation=jnp.zeros((7,), dtype=jnp.float32),
    )

    for action in [0, 1, 2, 3]:
        state = env._look(state, _jax_action(action), params)

    np.testing.assert_allclose(
        np.asarray(state.q_values),
        np.array([0.0, 0.0, 110.0, 100.0, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_inactive_root_stops_gated_ancestor_backup():
    env = _env(num_nodes=3)
    params = _env_params(env, wm_decay=0.0)
    state = env._sample_initial_state(jax.random.PRNGKey(961))
    state = state._replace(
        child_nodes=jnp.array([[1, -1], [2, -1], [-1, -1]], dtype=jnp.int32),
        parent_nodes=jnp.array([-1, 0, 1], dtype=jnp.int32),
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(0, dtype=jnp.int32),
        points=jnp.array([0.0, 1.0, 10.0], dtype=jnp.float32),
        q_values=jnp.zeros((3,), dtype=jnp.float32),
        n_visits=jnp.ones((3,), dtype=jnp.int32),
        activation=jnp.zeros((3,), dtype=jnp.float32),
    )

    state = env._look(state, _jax_action(2), params)

    np.testing.assert_allclose(np.asarray(state.activation), np.array([0.0, 1.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.q_values), np.array([0.0, 11.0, 10.0], dtype=np.float32), atol=1e-6)


def test_backup_preserves_negative_value_when_other_child_is_unavailable():
    child_nodes = jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32)
    parent_nodes = jnp.array([-1, 0, 0], dtype=jnp.int32)
    points = jnp.array([0.0, -8.0, 0.0], dtype=jnp.float32)
    activation = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)

    env = _env(num_nodes=3)
    params = _env_params(env)
    state, _, _ = env.reset(jax.random.PRNGKey(0), params)
    state = state._replace(
        q_values=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        child_nodes=child_nodes,
        parent_nodes=parent_nodes,
        root_node=jnp.asarray(0, dtype=jnp.int32),
        fixation_node=jnp.asarray(1, dtype=jnp.int32),
        points=points,
        n_visits=jnp.ones((3,), dtype=jnp.int32),
        activation=activation,
    )

    updated = env._update_q(state)

    np.testing.assert_allclose(float(updated.q_values[0]), -8.0, atol=1e-6)


@pytest.mark.parametrize(
    'values, active, expected',
    [([-3.0, -8.0], [True, True], -3.0),
     ([-3.0, 10.0], [True, False], -3.0),
     ([-3.0, 10.0], [False, False], 0.0),
     ([2.0, 2.0], [True, True], 2.0)],
)
def test_backup_uses_maximum_of_available_children(values, active, expected):
    env = _env(num_nodes=3)
    state = env._sample_initial_state(jax.random.PRNGKey(0))._replace(
        child_nodes=jnp.array([[1, 2], [-1, -1], [-1, -1]], dtype=jnp.int32),
        q_values=jnp.array([0.0, *values], dtype=jnp.float32),
        activation=jnp.array([1.0, *active], dtype=jnp.float32),
        n_visits=jnp.zeros(3, dtype=jnp.int32),
    )
    assert float(jax.jit(env._backup_target)(state, 0)) == expected


@pytest.mark.parametrize('values, expected', [([2.0, 1.99], [1.0, 0.0]),
                                             ([2.0, 2.0], [0.5, 0.5]),
                                             ([-8.0, -3.0], [0.0, 1.0])])
def test_greedy_movement_preserves_uniform_ties(values, expected):
    env = _env(num_nodes=3)
    np.testing.assert_array_equal(env._greedy_child_probs(jnp.array(values)), expected)


def test_neighbor_refresh_protects_memory_before_inactive_clearing():
    env = _env(num_nodes=7)
    params = _env_params(env, wm_decay=0.0, wm_neighbor_activation=0.25)
    state = env._sample_initial_state(jax.random.PRNGKey(0))._replace(
        root_node=jnp.int32(0),
        child_nodes=jnp.array([[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]),
        parent_nodes=jnp.array([-1, 0, 0, 1, 1, 2, 2]),
        q_values=jnp.arange(7, dtype=jnp.float32),
        n_visits=jnp.ones(7, dtype=jnp.int32),
        activation=jnp.ones(7),
        is_terminal=jnp.array([False, False, False, True, True, True, True]),
    )
    updated = env._look(state, jnp.int32(1), params, skip_q_update=True)
    np.testing.assert_array_equal(updated.q_values, [0, 1, 0, 3, 4, 0, 0])
    np.testing.assert_array_equal(updated.is_terminal, [False, False, False, True, True, False, False])
    assert float(updated.g_values[0]) == 0.0


def test_removed_history_fields_are_absent_from_state_observations_and_move_trace():
    env = _env(num_nodes=7)
    state, obs, _ = env.reset(jax.random.PRNGKey(132), _env_params(env))
    assert "is_discovered" not in state._fields
    assert "fixation_recency" not in state._fields
    assert "recency" not in obs._fields
    trace = env._empty_move_trace()
    assert "is_discovered" not in trace._fields
    assert "fixation_recency" not in trace._fields
