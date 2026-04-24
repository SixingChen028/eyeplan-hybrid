import jax
import jax.numpy as jnp
import numpy as np

from modules.jax_environment import JaxDecisionTreeEnv
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


def _reference_planner_known(env: ReferenceDecisionTreeEnv) -> np.ndarray:
    return env.planner_known


def _reference_planner_expanded(env: ReferenceDecisionTreeEnv) -> np.ndarray:
    return env.planner_expanded


def _assert_state_matches_reference(env: ReferenceDecisionTreeEnv, state) -> None:
    np.testing.assert_equal(int(state.time_elapsed), env.time_elapsed)
    np.testing.assert_equal(int(state.fixation_node), env.fixation_node)
    np.testing.assert_equal(int(state.root_node), env.root_node)

    np.testing.assert_array_equal(np.asarray(state.child_nodes), _reference_children_array(env))
    np.testing.assert_array_equal(np.asarray(state.parent_nodes), _reference_parent_array(env))
    np.testing.assert_array_equal(np.asarray(state.points), env.points)

    np.testing.assert_array_equal(np.asarray(state.planner_known), _reference_planner_known(env))
    np.testing.assert_array_equal(np.asarray(state.planner_expanded), _reference_planner_expanded(env))

    np.testing.assert_allclose(np.asarray(state.q_values), env.get_q_values(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.g_values), env.get_path_values(), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.n_visits), env.get_num_visits())

    np.testing.assert_allclose(np.asarray(state.activation), env.activation, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.active_mask), env.active_mask)


def _make_envs(seed: int = 0, t_max: int = 20):
    kwargs = dict(
        num_nodes=3,
        beta_move=50.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=0.5,
        wm_decay=1.0,
        t_max=t_max,
        cost=0.01,
        scale_factor=1 / 8,
        shuffle_nodes=False,
        mask_fixation=True,
    )

    reference_env = ReferenceDecisionTreeEnv(seed=seed, **kwargs)
    reference_env.point_set = np.array([1.0], dtype=np.float32)

    jax_env = JaxDecisionTreeEnv(
        num_nodes=kwargs["num_nodes"],
        beta_move=kwargs["beta_move"],
        eps_move=kwargs["eps_move"],
        learning_rate=kwargs["learning_rate"],
        lamda_backup=kwargs["lamda_backup"],
        wm_decay=kwargs["wm_decay"],
        t_max=kwargs["t_max"],
        cost=kwargs["cost"],
        scale_factor=kwargs["scale_factor"],
        shuffle_nodes=kwargs["shuffle_nodes"],
        point_set=jnp.array([1.0], dtype=jnp.float32),
    )

    key = jax.random.PRNGKey(seed)
    return reference_env, jax_env, key


def test_reset_matches_reference_environment():
    reference_env, jax_env, key = _make_envs(seed=3)

    obs_reference, info_reference = reference_env.reset()
    state, obs_jax, info_jax = jax_env.reset(key)

    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])
    _assert_state_matches_reference(reference_env, state)


def test_fixation_step_matches_reference_environment():
    reference_env, jax_env, key = _make_envs(seed=4)

    reference_env.reset()
    state, _, _ = jax_env.reset(key)

    action = 1
    obs_reference, reward_reference, done_reference, truncated_reference, info_reference = reference_env.step(action)
    state, obs_jax, reward_jax, done_jax, truncated_jax, info_jax = jax_env.step(state, action)

    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_allclose(float(reward_jax), reward_reference, atol=1e-6)
    np.testing.assert_equal(bool(done_jax), done_reference)
    np.testing.assert_equal(bool(truncated_jax), truncated_reference)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])

    _assert_state_matches_reference(reference_env, state)


def test_move_step_matches_reference_environment():
    reference_env, jax_env, key = _make_envs(seed=5)

    reference_env.reset()
    state, _, _ = jax_env.reset(key)

    reference_env.step(1)
    state, _, _, _, _, _ = jax_env.step(state, 1)

    move_action = reference_env.num_nodes
    obs_reference, reward_reference, done_reference, truncated_reference, info_reference = reference_env.step(move_action)
    state, obs_jax, reward_jax, done_jax, truncated_jax, info_jax = jax_env.step(state, move_action)

    np.testing.assert_allclose(np.asarray(obs_jax), obs_reference, atol=1e-6)
    np.testing.assert_allclose(float(reward_jax), reward_reference, atol=1e-6)
    np.testing.assert_equal(bool(done_jax), done_reference)
    np.testing.assert_equal(bool(truncated_jax), truncated_reference)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_reference["mask"])

    chosen_path_jax = np.asarray(state.chosen_path)[: int(state.chosen_path_len)]
    np.testing.assert_array_equal(chosen_path_jax, np.asarray(reference_env.chosen_path, dtype=np.int32))


def test_compiled_rollout_matches_reference_environment():
    reference_env, jax_env, key = _make_envs(seed=6)

    actions = jnp.array([1, reference_env.num_nodes], dtype=jnp.int32)

    def rollout(k, action_seq):
        state, reset_obs, reset_info = jax_env.reset(k)

        def body_fn(carry, action):
            state = carry
            state, obs, reward, done, truncated, info = jax_env.step(state, action)
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

    obs_reference_reset, info_reference_reset = reference_env.reset()
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


def test_chosen_path_does_not_leak_across_trials():
    reference_env, jax_env, key = _make_envs(seed=7, t_max=3)

    reference_env.reset()
    state, _, _ = jax_env.reset(key)

    reference_env.step(1)
    state, _, _, _, _, _ = jax_env.step(state, 1)
    move_action = reference_env.num_nodes
    reference_env.step(move_action)
    state, _, _, _, _, _ = jax_env.step(state, move_action)

    assert len(reference_env.chosen_path) > 0
    assert int(state.chosen_path_len) > 0

    reference_env.reset()
    state, _, _ = jax_env.reset(key)

    done_reference = False
    done_jax = False
    for _ in range(reference_env.t_max - 1):
        _, _, done_reference, _, _ = reference_env.step(1)
        state, _, _, done_jax, _, _ = jax_env.step(state, 1)

    assert not done_reference
    assert not bool(done_jax)
    np.testing.assert_array_equal(np.asarray(state.chosen_path[: int(state.chosen_path_len)]), np.asarray([], dtype=np.int32))
    assert reference_env.chosen_path == []


def test_timeout_forces_move_action():
    reference_env, jax_env, key = _make_envs(seed=11, t_max=3)

    reference_env.reset()
    state, _, _ = jax_env.reset(key)

    for _ in range(reference_env.t_max - 1):
        reference_env.step(1)
        state, _, _, _, _, _ = jax_env.step(state, 1)

    _, reference_reward, reference_done, _, _ = reference_env.step(1)
    state, _, reward_jax, done_jax, _, _ = jax_env.step(state, 1)

    assert reference_done
    assert bool(done_jax)
    assert reference_reward > 0.0
    assert float(reward_jax) > 0.0
    assert len(reference_env.chosen_path) > 0
    assert int(state.chosen_path_len) > 0


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
        mask_fixation=True,
        seed=19,
    )

    _, info = env.reset()
    child_nodes = _reference_children_array(env)
    points = env.points
    root = int(env.root_node)

    visit_order = _bfs_visit_order(child_nodes, root)
    assert len(visit_order) == env.num_nodes
    assert len(set(visit_order)) == env.num_nodes

    for action in visit_order:
        assert bool(info["mask"][action])
        _, _, done, truncated, info = env.step(action)
        assert not done
        assert not truncated

    _, reward, done, truncated, _ = env.step(env.num_nodes)
    assert done
    assert not truncated

    optimal_scaled = _optimal_path_reward_raw(child_nodes, points, root) * env.scale_factor
    np.testing.assert_allclose(reward, optimal_scaled, atol=1e-6)

    chosen_path = np.asarray(env.chosen_path, dtype=np.int32)
    chosen_scaled = float(points[chosen_path].sum()) * env.scale_factor
    np.testing.assert_allclose(chosen_scaled, optimal_scaled, atol=1e-6)


def test_visit_all_once_then_terminate_is_optimal_jax():
    env = JaxDecisionTreeEnv(
        num_nodes=7,
        beta_move=100.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=1.0,
        wm_decay=1.0,
        t_max=8,
        cost=0.0,
        shuffle_nodes=True,
    )

    state, _, info = env.reset(jax.random.PRNGKey(19))
    child_nodes = np.asarray(state.child_nodes)
    points = np.asarray(state.points)
    root = int(state.root_node)

    visit_order = _bfs_visit_order(child_nodes, root)
    assert len(visit_order) == env.num_nodes
    assert len(set(visit_order)) == env.num_nodes

    for action in visit_order:
        assert bool(np.asarray(info["mask"])[action])
        state, _, _, done, truncated, info = env.step(state, action)
        assert not bool(done)
        assert not bool(truncated)

    state, _, reward, done, truncated, _ = env.step(state, env.num_nodes)
    assert bool(done)
    assert not bool(truncated)

    optimal_scaled = _optimal_path_reward_raw(child_nodes, points, root) * env.scale_factor
    np.testing.assert_allclose(float(reward), optimal_scaled, atol=1e-6)

    chosen_path = np.asarray(state.chosen_path)[: int(state.chosen_path_len)]
    chosen_scaled = float(points[chosen_path].sum()) * env.scale_factor
    np.testing.assert_allclose(chosen_scaled, optimal_scaled, atol=1e-6)
