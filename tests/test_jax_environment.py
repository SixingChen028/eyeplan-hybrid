import jax
import jax.numpy as jnp
import numpy as np

from modules.environment import DecisionTreeEnv
from modules.jax_environment import JaxDecisionTreeEnv


def _legacy_children_array(env: DecisionTreeEnv) -> np.ndarray:
    children = -np.ones((env.num_nodes, 2), dtype=np.int32)
    for parent, child_list in env.graph.child_dict.items():
        children[parent] = np.asarray(child_list, dtype=np.int32)
    return children


def _legacy_parent_array(env: DecisionTreeEnv) -> np.ndarray:
    parents = -np.ones((env.num_nodes,), dtype=np.int32)
    for child, parent in env.graph.parent_dict.items():
        parents[child] = parent
    return parents


def _legacy_planner_known(env: DecisionTreeEnv) -> np.ndarray:
    known = np.zeros((env.num_nodes,), dtype=bool)
    for node in env.planner.parents.keys():
        known[node] = True
    return known


def _legacy_planner_expanded(env: DecisionTreeEnv) -> np.ndarray:
    expanded = np.zeros((env.num_nodes,), dtype=bool)
    for node in env.planner.expanded:
        expanded[node] = True
    return expanded


def _assert_state_matches_legacy(env: DecisionTreeEnv, state) -> None:
    np.testing.assert_equal(int(state.time_elapsed), env.time_elapsed)
    np.testing.assert_equal(int(state.fixation_node), env.fixation_node)
    np.testing.assert_equal(int(state.root_node), env.graph.root_node)

    np.testing.assert_array_equal(np.asarray(state.child_nodes), _legacy_children_array(env))
    np.testing.assert_array_equal(np.asarray(state.parent_nodes), _legacy_parent_array(env))
    np.testing.assert_array_equal(np.asarray(state.points), env.graph.points)

    np.testing.assert_array_equal(np.asarray(state.planner_known), _legacy_planner_known(env))
    np.testing.assert_array_equal(np.asarray(state.planner_expanded), _legacy_planner_expanded(env))

    np.testing.assert_allclose(np.asarray(state.q_values), env.get_q_values(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.g_values), env.get_path_values(), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.n_visits), env.get_num_visits())

    np.testing.assert_allclose(np.asarray(state.activation), env.activation, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state.active_mask), env.active_mask)


def _make_envs(seed: int = 0):
    kwargs = dict(
        num_nodes=3,
        beta_move=50.0,
        eps_move=0.0,
        learning_rate=1.0,
        wm_decay=1.0,
        t_max=20,
        cost=0.01,
        scale_factor=1 / 8,
        shuffle_nodes=False,
        mask_fixation=True,
    )

    legacy_env = DecisionTreeEnv(seed=seed, **kwargs)
    legacy_env.graph.point_set = np.array([1.0], dtype=np.float32)

    jax_env = JaxDecisionTreeEnv(
        num_nodes=kwargs["num_nodes"],
        beta_move=kwargs["beta_move"],
        eps_move=kwargs["eps_move"],
        learning_rate=kwargs["learning_rate"],
        wm_decay=kwargs["wm_decay"],
        t_max=kwargs["t_max"],
        cost=kwargs["cost"],
        scale_factor=kwargs["scale_factor"],
        shuffle_nodes=kwargs["shuffle_nodes"],
        point_set=jnp.array([1.0], dtype=jnp.float32),
    )

    key = jax.random.PRNGKey(seed)
    return legacy_env, jax_env, key


def test_reset_matches_legacy_environment():
    legacy_env, jax_env, key = _make_envs(seed=3)

    obs_legacy, info_legacy = legacy_env.reset()
    state, obs_jax, info_jax = jax_env.reset(key)

    np.testing.assert_allclose(np.asarray(obs_jax), obs_legacy, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_legacy["mask"])
    _assert_state_matches_legacy(legacy_env, state)


def test_fixation_step_matches_legacy_environment():
    legacy_env, jax_env, key = _make_envs(seed=4)

    legacy_env.reset()
    state, _, _ = jax_env.reset(key)

    action = 1
    obs_legacy, reward_legacy, done_legacy, truncated_legacy, info_legacy = legacy_env.step(action)
    state, obs_jax, reward_jax, done_jax, truncated_jax, info_jax = jax_env.step(state, action)

    np.testing.assert_allclose(np.asarray(obs_jax), obs_legacy, atol=1e-6)
    np.testing.assert_allclose(float(reward_jax), reward_legacy, atol=1e-6)
    np.testing.assert_equal(bool(done_jax), done_legacy)
    np.testing.assert_equal(bool(truncated_jax), truncated_legacy)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_legacy["mask"])

    _assert_state_matches_legacy(legacy_env, state)


def test_move_step_matches_legacy_environment():
    legacy_env, jax_env, key = _make_envs(seed=5)

    legacy_env.reset()
    state, _, _ = jax_env.reset(key)

    legacy_env.step(1)
    state, _, _, _, _, _ = jax_env.step(state, 1)

    move_action = legacy_env.num_nodes
    obs_legacy, reward_legacy, done_legacy, truncated_legacy, info_legacy = legacy_env.step(move_action)
    state, obs_jax, reward_jax, done_jax, truncated_jax, info_jax = jax_env.step(state, move_action)

    np.testing.assert_allclose(np.asarray(obs_jax), obs_legacy, atol=1e-6)
    np.testing.assert_allclose(float(reward_jax), reward_legacy, atol=1e-6)
    np.testing.assert_equal(bool(done_jax), done_legacy)
    np.testing.assert_equal(bool(truncated_jax), truncated_legacy)
    np.testing.assert_array_equal(np.asarray(info_jax["mask"]), info_legacy["mask"])

    chosen_path_jax = np.asarray(state.chosen_path)[: int(state.chosen_path_len)]
    np.testing.assert_array_equal(chosen_path_jax, np.asarray(legacy_env.chosen_path, dtype=np.int32))


def test_compiled_rollout_matches_legacy_environment():
    legacy_env, jax_env, key = _make_envs(seed=6)

    actions = jnp.array([1, legacy_env.num_nodes], dtype=jnp.int32)

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

    obs_legacy_reset, info_legacy_reset = legacy_env.reset()
    legacy_step_obs = []
    legacy_step_rewards = []
    legacy_step_dones = []
    legacy_step_truncateds = []
    legacy_step_masks = []

    for action in np.asarray(actions):
        obs, reward, done, truncated, info = legacy_env.step(int(action))
        legacy_step_obs.append(obs)
        legacy_step_rewards.append(reward)
        legacy_step_dones.append(done)
        legacy_step_truncateds.append(truncated)
        legacy_step_masks.append(info["mask"])

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

    np.testing.assert_allclose(np.asarray(obs_reset_jax), obs_legacy_reset, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(mask_reset_jax), info_legacy_reset["mask"])
    np.testing.assert_allclose(np.asarray(obses_jax), np.asarray(legacy_step_obs), atol=1e-6)
    np.testing.assert_allclose(np.asarray(rewards_jax), np.asarray(legacy_step_rewards), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(dones_jax), np.asarray(legacy_step_dones))
    np.testing.assert_array_equal(np.asarray(truncateds_jax), np.asarray(legacy_step_truncateds))
    np.testing.assert_array_equal(np.asarray(masks_jax), np.asarray(legacy_step_masks))
