import jax
import jax.numpy as jnp
import numpy as np

from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import DecisionTreeEnv
from modules.mcts_search import MCTSSimulator

_, _DEFAULT_PARAMS = load_canonical_defaults()


def _env(**overrides):
    params = dict(_DEFAULT_PARAMS)
    params.update(overrides)
    return DecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        shuffle_nodes=bool(params["shuffle_nodes"]),
        disable_persistence=bool(params["disable_persistence"]),
        activation_masks_actions=bool(params["activation_masks_actions"]),
        activation_gates_backup_sink=bool(params["activation_gates_backup_sink"]),
        activation_gates_backup_source=bool(params["activation_gates_backup_source"]),
        disable_corruption=bool(params["disable_corruption"]),
        activation_prevents_corruption=bool(params["activation_prevents_corruption"]),
        forget_discovered=bool(params["forget_discovered"]),
        activation_masks_observation=bool(params["activation_masks_observation"]),
        excluded_child_value=params["excluded_child_value"],
        use_recency_obs=bool(params["use_recency_obs"]),
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


def test_mcts_evaluation_scales_c_by_environment_reward_scale():
    env = _env(num_nodes=3, t_max=5, scale_factor=0.25, shuffle_nodes=False)
    simulator = MCTSSimulator(env, _env_params(env), num_rollouts=2)

    evaluation = simulator.evaluate(c_raw=4.0, seed=1, num_trials=3)

    assert evaluation.c_raw == 4.0
    assert evaluation.c_scaled == 1.0
    assert evaluation.num_trials == 3
    assert np.isfinite(evaluation.mean_episode_reward)


def test_mcts_simulation_writes_existing_simulation_shape():
    env = _env(num_nodes=3, t_max=5, shuffle_nodes=False, point_set=np.array([1.0], dtype=np.float32))
    simulator = MCTSSimulator(env, _env_params(env), num_rollouts=2)

    data = simulator.simulate(c_raw=1.0, seed=1, num_trials=5, skip_timeout_trials=False)

    assert set(data) == {"adj_lists", "starts", "rewards", "actions", "chosen_paths"}
    assert len(data["actions"]) == 5
    assert all(actions[0] == start for actions, start in zip(data["actions"], data["starts"]))
    assert all(actions[-1] == env.num_nodes for actions in data["actions"])


def test_mcts_path_planning_can_choose_high_value_branch():
    env = _env(num_nodes=3, t_max=5, scale_factor=1.0, shuffle_nodes=False)
    simulator = MCTSSimulator(env, _env_params(env), num_rollouts=16)
    state, _, _ = env.reset(jax.random.PRNGKey(0), _env_params(env))
    root = int(state.root_node)
    left, right = [int(child) for child in np.asarray(state.child_nodes[root])]
    points = np.asarray(state.points).copy()
    points[left] = -8.0
    points[right] = 8.0
    state = state._replace(points=jnp.asarray(points))

    planned_path = simulator._plan_path(state, c_raw=1.0, rng=np.random.default_rng(0))

    assert planned_path == [right]
