import numpy as np
import jax

from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import DecisionTreeEnv
from modules.random_search import RANDOM_SEARCH_STOP_MAX_FIXATIONS, RandomSearchSimulator
from generate_random_search import _with_random_search_metadata

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


def test_random_search_fixation_target_accounts_for_root_fixation():
    env = _env(num_nodes=3, t_max=5, shuffle_nodes=False, point_set=np.array([1.0], dtype=np.float32))
    simulator = RandomSearchSimulator(env, _env_params(env))

    keys = jax.random.split(jax.random.PRNGKey(0), 100)
    targets = jax.vmap(simulator._sample_fixation_target)(keys)

    assert int(np.min(targets)) >= 0
    assert int(np.max(targets)) <= RANDOM_SEARCH_STOP_MAX_FIXATIONS - 1


def test_random_search_simulation_writes_existing_simulation_shape():
    env = _env(num_nodes=3, t_max=5, shuffle_nodes=False, point_set=np.array([1.0], dtype=np.float32))
    simulator = RandomSearchSimulator(env, _env_params(env))

    data = simulator.simulate(seed=1, num_trials=5, batch_size=2, skip_timeout_trials=False)

    assert set(data) == {"adj_lists", "starts", "rewards", "actions", "chosen_paths"}
    assert len(data["actions"]) == 5
    assert all(actions[0] == start for actions, start in zip(data["actions"], data["starts"]))
    assert all(actions[-1] == env.num_nodes for actions in data["actions"])


def test_random_search_metadata_suffixes_condition_label():
    run = _with_random_search_metadata({"seed": 1}, label="wm_only")

    assert run["label"] == "wm_only_random_search"
    assert run["lesion_policy"] == "random_search_gamma_stopping"


def test_random_search_metadata_uses_default_label_without_condition():
    run = _with_random_search_metadata({"seed": 1}, label=None)

    assert run["label"] == "random_search"

