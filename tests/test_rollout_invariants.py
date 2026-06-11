import pytest

from modules.config import DEFAULT_PARAMS, ENV_DYNAMIC_PARAM_KEYS
from modules.environment import JaxDecisionTreeEnv
from modules.rollout_invariants import collect_random_fixation_rollouts, assert_fixation_rollout_invariants


GENERAL_ENVIRONMENTS = [
    pytest.param("default", {}, {}, id="default"),
    pytest.param(
        "transient_wm",
        {},
        {
            "wm_decay": 0.5,
            "wm_neighbor_activation": 0.5,
            "forget_rate": 0.25,
            "q_decay": 0.75,
            "q_drift": 0.1,
        },
        id="transient_wm",
    ),
    pytest.param(
        "disable_persistence",
        {"disable_persistence": True},
        {"wm_decay": 0.5},
        id="disable_persistence",
    ),
]


def _env(**overrides):
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    return JaxDecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        shuffle_nodes=bool(params["shuffle_nodes"]),
        disable_persistence=bool(params["disable_persistence"]),
        activation_masks_actions=bool(params["activation_masks_actions"]),
        activation_gates_backup_sink=bool(params["activation_gates_backup_sink"]),
        activation_gates_backup_source=bool(params["activation_gates_backup_source"]),
        disable_corruption=bool(params["disable_corruption"]),
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
    params = {key: DEFAULT_PARAMS[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    params.update(overrides)
    return env.make_params(**params)


def _run_invariant_batch(
    *,
    env_overrides,
    param_overrides,
    num_rollouts,
    num_steps,
    seed=0,
    expect_max_consistent_q=False,
):
    env = _env(**env_overrides)
    params = _env_params(env, **param_overrides)
    trace = collect_random_fixation_rollouts(
        env,
        params,
        seed=seed,
        num_rollouts=num_rollouts,
        num_steps=num_steps,
    )
    assert_fixation_rollout_invariants(
        env,
        params,
        trace,
        expect_max_consistent_q=expect_max_consistent_q,
    )


@pytest.mark.parametrize("_, env_overrides, param_overrides", GENERAL_ENVIRONMENTS)
@pytest.mark.parametrize("num_nodes", [15])
def test_random_fixation_rollout_invariants(_, env_overrides, param_overrides, num_nodes):
    _run_invariant_batch(
        env_overrides={"num_nodes": num_nodes, "shuffle_nodes": True, **env_overrides},
        param_overrides=param_overrides,
        num_rollouts=100,
        num_steps=30,
        seed=num_nodes,
    )


@pytest.mark.parametrize("num_nodes", [15])
def test_low_noise_rollout_q_values_are_max_consistent(num_nodes):
    _run_invariant_batch(
        env_overrides={"num_nodes": num_nodes, "shuffle_nodes": True},
        param_overrides={
            "beta_move": 1000.0,
            "eps_move": 0.0,
            "learning_rate": 1.0,
            "lamda_backup": 1.0,
            "backup_steps": 100,
            "wm_decay": 1.0,
            "wm_neighbor_activation": 1.0,
            "forget_rate": 0.0,
            "q_drift": 0.0,
            "q_decay": 1.0,
        },
        num_rollouts=100,
        num_steps=20,
        seed=100 + num_nodes,
        expect_max_consistent_q=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("_, env_overrides, param_overrides", GENERAL_ENVIRONMENTS)
@pytest.mark.parametrize("num_nodes", [7, 15, 21])
def test_large_random_fixation_rollout_invariants(_, env_overrides, param_overrides, num_nodes):
    _run_invariant_batch(
        env_overrides={"num_nodes": num_nodes, "shuffle_nodes": True, **env_overrides},
        param_overrides=param_overrides,
        num_rollouts=1000,
        num_steps=30,
        seed=200 + num_nodes,
    )


@pytest.mark.slow
@pytest.mark.parametrize("num_nodes", [7, 15, 21])
def test_large_low_noise_rollout_q_values_are_max_consistent(num_nodes):
    _run_invariant_batch(
        env_overrides={"num_nodes": num_nodes, "shuffle_nodes": True},
        param_overrides={
            "beta_move": 1000.0,
            "eps_move": 0.0,
            "learning_rate": 1.0,
            "lamda_backup": 1.0,
            "backup_steps": 100,
            "wm_decay": 1.0,
            "wm_neighbor_activation": 1.0,
            "forget_rate": 0.0,
            "q_drift": 0.0,
            "q_decay": 1.0,
        },
        num_rollouts=1000,
        num_steps=20,
        seed=300 + num_nodes,
        expect_max_consistent_q=True,
    )
