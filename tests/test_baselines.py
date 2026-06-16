import jax
import numpy as np
import pytest

from modules.baselines import evaluate_baseline_policies
from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import DecisionTreeEnv

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


@pytest.mark.slow
def test_baseline_policy_evaluation_runs():
    env = _env(
        num_nodes=3,
        t_max=10,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )
    env_params = _env_params(env, beta_move=4.0, eps_move=0.02, learning_rate=1.0, wm_decay=1.0, cost=0.01)

    reset_keys = jax.random.split(jax.random.PRNGKey(0), 12)

    stats, optimal_scaled, optimal_raw = evaluate_baseline_policies(
        env=env,
        env_params=env_params,
        policy_names=[
            "depth1_then_terminate",
            "visit_all_then_bestg_then_parent_chain",
        ],
        reset_keys=reset_keys,
    )

    assert len(stats) == 2
    assert np.isfinite(optimal_scaled)
    assert np.isfinite(optimal_raw)

    for item in stats:
        assert np.isfinite(item.mean_episode_reward)
        assert np.isfinite(item.mean_no_cost_reward_scaled)
        assert np.isfinite(item.mean_no_cost_reward_raw)
        assert item.mean_episode_length > 0


@pytest.mark.slow
def test_visit_all_policy_has_bounded_length_after_fix():
    env = _env(
        num_nodes=7,
        t_max=100,
        scale_factor=1 / 8,
        shuffle_nodes=True,
    )
    env_params = _env_params(env, beta_move=100.0, eps_move=0.0, learning_rate=1.0, wm_decay=1.0, cost=0.01)

    reset_keys = jax.random.split(jax.random.PRNGKey(11), 512)
    stats, _, _ = evaluate_baseline_policies(
        env=env,
        env_params=env_params,
        policy_names=["visit_all_then_bestg_then_parent_chain"],
        reset_keys=reset_keys,
    )

    assert len(stats) == 1
    # For 7-node trees the policy should terminate quickly, not drift toward t_max.
    assert stats[0].mean_episode_length <= 14.0
