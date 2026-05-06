import jax
import jax.numpy as jnp
import numpy as np

from modules.config_defaults import load_canonical_defaults
from modules.environment import JaxDecisionTreeEnv, make_decision_tree_params
from modules.environment_regression_reference import JaxDecisionTreeEnv as ReferenceJaxDecisionTreeEnv

_, _DEFAULT_PARAMS = load_canonical_defaults()

ENV_CONFIGS = [
    {
        "num_nodes": 7,
        "t_max": 6,
        "shuffle_nodes": False,
        "point_set": [-2.0, -1.0, 1.0, 2.0],
    },
    {
        "num_nodes": 7,
        "t_max": 6,
        "shuffle_nodes": True,
        "use_recency_obs": True,
        "learning_rate": 0.4,
        "lamda_backup": 0.7,
        "backup_steps": 2,
        "recency_decay": 0.6,
        "wm_decay": 0.6,
    },
    {
        "num_nodes": 15,
        "t_max": 7,
        "shuffle_nodes": True,
        "eps_move": 0.1,
        "q_drop_rate": 0.2,
        "q_drift": 0.05,
        "q_decay": 0.95,
        "wm_backup": True,
    },
    {
        "num_nodes": 15,
        "t_max": 30,
        "shuffle_nodes": True,
        "beta_move": 0.4,
        "eps_move": 0.03,
        "learning_rate": 0.7,
        "lamda_backup": 0.4,
        "backup_steps": 2,
        "wm_decay": 0.8,
        "q_drop_rate": 0.2,
        "q_drift": 0.02,
        "q_decay": 0.8,
        "recency_decay": 0.7,
        "cost": 0.02,
        "wm_backup": True,
    },
]


ENV_INIT_KEYS = {"num_nodes", "t_max", "scale_factor", "shuffle_nodes", "use_recency_obs", "point_set"}
PARAM_KEYS = {
    "beta_move",
    "eps_move",
    "learning_rate",
    "lamda_backup",
    "backup_steps",
    "wm_decay",
    "q_drop_rate",
    "q_drift",
    "q_decay",
    "recency_decay",
    "cost",
    "wm_backup",
}


def _live_env_from_config(config):
    env_kwargs = {key: value for key, value in config.items() if key in ENV_INIT_KEYS}
    if "scale_factor" not in env_kwargs:
        env_kwargs["scale_factor"] = _DEFAULT_PARAMS["scale_factor"]
    # The frozen reference currently ignores shuffle_nodes=False at init time.
    env_kwargs["shuffle_nodes"] = True
    return JaxDecisionTreeEnv(**env_kwargs)


def _live_params_from_config(env, config):
    params = {
        "beta_move": 4.0,
        "eps_move": 0.02,
        "learning_rate": 0.2,
        "lamda_backup": 0.0,
        "backup_steps": 100,
        "wm_decay": 0.8,
        "q_drop_rate": 0.0,
        "q_drift": 0.0,
        "q_decay": 0.0,
        "recency_decay": 0.0,
        "cost": 0.01,
        "wm_backup": False,
    }
    params.update({key: value for key, value in config.items() if key in PARAM_KEYS})
    return make_decision_tree_params(env, **params)


def _assert_public_outputs_match(live, reference):
    live_obs, live_reward, live_done, live_truncated, live_mask = live
    reference_obs, reference_reward, reference_done, reference_truncated, reference_mask = reference

    np.testing.assert_allclose(np.asarray(live_obs), np.asarray(reference_obs), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(live_mask), np.asarray(reference_mask))
    np.testing.assert_allclose(float(live_reward), float(reference_reward), atol=1e-6)
    assert bool(live_done) == bool(reference_done)
    assert bool(live_truncated) == bool(reference_truncated)


def _init_policy_params(key, feature_size, action_size):
    weight_key, bias_key = jax.random.split(key)
    return {
        "w": jax.random.normal(weight_key, (feature_size, action_size), dtype=jnp.float32),
        "b": jax.random.normal(bias_key, (action_size,), dtype=jnp.float32),
    }


def _sample_policy_action(params, obs, mask, key):
    logits = obs @ params["w"] + params["b"]
    masked_logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    return jax.random.categorical(key, masked_logits)


def test_live_environment_matches_frozen_reference_for_random_policy_rollouts():
    sample_policy_action_jit = jax.jit(_sample_policy_action)

    for config_index, config in enumerate(ENV_CONFIGS):
        live_env = _live_env_from_config(config)
        live_params = _live_params_from_config(live_env, config)
        reference_env = ReferenceJaxDecisionTreeEnv(**config)
        live_reset_jit = jax.jit(live_env.reset_with_params)
        reference_reset_jit = jax.jit(reference_env.reset)
        live_step_jit = jax.jit(live_env.step_with_params)
        reference_step_jit = jax.jit(reference_env.step)

        for policy_index in range(10):
            base_key = jax.random.PRNGKey(10_000 * config_index + policy_index)
            reset_key, policy_key, rollout_key = jax.random.split(base_key, 3)
            policy_params = _init_policy_params(
                policy_key,
                feature_size=live_env.observation_shape[0],
                action_size=live_env.action_size,
            )

            live_state, live_obs, live_info = live_reset_jit(reset_key, live_params)
            reference_state, reference_obs, reference_info = reference_reset_jit(reset_key)

            np.testing.assert_allclose(np.asarray(live_obs), np.asarray(reference_obs), atol=1e-6)
            np.testing.assert_array_equal(np.asarray(live_info["mask"]), np.asarray(reference_info["mask"]))

            done = False
            step = 0
            max_steps = live_env.t_max
            while not done and step < max_steps:
                rollout_key, action_key = jax.random.split(rollout_key)
                live_action = sample_policy_action_jit(policy_params, live_obs, live_info["mask"], action_key)
                reference_action = sample_policy_action_jit(
                    policy_params,
                    reference_obs,
                    reference_info["mask"],
                    action_key,
                )
                np.testing.assert_array_equal(np.asarray(live_action), np.asarray(reference_action))

                live_state, live_obs, live_reward, live_done, live_truncated, live_info = live_step_jit(
                    live_state,
                    live_action,
                    live_params,
                )
                (
                    reference_state,
                    reference_obs,
                    reference_reward,
                    reference_done,
                    reference_truncated,
                    reference_info,
                ) = reference_step_jit(reference_state, reference_action)

                _assert_public_outputs_match(
                    (live_obs, live_reward, live_done, live_truncated, live_info["mask"]),
                    (
                        reference_obs,
                        reference_reward,
                        reference_done,
                        reference_truncated,
                        reference_info["mask"],
                    ),
                )
                done = bool(live_done)
                step += 1
