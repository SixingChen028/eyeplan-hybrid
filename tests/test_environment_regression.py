import jax
import jax.numpy as jnp
import numpy as np

from modules.environment import JaxDecisionTreeEnv
from modules.environment_regression_reference import JaxDecisionTreeEnv as ReferenceJaxDecisionTreeEnv


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
        "learning_rate": 0.4,
        "lamda_backup": 0.7,
        "backup_steps": 2,
        "recency_decay": "auto",
        "wm_decay": 0.6,
    },
    {
        "num_nodes": 15,
        "t_max": 7,
        "shuffle_nodes": True,
        "eps_move": 0.1,
        "q_drop_rate": 0.2,
        "q_drift": 0.05,
        "q_decay": "auto",
        "wm_backup": True,
    },
]


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
    for config_index, config in enumerate(ENV_CONFIGS):
        live_env = JaxDecisionTreeEnv(**config)
        reference_env = ReferenceJaxDecisionTreeEnv(**config)

        for policy_index in range(10):
            base_key = jax.random.PRNGKey(10_000 * config_index + policy_index)
            reset_key, policy_key, rollout_key = jax.random.split(base_key, 3)
            policy_params = _init_policy_params(
                policy_key,
                feature_size=live_env.observation_shape[0],
                action_size=live_env.action_size,
            )

            live_state, live_obs, live_info = live_env.reset(reset_key)
            reference_state, reference_obs, reference_info = reference_env.reset(reset_key)

            np.testing.assert_allclose(np.asarray(live_obs), np.asarray(reference_obs), atol=1e-6)
            np.testing.assert_array_equal(np.asarray(live_info["mask"]), np.asarray(reference_info["mask"]))

            done = False
            step = 0
            max_steps = live_env.t_max
            while not done and step < max_steps:
                rollout_key, action_key = jax.random.split(rollout_key)
                live_action = _sample_policy_action(policy_params, live_obs, live_info["mask"], action_key)
                reference_action = _sample_policy_action(
                    policy_params,
                    reference_obs,
                    reference_info["mask"],
                    action_key,
                )
                np.testing.assert_array_equal(np.asarray(live_action), np.asarray(reference_action))

                live_state, live_obs, live_reward, live_done, live_truncated, live_info = live_env.step(
                    live_state,
                    live_action,
                )
                (
                    reference_state,
                    reference_obs,
                    reference_reward,
                    reference_done,
                    reference_truncated,
                    reference_info,
                ) = reference_env.step(reference_state, reference_action)

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
