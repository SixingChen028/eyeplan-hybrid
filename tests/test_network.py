import jax
import numpy as np

from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import JaxDecisionTreeEnv
from modules.network import (
    NETWORK_NODE_SHARED,
    actor_critic_forward,
    flatten_observation,
    init_actor_critic_params,
    init_mlp_actor_critic_params,
)

_, _DEFAULT_PARAMS = load_canonical_defaults()


def _env(**overrides):
    params = dict(_DEFAULT_PARAMS)
    params.update(overrides)
    return JaxDecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        shuffle_nodes=bool(params["shuffle_nodes"]),
        use_recency_obs=bool(params["use_recency_obs"]),
        use_best_open_value_obs=bool(params["use_best_open_value_obs"]),
        use_best_terminal_value_obs=bool(params["use_best_terminal_value_obs"]),
        wm_backup=bool(params["wm_backup"]),
        point_set=params["point_set"],
    )


def _env_params(env, **overrides):
    params = {key: _DEFAULT_PARAMS[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    params.update(overrides)
    return env.make_params(**params)


def _batch_obs(obs):
    return jax.tree_util.tree_map(
        lambda value: None if value is None else value[None, ...],
        obs,
    )


def _permute_node_observation(obs, permutation):
    def permute(value):
        if value is None:
            return None
        if value.shape[-1] == len(permutation):
            return value[permutation]
        return value

    return jax.tree_util.tree_map(permute, obs)


def test_mlp_forward_shape_is_unchanged():
    env = _env(num_nodes=5, shuffle_nodes=False)
    _, obs, info = env.reset(jax.random.PRNGKey(0), _env_params(env))
    params = init_mlp_actor_critic_params(
        jax.random.PRNGKey(1),
        feature_size=int(flatten_observation(env.observation_template).shape[0]),
        action_size=env.action_size,
        hidden_size=16,
    )

    logits, values = actor_critic_forward(params, _batch_obs(obs), info["mask"][None, :])

    assert logits.shape == (1, env.action_size)
    assert values.shape == (1,)


def test_node_shared_forward_shape_with_and_without_recency():
    for use_recency_obs, recency_decay in [(False, 0.0), (True, 0.5)]:
        for use_best_open_value_obs, use_best_terminal_value_obs in [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]:
            env = _env(
                num_nodes=5,
                shuffle_nodes=False,
                use_recency_obs=use_recency_obs,
                use_best_open_value_obs=use_best_open_value_obs,
                use_best_terminal_value_obs=use_best_terminal_value_obs,
            )
            _, obs, info = env.reset(jax.random.PRNGKey(0), _env_params(env, recency_decay=recency_decay))
            params = init_actor_critic_params(
                jax.random.PRNGKey(1),
                observation_template=env.observation_template,
                action_size=env.action_size,
                hidden_size=16,
                network_type=NETWORK_NODE_SHARED,
            )

            logits, values = actor_critic_forward(params, _batch_obs(obs), info["mask"][None, :])

            expected_node_features = 10 if use_recency_obs else 9
            expected_global_features = 16 * 2 + 2
            if use_best_open_value_obs:
                expected_global_features += 1
            if use_best_terminal_value_obs:
                expected_global_features += 1
            assert params["node_fc1"]["w"].shape == (expected_node_features, 16)
            assert params["global_fc"]["w"].shape == (expected_global_features, 16)
            assert logits.shape == (1, env.action_size)
            assert values.shape == (1,)
            assert np.all(np.isfinite(np.asarray(logits)))
            assert np.all(np.isfinite(np.asarray(values)))


def test_node_shared_forward_is_permutation_equivariant_for_node_logits():
    env = _env(num_nodes=5, shuffle_nodes=False, use_recency_obs=True)
    _, obs, info = env.reset(jax.random.PRNGKey(0), _env_params(env, recency_decay=0.5))
    params = init_actor_critic_params(
        jax.random.PRNGKey(1),
        observation_template=env.observation_template,
        action_size=env.action_size,
        hidden_size=16,
        network_type=NETWORK_NODE_SHARED,
    )
    permutation = np.array([2, 0, 4, 1, 3], dtype=np.int32)
    permuted_obs = _permute_node_observation(obs, permutation)
    permuted_mask = np.concatenate([np.asarray(info["mask"])[:-1][permutation], np.asarray(info["mask"])[-1:]])

    logits, values = actor_critic_forward(params, _batch_obs(obs), info["mask"][None, :])
    permuted_logits, permuted_values = actor_critic_forward(
        params,
        _batch_obs(permuted_obs),
        permuted_mask[None, :],
    )

    np.testing.assert_allclose(
        np.asarray(permuted_logits[0, :-1]),
        np.asarray(logits[0, :-1])[permutation],
        atol=1e-6,
    )
    np.testing.assert_allclose(float(permuted_logits[0, -1]), float(logits[0, -1]), atol=1e-6)
    np.testing.assert_allclose(float(permuted_values[0]), float(values[0]), atol=1e-6)
