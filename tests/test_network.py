import jax
import numpy as np

from modules.environment import JaxDecisionTreeEnv
from modules.network import (
    NETWORK_NODE_SHARED,
    actor_critic_forward,
    init_actor_critic_params,
    init_mlp_actor_critic_params,
)


def _permute_node_observation(obs, permutation, has_recency=False):
    num_nodes = len(permutation)
    index = 0
    parts = []

    for width in [
        num_nodes,
        1,
        num_nodes,
        num_nodes,
        num_nodes,
        num_nodes,
        num_nodes,
        num_nodes,
        num_nodes,
        1,
        1,
    ]:
        part = obs[index : index + width]
        if width == num_nodes:
            part = part[permutation]
        parts.append(part)
        index += width

    if has_recency:
        parts.append(obs[index : index + num_nodes][permutation])
        index += num_nodes

    parts.append(obs[index : index + 1])
    return np.concatenate(parts)


def test_mlp_forward_shape_is_unchanged():
    env = JaxDecisionTreeEnv(num_nodes=5, shuffle_nodes=False)
    _, obs, info = env.reset_with_params(jax.random.PRNGKey(0), env.params())
    params = init_mlp_actor_critic_params(
        jax.random.PRNGKey(1),
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
    )

    logits, values = actor_critic_forward(params, obs[None, :], info["mask"][None, :])

    assert logits.shape == (1, env.action_size)
    assert values.shape == (1,)


def test_node_shared_forward_shape_with_and_without_recency():
    for recency_decay in ["off", 0.5]:
        env = JaxDecisionTreeEnv(num_nodes=5, shuffle_nodes=False, recency_decay=recency_decay)
        _, obs, info = env.reset_with_params(jax.random.PRNGKey(0), env.params())
        params = init_actor_critic_params(
            jax.random.PRNGKey(1),
            feature_size=env.observation_shape[0],
            action_size=env.action_size,
            hidden_size=16,
            network_type=NETWORK_NODE_SHARED,
        )

        logits, values = actor_critic_forward(params, obs[None, :], info["mask"][None, :])

        assert logits.shape == (1, env.action_size)
        assert values.shape == (1,)
        assert np.all(np.isfinite(np.asarray(logits)))
        assert np.all(np.isfinite(np.asarray(values)))


def test_node_shared_forward_is_permutation_equivariant_for_node_logits():
    env = JaxDecisionTreeEnv(num_nodes=5, shuffle_nodes=False, recency_decay=0.5)
    _, obs, info = env.reset_with_params(jax.random.PRNGKey(0), env.params())
    params = init_actor_critic_params(
        jax.random.PRNGKey(1),
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        network_type=NETWORK_NODE_SHARED,
    )
    permutation = np.array([2, 0, 4, 1, 3], dtype=np.int32)
    permuted_obs = _permute_node_observation(np.asarray(obs), permutation, has_recency=True)
    permuted_mask = np.concatenate([np.asarray(info["mask"])[:-1][permutation], np.asarray(info["mask"])[-1:]])

    logits, values = actor_critic_forward(params, obs[None, :], info["mask"][None, :])
    permuted_logits, permuted_values = actor_critic_forward(
        params,
        permuted_obs[None, :],
        permuted_mask[None, :],
    )

    np.testing.assert_allclose(
        np.asarray(permuted_logits[0, :-1]),
        np.asarray(logits[0, :-1])[permutation],
        atol=1e-6,
    )
    np.testing.assert_allclose(float(permuted_logits[0, -1]), float(logits[0, -1]), atol=1e-6)
    np.testing.assert_allclose(float(permuted_values[0]), float(values[0]), atol=1e-6)
