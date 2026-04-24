from typing import Dict

import jax
import jax.numpy as jnp


def _xavier_uniform(key: jax.Array, fan_in: int, fan_out: int) -> jax.Array:
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-limit, maxval=limit)


def init_mlp_actor_critic_params(
    key: jax.Array,
    feature_size: int,
    action_size: int,
    hidden_size: int = 128,
) -> Dict[str, Dict[str, jax.Array]]:
    k1, k2, k3, k4 = jax.random.split(key, 4)

    params = {
        "fc1": {
            "w": _xavier_uniform(k1, feature_size, hidden_size),
            "b": jnp.zeros((hidden_size,), dtype=jnp.float32),
        },
        "fc2": {
            "w": _xavier_uniform(k2, hidden_size, hidden_size),
            "b": jnp.zeros((hidden_size,), dtype=jnp.float32),
        },
        "policy": {
            "w": _xavier_uniform(k3, hidden_size, action_size),
            "b": jnp.zeros((action_size,), dtype=jnp.float32),
        },
        "value": {
            "w": _xavier_uniform(k4, hidden_size, 1),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
    }

    return params


def actor_critic_forward(params: Dict[str, Dict[str, jax.Array]], obs: jax.Array):
    h1 = jax.nn.relu(obs @ params["fc1"]["w"] + params["fc1"]["b"])
    h2 = jax.nn.relu(h1 @ params["fc2"]["w"] + params["fc2"]["b"])

    logits = h2 @ params["policy"]["w"] + params["policy"]["b"]
    value = (h2 @ params["value"]["w"] + params["value"]["b"]).squeeze(-1)

    return logits, value


def apply_action_mask(logits: jax.Array, action_mask: jax.Array) -> jax.Array:
    mask_value = jnp.finfo(logits.dtype).min
    return jnp.where(action_mask, logits, mask_value)


def sample_actions(
    key: jax.Array,
    logits: jax.Array,
    action_mask: jax.Array,
):
    masked_logits = apply_action_mask(logits, action_mask)
    actions = jax.random.categorical(key, masked_logits, axis=-1)

    log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
    probs_all = jax.nn.softmax(masked_logits, axis=-1)

    batch_idx = jnp.arange(masked_logits.shape[0])
    log_probs = log_probs_all[batch_idx, actions]

    entropy = -jnp.sum(
        jnp.where(action_mask, probs_all * log_probs_all, 0.0),
        axis=-1,
    )

    return actions, log_probs, entropy


def greedy_actions(logits: jax.Array, action_mask: jax.Array) -> jax.Array:
    masked_logits = apply_action_mask(logits, action_mask)
    return jnp.argmax(masked_logits, axis=-1)
