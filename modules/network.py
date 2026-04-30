from typing import Dict

import jax
import jax.numpy as jnp


NETWORK_MLP = "mlp"
NETWORK_NODE_SHARED = "node_shared"
NETWORK_TYPES = (NETWORK_MLP, NETWORK_NODE_SHARED)


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


def _node_feature_size(feature_size: int, action_size: int) -> int:
    num_nodes = int(action_size) - 1
    base_size = 7 * num_nodes + 2
    recency_size = base_size + num_nodes
    if int(feature_size) == base_size:
        return 8
    if int(feature_size) == recency_size:
        return 9
    raise ValueError(
        "node_shared network requires the existing decision-tree observation layout "
        f"with feature_size={base_size} or {recency_size}; got {feature_size}."
    )


def init_node_shared_actor_critic_params(
    key: jax.Array,
    feature_size: int,
    action_size: int,
    hidden_size: int = 128,
) -> Dict[str, Dict[str, jax.Array]]:
    node_feature_size = _node_feature_size(feature_size, action_size)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    return {
        "node_fc1": {
            "w": _xavier_uniform(k1, node_feature_size, hidden_size),
            "b": jnp.zeros((hidden_size,), dtype=jnp.float32),
        },
        "node_fc2": {
            "w": _xavier_uniform(k2, hidden_size, hidden_size),
            "b": jnp.zeros((hidden_size,), dtype=jnp.float32),
        },
        "node_policy": {
            "w": _xavier_uniform(k3, hidden_size, 1),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
        "global_fc": {
            "w": _xavier_uniform(k4, hidden_size * 2 + 2, hidden_size),
            "b": jnp.zeros((hidden_size,), dtype=jnp.float32),
        },
        "terminate": {
            "w": _xavier_uniform(k5, hidden_size, 1),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
        "value": {
            "w": _xavier_uniform(k6, hidden_size, 1),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
    }


def init_actor_critic_params(
    key: jax.Array,
    feature_size: int,
    action_size: int,
    hidden_size: int = 128,
    network_type: str = NETWORK_MLP,
) -> Dict[str, Dict[str, jax.Array]]:
    if network_type == NETWORK_MLP:
        return init_mlp_actor_critic_params(
            key,
            feature_size=feature_size,
            action_size=action_size,
            hidden_size=hidden_size,
        )
    if network_type == NETWORK_NODE_SHARED:
        return init_node_shared_actor_critic_params(
            key,
            feature_size=feature_size,
            action_size=action_size,
            hidden_size=hidden_size,
        )
    raise ValueError(f"Unknown network_type={network_type!r}. Expected one of {NETWORK_TYPES}.")


def _linear(x: jax.Array, layer: Dict[str, jax.Array]) -> jax.Array:
    return x @ layer["w"] + layer["b"]


def _split_node_observation(obs: jax.Array, num_nodes: int):
    index = 0
    fixation = obs[..., index : index + num_nodes]
    index += num_nodes
    fixation_point = obs[..., index : index + 1]
    index += 1
    parent = obs[..., index : index + num_nodes]
    index += num_nodes
    child = obs[..., index : index + num_nodes]
    index += num_nodes
    root = obs[..., index : index + num_nodes]
    index += num_nodes
    g_values = obs[..., index : index + num_nodes]
    index += num_nodes
    q_values = obs[..., index : index + num_nodes]
    index += num_nodes
    n_visits = obs[..., index : index + num_nodes]
    index += num_nodes

    maybe_recency_size = obs.shape[-1] - index - 1
    if maybe_recency_size == num_nodes:
        recency = obs[..., index : index + num_nodes]
        index += num_nodes
    else:
        recency = None

    time_elapsed = obs[..., index : index + 1]
    return fixation, fixation_point, parent, child, root, g_values, q_values, n_visits, recency, time_elapsed


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(values.dtype)[..., None]
    total = jnp.sum(values * weights, axis=-2)
    count = jnp.maximum(jnp.sum(weights, axis=-2), 1.0)
    return total / count


def _masked_max(values: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(jnp.bool_)[..., None]
    has_any = jnp.any(mask, axis=-1, keepdims=True)
    fill = jnp.finfo(values.dtype).min
    pooled = jnp.max(jnp.where(weights, values, fill), axis=-2)
    return jnp.where(has_any, pooled, jnp.zeros_like(pooled))


def _node_shared_forward(
    params: Dict[str, Dict[str, jax.Array]],
    obs: jax.Array,
    action_mask: jax.Array | None = None,
):
    if action_mask is None:
        raise ValueError("node_shared network requires action_mask.")
    num_nodes = action_mask.shape[-1] - 1

    (
        fixation,
        fixation_point,
        parent,
        child,
        root,
        g_values,
        q_values,
        n_visits,
        recency,
        time_elapsed,
    ) = _split_node_observation(obs, num_nodes)

    legal_nodes = action_mask[..., :num_nodes]
    legal_feature = legal_nodes.astype(obs.dtype)

    parts = [
        fixation,
        parent,
        child,
        root,
        g_values,
        q_values,
        n_visits,
    ]
    if recency is not None:
        parts.append(recency)
    parts.append(legal_feature)
    node_features = jnp.stack(parts, axis=-1)

    h1 = jax.nn.relu(_linear(node_features, params["node_fc1"]))
    node_embeddings = jax.nn.relu(_linear(h1, params["node_fc2"]))
    node_logits = _linear(node_embeddings, params["node_policy"]).squeeze(-1)

    legal_mean = _masked_mean(node_embeddings, legal_nodes)
    legal_max = _masked_max(node_embeddings, legal_nodes)

    global_features = jnp.concatenate(
        [legal_mean, legal_max, fixation_point, time_elapsed],
        axis=-1,
    )
    global_hidden = jax.nn.relu(_linear(global_features, params["global_fc"]))
    terminate_logit = _linear(global_hidden, params["terminate"]).squeeze(-1)
    value = _linear(global_hidden, params["value"]).squeeze(-1)
    logits = jnp.concatenate([node_logits, terminate_logit[..., None]], axis=-1)
    return logits, value


def _mlp_forward(params: Dict[str, Dict[str, jax.Array]], obs: jax.Array):
    h1 = jax.nn.relu(obs @ params["fc1"]["w"] + params["fc1"]["b"])
    h2 = jax.nn.relu(h1 @ params["fc2"]["w"] + params["fc2"]["b"])

    logits = h2 @ params["policy"]["w"] + params["policy"]["b"]
    value = (h2 @ params["value"]["w"] + params["value"]["b"]).squeeze(-1)

    return logits, value


def actor_critic_forward(
    params: Dict[str, Dict[str, jax.Array]],
    obs: jax.Array,
    action_mask: jax.Array | None = None,
):
    if "node_fc1" in params:
        return _node_shared_forward(params, obs, action_mask)
    return _mlp_forward(params, obs)


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
