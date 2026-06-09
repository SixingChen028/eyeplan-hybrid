from typing import Dict

import jax
import jax.numpy as jnp

from .environment import DecisionTreeObs

# When changing this file in a way that can affect results, update
# docs/changes.md. If existing checkpoint weights become incompatible,
# also bump COMPAT_VERSION.


NETWORK_MLP = "mlp"
NETWORK_NODE_SHARED = "node_shared"
NETWORK_GLOBAL_SHARED = "global_shared"
NETWORK_TYPES = (NETWORK_MLP, NETWORK_NODE_SHARED, NETWORK_GLOBAL_SHARED)


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


def flatten_observation(obs: DecisionTreeObs) -> jax.Array:
    parts = [
        obs.fixation,
        obs.fixation_point,
        obs.parent,
        obs.child,
        obs.root,
    ]
    if obs.g_values is not None:
        parts.append(obs.g_values)
    if obs.q_values is not None:
        parts.append(obs.q_values)
    if obs.n_visits is not None:
        parts.append(obs.n_visits)
    if obs.is_terminal is not None:
        parts.append(obs.is_terminal)
    if obs.recency is not None:
        parts.append(obs.recency)
    if obs.time_elapsed is not None:
        parts.append(obs.time_elapsed)
    return jnp.concatenate(parts, axis=-1)


def init_node_shared_actor_critic_params(
    key: jax.Array,
    observation_template: DecisionTreeObs,
    hidden_size: int = 128,
) -> Dict[str, Dict[str, jax.Array]]:
    node_feature_size = 5
    for feature in (
        observation_template.g_values,
        observation_template.q_values,
        observation_template.n_visits,
        observation_template.is_terminal,
        observation_template.recency,
    ):
        if feature is not None:
            node_feature_size += 1

    global_feature_size = hidden_size * 2 + 1
    if observation_template.time_elapsed is not None:
        global_feature_size += 1
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
            "w": _xavier_uniform(k4, global_feature_size, hidden_size),
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


def init_global_shared_actor_critic_params(
    key: jax.Array,
    observation_template: DecisionTreeObs,
    hidden_size: int = 128,
) -> Dict[str, Dict[str, jax.Array]]:
    params = init_node_shared_actor_critic_params(
        key,
        observation_template=observation_template,
        hidden_size=hidden_size,
    )
    k1, k2 = jax.random.split(jax.random.fold_in(key, 1), 2)
    params["node_policy_context"] = {
        "w": _xavier_uniform(k1, hidden_size * 2, hidden_size),
        "b": jnp.zeros((hidden_size,), dtype=jnp.float32),
    }
    params["node_policy"] = {
        "w": _xavier_uniform(k2, hidden_size, 1),
        "b": jnp.zeros((1,), dtype=jnp.float32),
    }
    return params


def init_actor_critic_params(
    key: jax.Array,
    observation_template: DecisionTreeObs,
    action_size: int,
    hidden_size: int = 128,
    network_type: str = NETWORK_MLP,
) -> Dict[str, Dict[str, jax.Array]]:
    if network_type == NETWORK_MLP:
        feature_size = int(flatten_observation(observation_template).shape[-1])
        return init_mlp_actor_critic_params(
            key,
            feature_size=feature_size,
            action_size=action_size,
            hidden_size=hidden_size,
        )
    if network_type == NETWORK_NODE_SHARED:
        return init_node_shared_actor_critic_params(
            key,
            observation_template=observation_template,
            hidden_size=hidden_size,
        )
    if network_type == NETWORK_GLOBAL_SHARED:
        return init_global_shared_actor_critic_params(
            key,
            observation_template=observation_template,
            hidden_size=hidden_size,
        )
    raise ValueError(f"Unknown network_type={network_type!r}. Expected one of {NETWORK_TYPES}.")


def _linear(x: jax.Array, layer: Dict[str, jax.Array]) -> jax.Array:
    return x @ layer["w"] + layer["b"]


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
    obs: DecisionTreeObs,
    action_mask: jax.Array | None = None,
    observation_mask: jax.Array | None = None,
):
    if action_mask is None:
        raise ValueError("node_shared network requires action_mask.")
    num_nodes = action_mask.shape[-1] - 1
    if observation_mask is None:
        raise ValueError("node_shared network requires observation_mask.")

    fixation = obs.fixation
    fixation_point = obs.fixation_point
    parent = obs.parent
    child = obs.child
    root = obs.root
    g_values = obs.g_values
    q_values = obs.q_values
    n_visits = obs.n_visits
    is_terminal = obs.is_terminal
    recency = obs.recency
    time_elapsed = obs.time_elapsed

    observable_nodes = observation_mask[..., :num_nodes]
    observation_feature = observable_nodes.astype(fixation.dtype)

    parts = [
        fixation,
        parent,
        child,
        root,
    ]
    if g_values is not None:
        parts.append(g_values)
    if q_values is not None:
        parts.append(q_values)
    if n_visits is not None:
        parts.append(n_visits)
    if is_terminal is not None:
        parts.append(is_terminal)
    if recency is not None:
        parts.append(recency)
    parts.append(observation_feature)
    node_features = jnp.stack(parts, axis=-1)

    h1 = jax.nn.relu(_linear(node_features, params["node_fc1"]))
    node_embeddings = jax.nn.relu(_linear(h1, params["node_fc2"]))

    legal_mean = _masked_mean(node_embeddings, observable_nodes)
    legal_max = _masked_max(node_embeddings, observable_nodes)

    global_parts = [legal_mean, legal_max, fixation_point]
    if time_elapsed is not None:
        global_parts.append(time_elapsed)
    global_features = jnp.concatenate(global_parts, axis=-1)
    global_hidden = jax.nn.relu(_linear(global_features, params["global_fc"]))
    if "node_policy_context" in params:
        global_context = jnp.broadcast_to(global_hidden[..., None, :], node_embeddings.shape)
        policy_inputs = jnp.concatenate([node_embeddings, global_context], axis=-1)
        policy_hidden = jax.nn.relu(_linear(policy_inputs, params["node_policy_context"]))
        node_logits = _linear(policy_hidden, params["node_policy"]).squeeze(-1)
    else:
        node_logits = _linear(node_embeddings, params["node_policy"]).squeeze(-1)
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
    obs: DecisionTreeObs | jax.Array,
    action_mask: jax.Array | None = None,
    observation_mask: jax.Array | None = None,
):
    if "node_fc1" in params:
        return _node_shared_forward(params, obs, action_mask, observation_mask)
    if isinstance(obs, DecisionTreeObs):
        obs = flatten_observation(obs)
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
