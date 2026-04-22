from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .jax_environment import JaxDecisionTreeEnv
from .jax_network import actor_critic_forward, apply_action_mask


@dataclass
class PolicyStats:
    name: str
    mean_episode_reward: float
    mean_no_cost_reward_scaled: float
    mean_no_cost_reward_raw: float
    mean_episode_length: float


@dataclass
class ObsView:
    fixation_node: int | None
    parent_node: int | None
    child1_node: int | None
    child2_node: int | None
    root_node: int | None
    g_values: np.ndarray


class ObsLayout:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        index = 0

        self.fixation = slice(index, index + num_nodes)
        index += num_nodes

        self.point = slice(index, index + 1)
        index += 1

        self.parent = slice(index, index + num_nodes)
        index += num_nodes

        self.child1 = slice(index, index + num_nodes)
        index += num_nodes

        self.child2 = slice(index, index + num_nodes)
        index += num_nodes

        self.root = slice(index, index + num_nodes)
        index += num_nodes

        self.g = slice(index, index + num_nodes)
        index += num_nodes

        self.q = slice(index, index + num_nodes)
        index += num_nodes

        self.visits = slice(index, index + num_nodes)
        index += num_nodes

        self.time = slice(index, index + 1)


def _decode_one_hot(one_hot: np.ndarray) -> int | None:
    if np.allclose(one_hot, 0.0):
        return None
    return int(np.argmax(one_hot))


def parse_obs(obs: np.ndarray, layout: ObsLayout) -> ObsView:
    return ObsView(
        fixation_node=_decode_one_hot(obs[layout.fixation]),
        parent_node=_decode_one_hot(obs[layout.parent]),
        child1_node=_decode_one_hot(obs[layout.child1]),
        child2_node=_decode_one_hot(obs[layout.child2]),
        root_node=_decode_one_hot(obs[layout.root]),
        g_values=obs[layout.g],
    )


def _choose_legal_fixation(
    preferred: int | None,
    action_mask: np.ndarray,
    root_node: int | None,
    num_nodes: int,
) -> int:
    if preferred is not None and 0 <= preferred < num_nodes and action_mask[preferred]:
        return int(preferred)

    if root_node is not None and action_mask[root_node]:
        return int(root_node)

    legal = np.where(action_mask[:num_nodes])[0]
    if legal.size > 0:
        return int(legal[0])

    return num_nodes


def _optimal_path_reward_raw(child_nodes: np.ndarray, points: np.ndarray, root: int) -> float:
    def dfs(node: int) -> float:
        left = int(child_nodes[node, 0])
        if left < 0:
            return float(points[node])

        right = int(child_nodes[node, 1])
        return float(points[node]) + max(dfs(left), dfs(right))

    return float(dfs(root))


def _episode_no_cost_rewards(state: Any, scale_factor: float) -> Tuple[float, float]:
    chosen_len = int(state.chosen_path_len)
    if chosen_len <= 0:
        return 0.0, 0.0

    chosen_path = np.asarray(state.chosen_path[:chosen_len], dtype=np.int32)
    points = np.asarray(state.points, dtype=np.float32)
    raw_reward = float(points[chosen_path].sum())
    scaled_reward = raw_reward * float(scale_factor)

    return scaled_reward, raw_reward


def _depth1_then_terminate_action(
    obs_view: ObsView,
    action_mask: np.ndarray,
    policy_state: Dict[str, Any],
    num_nodes: int,
) -> Tuple[int, Dict[str, Any]]:
    step_idx = int(policy_state["step_idx"])

    if step_idx == 0:
        action = _choose_legal_fixation(
            policy_state["depth1_nodes"][0],
            action_mask,
            obs_view.root_node,
            num_nodes,
        )
    elif step_idx == 1:
        action = _choose_legal_fixation(
            policy_state["depth1_nodes"][1],
            action_mask,
            obs_view.root_node,
            num_nodes,
        )
    else:
        action = num_nodes

    policy_state["step_idx"] = step_idx + 1
    return action, policy_state


def _visit_all_then_bestg_then_parent_chain_action(
    obs_view: ObsView,
    action_mask: np.ndarray,
    policy_state: Dict[str, Any],
    num_nodes: int,
) -> Tuple[int, Dict[str, Any]]:
    phase = policy_state["phase"]

    if phase == "visit_all":
        visited = policy_state["visited"]
        legal = np.where(action_mask[:num_nodes])[0]
        unvisited_legal = [idx for idx in legal if not visited[idx]]

        if len(unvisited_legal) > 0:
            action = int(unvisited_legal[0])
        else:
            action = _choose_legal_fixation(None, action_mask, obs_view.root_node, num_nodes)

        if action < num_nodes:
            visited[action] = True

        if visited.all():
            policy_state["phase"] = "best_g"

        return action, policy_state

    if phase == "best_g":
        g_values = obs_view.g_values
        target = int(np.argmax(g_values))
        action = _choose_legal_fixation(target, action_mask, obs_view.root_node, num_nodes)

        policy_state["phase"] = "climb"
        return action, policy_state

    parent = obs_view.parent_node
    if parent is None:
        return num_nodes, policy_state

    action = _choose_legal_fixation(parent, action_mask, obs_view.root_node, num_nodes)
    return action, policy_state


def _immediate_move_action(
    obs_view: ObsView,
    action_mask: np.ndarray,
    policy_state: Dict[str, Any],
    num_nodes: int,
) -> Tuple[int, Dict[str, Any]]:
    del obs_view, action_mask, policy_state
    return num_nodes, {}


def _best_depth1_then_move_action(
    obs_view: ObsView,
    action_mask: np.ndarray,
    policy_state: Dict[str, Any],
    num_nodes: int,
) -> Tuple[int, Dict[str, Any]]:
    step_idx = int(policy_state["step_idx"])

    if step_idx == 0:
        action = _choose_legal_fixation(
            policy_state["depth1_nodes"][0],
            action_mask,
            obs_view.root_node,
            num_nodes,
        )
    elif step_idx == 1:
        action = _choose_legal_fixation(
            policy_state["depth1_nodes"][1],
            action_mask,
            obs_view.root_node,
            num_nodes,
        )
    elif step_idx == 2:
        depth1_nodes = [node for node in policy_state["depth1_nodes"] if node is not None]
        if len(depth1_nodes) == 0:
            action = _choose_legal_fixation(None, action_mask, obs_view.root_node, num_nodes)
        else:
            g_values = obs_view.g_values
            best_node = max(depth1_nodes, key=lambda idx: g_values[idx])
            action = _choose_legal_fixation(best_node, action_mask, obs_view.root_node, num_nodes)
    else:
        action = num_nodes

    policy_state["step_idx"] = step_idx + 1
    return action, policy_state


def _init_policy_state(policy_name: str, obs_view: ObsView, num_nodes: int) -> Dict[str, Any]:
    if policy_name in {"depth1_then_terminate", "best_depth1_then_move"}:
        return {
            "step_idx": 0,
            "depth1_nodes": [obs_view.child1_node, obs_view.child2_node],
        }

    if policy_name == "visit_all_then_bestg_then_parent_chain":
        return {
            "phase": "visit_all",
            "visited": np.zeros((num_nodes,), dtype=bool),
        }

    return {}


def _get_policy_action(
    policy_name: str,
    obs_view: ObsView,
    action_mask: np.ndarray,
    policy_state: Dict[str, Any],
    num_nodes: int,
) -> Tuple[int, Dict[str, Any]]:
    if policy_name == "depth1_then_terminate":
        return _depth1_then_terminate_action(obs_view, action_mask, policy_state, num_nodes)

    if policy_name == "visit_all_then_bestg_then_parent_chain":
        return _visit_all_then_bestg_then_parent_chain_action(obs_view, action_mask, policy_state, num_nodes)

    if policy_name == "immediate_move":
        return _immediate_move_action(obs_view, action_mask, policy_state, num_nodes)

    if policy_name == "best_depth1_then_move":
        return _best_depth1_then_move_action(obs_view, action_mask, policy_state, num_nodes)

    raise ValueError(f"Unknown baseline policy: {policy_name}")


def evaluate_baseline_policies(
    env: JaxDecisionTreeEnv,
    policy_names: List[str],
    reset_keys: jax.Array,
) -> Tuple[List[PolicyStats], float, float]:
    layout = ObsLayout(env.num_nodes)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    policy_returns = {name: [] for name in policy_names}
    policy_no_cost_scaled = {name: [] for name in policy_names}
    policy_no_cost_raw = {name: [] for name in policy_names}
    policy_lengths = {name: [] for name in policy_names}

    optimal_raw_rewards = []

    for key in reset_keys:
        initial_state, _, _ = reset_fn(key)
        child_nodes = np.asarray(initial_state.child_nodes)
        points = np.asarray(initial_state.points)
        root = int(initial_state.root_node)
        optimal_raw = _optimal_path_reward_raw(child_nodes, points, root)
        optimal_raw_rewards.append(optimal_raw)

        for policy_name in policy_names:
            state, obs, info = reset_fn(key)
            obs_np = np.asarray(obs)
            action_mask_np = np.asarray(info["mask"])

            obs_view = parse_obs(obs_np, layout)
            policy_state = _init_policy_state(policy_name, obs_view, env.num_nodes)

            done = False
            episode_reward = 0.0
            steps = 0

            while (not done) and (steps < env.t_max):
                obs_view = parse_obs(obs_np, layout)
                action, policy_state = _get_policy_action(
                    policy_name,
                    obs_view,
                    action_mask_np,
                    policy_state,
                    env.num_nodes,
                )

                state, obs, reward, done, _, info = step_fn(state, int(action))

                obs_np = np.asarray(obs)
                action_mask_np = np.asarray(info["mask"])
                episode_reward += float(reward)
                steps += 1

            scaled_no_cost, raw_no_cost = _episode_no_cost_rewards(state, env.scale_factor)

            policy_returns[policy_name].append(episode_reward)
            policy_no_cost_scaled[policy_name].append(scaled_no_cost)
            policy_no_cost_raw[policy_name].append(raw_no_cost)
            policy_lengths[policy_name].append(steps)

    stats = []
    for policy_name in policy_names:
        stats.append(
            PolicyStats(
                name=policy_name,
                mean_episode_reward=float(np.mean(policy_returns[policy_name])),
                mean_no_cost_reward_scaled=float(np.mean(policy_no_cost_scaled[policy_name])),
                mean_no_cost_reward_raw=float(np.mean(policy_no_cost_raw[policy_name])),
                mean_episode_length=float(np.mean(policy_lengths[policy_name])),
            )
        )

    optimal_mean_raw = float(np.mean(optimal_raw_rewards))
    optimal_mean_scaled = optimal_mean_raw * float(env.scale_factor)

    return stats, optimal_mean_scaled, optimal_mean_raw


def evaluate_network_greedy(
    env: JaxDecisionTreeEnv,
    params: Any,
    reset_keys: jax.Array,
) -> PolicyStats:
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    forward_fn = jax.jit(actor_critic_forward)

    rewards = []
    no_cost_scaled = []
    no_cost_raw = []
    lengths = []

    for key in reset_keys:
        state, obs, info = reset_fn(key)

        done = False
        episode_reward = 0.0
        steps = 0

        while (not done) and (steps < env.t_max):
            logits, _ = forward_fn(params, obs[None, :])
            masked_logits = apply_action_mask(logits[0], info["mask"])
            action = int(jnp.argmax(masked_logits))

            state, obs, reward, done, _, info = step_fn(state, action)
            episode_reward += float(reward)
            steps += 1

        scaled_no_cost, raw_no_cost = _episode_no_cost_rewards(state, env.scale_factor)

        rewards.append(episode_reward)
        no_cost_scaled.append(scaled_no_cost)
        no_cost_raw.append(raw_no_cost)
        lengths.append(steps)

    return PolicyStats(
        name="network_greedy_final",
        mean_episode_reward=float(np.mean(rewards)),
        mean_no_cost_reward_scaled=float(np.mean(no_cost_scaled)),
        mean_no_cost_reward_raw=float(np.mean(no_cost_raw)),
        mean_episode_length=float(np.mean(lengths)),
    )
