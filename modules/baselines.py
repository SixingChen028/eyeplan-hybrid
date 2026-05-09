from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from .network import actor_critic_forward, apply_action_mask


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
    child_nodes: List[int]
    root_node: int | None
    g_values: np.ndarray


class ObsLayout:
    def __init__(
        self,
        num_nodes: int,
        use_recency_obs: bool = False,
        use_best_open_value_obs: bool = True,
        use_best_terminal_value_obs: bool = True,
    ):
        self.num_nodes = num_nodes
        self.use_recency_obs = bool(use_recency_obs)
        self.use_best_open_value_obs = bool(use_best_open_value_obs)
        self.use_best_terminal_value_obs = bool(use_best_terminal_value_obs)
        index = 0

        self.fixation = slice(index, index + num_nodes)
        index += num_nodes

        self.point = slice(index, index + 1)
        index += 1

        self.parent = slice(index, index + num_nodes)
        index += num_nodes

        self.children = slice(index, index + num_nodes)
        index += num_nodes

        self.root = slice(index, index + num_nodes)
        index += num_nodes

        self.g = slice(index, index + num_nodes)
        index += num_nodes

        self.q = slice(index, index + num_nodes)
        index += num_nodes

        self.visits = slice(index, index + num_nodes)
        index += num_nodes

        self.is_terminal = slice(index, index + num_nodes)
        index += num_nodes

        self.best_open_value = slice(index, index)
        if self.use_best_open_value_obs or self.use_best_terminal_value_obs:
            self.best_open_value = slice(index, index + 1)
            index += 1

        self.best_terminal_value = slice(index, index)
        if self.use_best_terminal_value_obs:
            self.best_terminal_value = slice(index, index + 1)
            index += 1

        self.recency = slice(index, index + num_nodes)
        if self.use_recency_obs:
            index += num_nodes

        self.time = slice(index, index + 1)


def _decode_one_hot(one_hot: np.ndarray) -> int | None:
    if np.allclose(one_hot, 0.0):
        return None
    return int(np.argmax(one_hot))


def _decode_multi_hot(multi_hot: np.ndarray) -> List[int]:
    return [int(idx) for idx in np.where(multi_hot > 0.5)[0]]


def parse_obs(obs: np.ndarray, layout: ObsLayout) -> ObsView:
    return ObsView(
        fixation_node=_decode_one_hot(obs[layout.fixation]),
        parent_node=_decode_one_hot(obs[layout.parent]),
        child_nodes=_decode_multi_hot(obs[layout.children]),
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


def _move_probs(q_children: np.ndarray, beta_move: float, eps_move: float) -> np.ndarray:
    z = beta_move * (q_children - np.max(q_children))
    probs = np.exp(z)
    probs = probs / np.sum(probs)
    return (1.0 - eps_move) * probs + eps_move * (1.0 / q_children.shape[0])


def _expected_move_reward_raw(state: Any, beta_move: float, eps_move: float) -> float:
    child_nodes = np.asarray(state.child_nodes, dtype=np.int32)
    points = np.asarray(state.points, dtype=np.float32)
    q_values = np.asarray(state.q_values, dtype=np.float32)
    expected = np.zeros(points.shape[0], dtype=np.float32)

    for _ in range(points.shape[0]):
        next_expected = np.zeros_like(expected)
        for node in range(points.shape[0]):
            children = child_nodes[node]
            if children[0] < 0:
                continue

            probs = _move_probs(q_values[children], beta_move, eps_move)
            next_expected[node] = float(np.sum(probs * (points[children] + expected[children])))
        expected = next_expected

    return float(expected[int(state.root_node)])


def _episode_no_cost_rewards(
    state: Any,
    env_params: JaxDecisionTreeParams,
    scale_factor: float,
) -> Tuple[float, float]:
    raw_reward = _expected_move_reward_raw(state, float(env_params.beta_move), float(env_params.eps_move))
    scaled_reward = raw_reward * scale_factor

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
    current = obs_view.fixation_node
    root = obs_view.root_node if obs_view.root_node is not None else policy_state["root_node"]

    if current is not None:
        policy_state["discovered"][current] = True

    if obs_view.parent_node is not None and current is not None:
        parent = obs_view.parent_node
        policy_state["discovered"][parent] = True
        policy_state["parent_map"][current] = parent

    for child in obs_view.child_nodes:
        if child is not None and current is not None:
            policy_state["discovered"][child] = True
            policy_state["parent_map"][child] = current

    if phase == "visit_all":
        discovered = policy_state["discovered"]
        visited = policy_state["visited"]
        parent_map = policy_state["parent_map"]

        unvisited_discovered = np.where(discovered & (~visited))[0]
        if unvisited_discovered.size == 0 and discovered.all():
            policy_state["phase"] = "best_g"
        elif policy_state["visit_steps"] >= policy_state["max_visit_steps"]:
            policy_state["phase"] = "best_g"
        else:
            legal = np.where(action_mask[:num_nodes])[0]

            legal_unvisited = [node for node in legal if discovered[node] and (not visited[node])]
            if len(legal_unvisited) > 0:
                action = int(legal_unvisited[0])
            else:
                parent_candidates = []
                for node in unvisited_discovered:
                    parent = int(parent_map[node])
                    if parent >= 0 and action_mask[parent]:
                        parent_candidates.append(parent)

                if len(parent_candidates) > 0:
                    action = int(parent_candidates[0])
                else:
                    legal_discovered = [node for node in legal if discovered[node]]
                    if len(legal_discovered) > 0:
                        action = int(legal_discovered[0])
                    else:
                        action = _choose_legal_fixation(None, action_mask, root, num_nodes)

            if action < num_nodes:
                visited[action] = True

            policy_state["visit_steps"] += 1
            return action, policy_state

    if phase == "best_g":
        if policy_state["target_node"] is None:
            candidate_mask = policy_state["discovered"] | policy_state["visited"]
            if np.any(candidate_mask):
                candidate_g = np.where(candidate_mask, obs_view.g_values, -np.inf)
                policy_state["target_node"] = int(np.argmax(candidate_g))
            else:
                policy_state["target_node"] = root

        target = int(policy_state["target_node"])

        if current == target:
            policy_state["phase"] = "climb"
        else:
            parent_map = policy_state["parent_map"]
            target_ancestor = target
            while target_ancestor >= 0 and (not action_mask[target_ancestor]):
                target_ancestor = int(parent_map[target_ancestor])

            if target_ancestor >= 0:
                action = int(target_ancestor)
            else:
                action = _choose_legal_fixation(None, action_mask, root, num_nodes)
            return action, policy_state

    if phase == "climb":
        parent = obs_view.parent_node
        if parent is None:
            return num_nodes, policy_state

        action = _choose_legal_fixation(parent, action_mask, root, num_nodes)
        return action, policy_state

    return num_nodes, policy_state


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
            "depth1_nodes": obs_view.child_nodes,
        }

    if policy_name == "visit_all_then_bestg_then_parent_chain":
        root = obs_view.root_node if obs_view.root_node is not None else 0
        discovered = np.zeros((num_nodes,), dtype=bool)
        visited = np.zeros((num_nodes,), dtype=bool)
        parent_map = -np.ones((num_nodes,), dtype=int)

        discovered[root] = True
        visited[root] = True

        for child in obs_view.child_nodes:
            discovered[child] = True
            parent_map[child] = root

        return {
            "phase": "visit_all",
            "root_node": root,
            "visited": visited,
            "discovered": discovered,
            "parent_map": parent_map,
            "visit_steps": 0,
            "max_visit_steps": int(2 * num_nodes),
            "target_node": None,
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
    env_params: JaxDecisionTreeParams,
    policy_names: List[str],
    reset_keys: jax.Array,
) -> Tuple[List[PolicyStats], float, float]:
    layout = ObsLayout(
        env.num_nodes,
        use_recency_obs=getattr(env, "use_recency_obs", False),
        use_best_open_value_obs=getattr(env, "use_best_open_value_obs", True),
        use_best_terminal_value_obs=getattr(env, "use_best_terminal_value_obs", True),
    )
    reset_fn = jax.jit(lambda key: env.reset(key, env_params))
    step_fn = jax.jit(lambda state, action: env.step(state, action, env_params))

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
            moved = False

            while (not done) and (steps < env.t_max):
                obs_view = parse_obs(obs_np, layout)
                action, policy_state = _get_policy_action(
                    policy_name,
                    obs_view,
                    action_mask_np,
                    policy_state,
                    env.num_nodes,
                )

                state, obs, reward, done, info = step_fn(state, int(action))

                obs_np = np.asarray(obs)
                action_mask_np = np.asarray(info["mask"])
                episode_reward += float(reward)
                moved = int(action) == env.num_nodes
                steps += 1

            scaled_no_cost, raw_no_cost = (
                _episode_no_cost_rewards(state, env_params, env.scale_factor)
                if moved
                else (0.0, 0.0)
            )

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
    optimal_mean_scaled = optimal_mean_raw * env.scale_factor

    return stats, optimal_mean_scaled, optimal_mean_raw


def evaluate_network_greedy(
    env: JaxDecisionTreeEnv,
    env_params: JaxDecisionTreeParams,
    params: Any,
    reset_keys: jax.Array,
) -> PolicyStats:
    reset_fn = jax.jit(lambda key: env.reset(key, env_params))
    step_fn = jax.jit(lambda state, action: env.step(state, action, env_params))
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
        moved = False

        while (not done) and (steps < env.t_max):
            logits, _ = forward_fn(params, obs[None, :])
            masked_logits = apply_action_mask(logits[0], info["mask"])
            action = int(jnp.argmax(masked_logits))

            state, obs, reward, done, info = step_fn(state, action)
            episode_reward += float(reward)
            moved = action == env.num_nodes
            steps += 1

        scaled_no_cost, raw_no_cost = (
            _episode_no_cost_rewards(state, env_params, env.scale_factor)
            if moved
            else (0.0, 0.0)
        )

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
