from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .jax_environment import JaxDecisionTreeEnv
from .jax_network import actor_critic_forward, apply_action_mask, sample_actions


def _child_array_to_dict(child_nodes: np.ndarray) -> Dict[int, List[int]]:
    child_dict = {}
    for parent in range(child_nodes.shape[0]):
        children = child_nodes[parent]
        if children[0] >= 0:
            child_dict[parent] = [int(children[0]), int(children[1])]
    return child_dict


def _parent_array_to_dict(parent_nodes: np.ndarray) -> Dict[int, int]:
    parent_dict = {}
    for child in range(parent_nodes.shape[0]):
        parent = int(parent_nodes[child])
        if parent >= 0:
            parent_dict[child] = parent
    return parent_dict


def _leaf_nodes_from_children(child_nodes: np.ndarray) -> np.ndarray:
    return np.where(child_nodes[:, 0] < 0)[0]


def _compute_cum_points(child_nodes: np.ndarray, root_node: int, points: np.ndarray) -> np.ndarray:
    cum_points = np.zeros((points.shape[0],), dtype=np.float32)

    def dfs(node: int, cum: float):
        new_cum = cum + float(points[node])
        cum_points[node] = new_cum
        children = child_nodes[node]
        if children[0] >= 0:
            dfs(int(children[0]), new_cum)
            dfs(int(children[1]), new_cum)

    dfs(root_node, 0.0)
    return cum_points


def _compute_depths(child_nodes: np.ndarray, root_node: int) -> np.ndarray:
    depths = np.zeros((child_nodes.shape[0],), dtype=np.int32)

    def dfs(node: int, depth: int):
        depths[node] = depth
        children = child_nodes[node]
        if children[0] >= 0:
            dfs(int(children[0]), depth + 1)
            dfs(int(children[1]), depth + 1)

    dfs(root_node, 0)
    return depths


class JaxSimulator:
    def __init__(self, env: JaxDecisionTreeEnv):
        self.env = env
        self._trial_jit = jax.jit(self._run_trial, static_argnames=("greedy",))

    def _run_trial(self, params: Any, rng_key: jax.Array, greedy: bool = False):
        state, obs, info = self.env.reset(rng_key)
        action_mask = info["mask"]

        action_seq = -jnp.ones((self.env.t_max,), dtype=jnp.int32)

        carry = (
            state,
            obs,
            action_mask,
            action_seq,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            rng_key,
        )

        def cond_fn(carry):
            _, _, _, _, step_count, done, _ = carry
            return (~done) & (step_count < self.env.t_max)

        def body_fn(carry):
            state, obs, action_mask, action_seq, step_count, _, rng_key = carry

            logits, _ = actor_critic_forward(params, obs[None, :])
            logits = logits[0]

            def greedy_action(_):
                masked_logits = apply_action_mask(logits, action_mask)
                return jnp.argmax(masked_logits)

            def sampled_action(rng_key):
                rng_key, action_key = jax.random.split(rng_key)
                action, _, _ = sample_actions(action_key, logits[None, :], action_mask[None, :])
                return action[0], rng_key

            if greedy:
                action = greedy_action(None)
            else:
                action, rng_key = sampled_action(rng_key)

            state, obs, _, done, _, info = self.env.step(state, action)
            action_mask = info["mask"]
            action_seq = action_seq.at[step_count].set(action)
            step_count = step_count + 1

            return state, obs, action_mask, action_seq, step_count, done, rng_key

        state, _, _, action_seq, action_len, _, rng_key = jax.lax.while_loop(cond_fn, body_fn, carry)

        return state, action_seq, action_len, rng_key

    def simulate(
        self,
        params: Any,
        seed: int,
        num_trials: int,
        greedy: bool = False,
    ):
        rng_key = jax.random.PRNGKey(seed)

        data = {
            "child_dicts": [],
            "parent_dicts": [],
            "root_nodes": [],
            "leaf_nodes": [],
            "depths": [],
            "points": [],
            "cum_points": [],
            "action_seqs": [],
            "choice_seqs": [],
        }

        for _ in range(num_trials):
            rng_key, trial_key = jax.random.split(rng_key)
            state, action_seq, action_len, _ = self._trial_jit(params, trial_key, greedy=greedy)

            child_nodes = np.asarray(state.child_nodes)
            parent_nodes = np.asarray(state.parent_nodes)
            points = np.asarray(state.points)
            root_node = int(state.root_node)

            action_seq = np.asarray(action_seq[: int(action_len)], dtype=np.int32).tolist()
            choice_seq = np.asarray(state.chosen_path[: int(state.chosen_path_len)], dtype=np.int32).tolist()

            data["child_dicts"].append(_child_array_to_dict(child_nodes))
            data["parent_dicts"].append(_parent_array_to_dict(parent_nodes))
            data["root_nodes"].append(root_node)
            data["leaf_nodes"].append(_leaf_nodes_from_children(child_nodes).tolist())
            data["depths"].append(_compute_depths(child_nodes, root_node).tolist())
            data["points"].append(points.tolist())
            data["cum_points"].append(_compute_cum_points(child_nodes, root_node, points).tolist())
            data["action_seqs"].append(action_seq)
            data["choice_seqs"].append(choice_seq)

        return data
