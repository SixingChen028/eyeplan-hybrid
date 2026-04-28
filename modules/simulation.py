from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .environment import JaxDecisionTreeEnv
from .network import actor_critic_forward, apply_action_mask, sample_actions


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
        self._trial_batch_jit = jax.jit(self._run_trial_batch, static_argnames=("greedy",))
        self._eval_batch_jit = jax.jit(self._run_eval_batch, static_argnames=("greedy",))

    def _run_trial(self, params: Any, rng_key: jax.Array, greedy: bool = False):
        state, obs, info = self.env.reset(rng_key)
        action_mask = info["mask"]

        action_seq = -jnp.ones((self.env.t_max,), dtype=jnp.int32)
        activation_seq = jnp.zeros((self.env.t_max, self.env.num_nodes), dtype=jnp.float32)
        count_seq = jnp.zeros((self.env.t_max, self.env.num_nodes), dtype=jnp.int32)
        g_seq = jnp.zeros((self.env.t_max, self.env.num_nodes), dtype=jnp.float32)
        q_seq = jnp.zeros((self.env.t_max, self.env.num_nodes), dtype=jnp.float32)
        logits_seq = jnp.zeros((self.env.t_max, self.env.action_size), dtype=jnp.float32)

        carry = (
            state,
            obs,
            action_mask,
            action_seq,
            activation_seq,
            count_seq,
            g_seq,
            q_seq,
            logits_seq,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            rng_key,
        )

        def cond_fn(carry):
            _, _, _, _, _, _, _, _, _, step_count, done, _ = carry
            return (~done) & (step_count < self.env.t_max)

        def body_fn(carry):
            (
                state,
                obs,
                action_mask,
                action_seq,
                activation_seq,
                count_seq,
                g_seq,
                q_seq,
                logits_seq,
                step_count,
                _,
                rng_key,
            ) = carry

            activation_seq = activation_seq.at[step_count].set(state.activation)
            count_seq = count_seq.at[step_count].set(state.n_visits)
            g_seq = g_seq.at[step_count].set(state.g_values)
            q_seq = q_seq.at[step_count].set(state.q_values)
            logits, _ = actor_critic_forward(params, obs[None, :])
            logits = logits[0]
            logits_seq = logits_seq.at[step_count].set(logits)

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

            raw_action = jnp.where(
                action < self.env.num_nodes,
                state.canon_to_raw[jnp.minimum(action, self.env.num_nodes - 1)],
                action,
            )
            state, obs, _, done, _, info = self.env.step(state, action)
            action_mask = info["mask"]
            action_seq = action_seq.at[step_count].set(raw_action)
            step_count = step_count + 1

            return (
                state,
                obs,
                action_mask,
                action_seq,
                activation_seq,
                count_seq,
                g_seq,
                q_seq,
                logits_seq,
                step_count,
                done,
                rng_key,
            )

        (
            state,
            _,
            _,
            action_seq,
            activation_seq,
            count_seq,
            g_seq,
            q_seq,
            logits_seq,
            action_len,
            _,
            rng_key,
        ) = jax.lax.while_loop(cond_fn, body_fn, carry)

        return state, action_seq, activation_seq, count_seq, g_seq, q_seq, logits_seq, action_len, rng_key

    def _run_trial_metrics(self, params: Any, rng_key: jax.Array, greedy: bool = False):
        state, obs, info = self.env.reset(rng_key)
        action_mask = info["mask"]

        carry = (
            state,
            obs,
            action_mask,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(False),
            rng_key,
        )

        def cond_fn(carry):
            _, _, _, step_count, _, done, _ = carry
            return (~done) & (step_count < self.env.t_max)

        def body_fn(carry):
            state, obs, action_mask, step_count, episode_reward, _, rng_key = carry

            logits, _ = actor_critic_forward(params, obs[None, :])
            logits = logits[0]

            def greedy_action(_):
                masked_logits = apply_action_mask(logits, action_mask)
                return jnp.argmax(masked_logits), rng_key

            def sampled_action(key):
                next_key, action_key = jax.random.split(key)
                action, _, _ = sample_actions(action_key, logits[None, :], action_mask[None, :])
                return action[0], next_key

            action, rng_key = jax.lax.cond(
                greedy,
                greedy_action,
                sampled_action,
                rng_key,
            )

            state, obs, reward, done, _, info = self.env.step(state, action)
            action_mask = info["mask"]
            step_count = step_count + 1
            episode_reward = episode_reward + reward

            return state, obs, action_mask, step_count, episode_reward, done, rng_key

        state, _, _, step_count, episode_reward, _, rng_key = jax.lax.while_loop(cond_fn, body_fn, carry)

        chosen_len = state.chosen_path_len
        chosen_mask = jnp.arange(self.env.num_nodes, dtype=jnp.int32) < chosen_len
        safe_path = jnp.where(chosen_mask, state.chosen_path, 0)
        no_cost_reward = jnp.sum(jnp.where(chosen_mask, state.points[safe_path], 0.0))
        no_cost_reward = no_cost_reward * self.env.scale_factor

        return episode_reward, no_cost_reward, step_count, rng_key

    def _run_eval_batch(self, params: Any, trial_keys: jax.Array, greedy: bool = False):
        rewards, rewards_no_cost, steps, _ = jax.vmap(
            lambda key: self._run_trial_metrics(params, key, greedy=greedy)
        )(trial_keys)
        return rewards, rewards_no_cost, steps

    def _run_trial_batch(self, params: Any, trial_keys: jax.Array, greedy: bool = False):
        states, action_seqs, activation_seqs, count_seqs, g_seqs, q_seqs, logits_seqs, action_lens, _ = jax.vmap(
            lambda key: self._run_trial(params, key, greedy=greedy)
        )(trial_keys)
        return states, action_seqs, activation_seqs, count_seqs, g_seqs, q_seqs, logits_seqs, action_lens

    def simulate(
        self,
        params: Any,
        seed: int,
        num_trials: int,
        greedy: bool = False,
        batch_size: int = 512,
        detailed: bool = False,
    ):
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rng_key = jax.random.PRNGKey(seed)
        num_batches = int(np.ceil(num_trials / batch_size))

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
        if detailed:
            data.update(
                {
                    "activations": [],
                    "counts": [],
                    "gs": [],
                    "qs": [],
                    "logits": [],
                }
            )

        for batch_idx in range(num_batches):
            rng_key, batch_key = jax.random.split(rng_key)
            trial_keys = jax.random.split(batch_key, batch_size)
            (
                states,
                action_seqs,
                activation_seqs,
                count_seqs,
                g_seqs,
                q_seqs,
                logits_seqs,
                action_lens,
            ) = self._trial_batch_jit(params, trial_keys, greedy=greedy)

            states = jax.device_get(states)
            action_seqs = np.asarray(action_seqs)
            action_lens = np.asarray(action_lens)
            if detailed:
                activation_seqs = np.asarray(activation_seqs)
                count_seqs = np.asarray(count_seqs)
                g_seqs = np.asarray(g_seqs)
                q_seqs = np.asarray(q_seqs)
                logits_seqs = np.asarray(logits_seqs)

            child_nodes_batch = np.asarray(states.child_nodes)
            parent_nodes_batch = np.asarray(states.parent_nodes)
            points_batch = np.asarray(states.points)
            root_nodes_batch = np.asarray(states.root_node)
            chosen_paths_batch = np.asarray(states.chosen_path)
            chosen_path_lens_batch = np.asarray(states.chosen_path_len)

            trials_remaining = num_trials - (batch_idx * batch_size)
            trials_in_batch = min(batch_size, trials_remaining)

            for trial_idx in range(trials_in_batch):
                child_nodes = child_nodes_batch[trial_idx]
                parent_nodes = parent_nodes_batch[trial_idx]
                points = points_batch[trial_idx]
                root_node = int(root_nodes_batch[trial_idx])

                action_len = int(action_lens[trial_idx])
                action_seq = np.asarray(action_seqs[trial_idx, :action_len], dtype=np.int32).tolist()

                choice_len = int(chosen_path_lens_batch[trial_idx])
                choice_seq = np.asarray(chosen_paths_batch[trial_idx, :choice_len], dtype=np.int32).tolist()

                data["child_dicts"].append(_child_array_to_dict(child_nodes))
                data["parent_dicts"].append(_parent_array_to_dict(parent_nodes))
                data["root_nodes"].append(root_node)
                data["leaf_nodes"].append(_leaf_nodes_from_children(child_nodes).tolist())
                data["depths"].append(_compute_depths(child_nodes, root_node).tolist())
                data["points"].append(points.tolist())
                data["cum_points"].append(_compute_cum_points(child_nodes, root_node, points).tolist())
                data["action_seqs"].append(action_seq)
                data["choice_seqs"].append(choice_seq)
                if detailed:
                    data["activations"].append(activation_seqs[trial_idx, :action_len].tolist())
                    data["counts"].append(count_seqs[trial_idx, :action_len].tolist())
                    data["gs"].append(g_seqs[trial_idx, :action_len].tolist())
                    data["qs"].append(q_seqs[trial_idx, :action_len].tolist())
                    logits_seq = np.asarray(logits_seqs[trial_idx, :action_len], dtype=np.float32).tolist()
                    data["logits"].append(logits_seq)

        return data

    def evaluate_policy(
        self,
        params: Any,
        seed: int,
        num_trials: int,
        greedy: bool = True,
        batch_size: int = 512,
    ):
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rng_key = jax.random.PRNGKey(seed)
        num_batches = int(np.ceil(num_trials / batch_size))

        rewards = []
        rewards_no_cost = []
        steps = []

        for _ in range(num_batches):
            rng_key, batch_key = jax.random.split(rng_key)
            trial_keys = jax.random.split(batch_key, batch_size)
            batch_rewards, batch_rewards_no_cost, batch_steps = self._eval_batch_jit(
                params,
                trial_keys,
                greedy=greedy,
            )
            rewards.append(np.asarray(batch_rewards, dtype=np.float32))
            rewards_no_cost.append(np.asarray(batch_rewards_no_cost, dtype=np.float32))
            steps.append(np.asarray(batch_steps, dtype=np.float32))

        rewards = np.concatenate(rewards)[:num_trials]
        rewards_no_cost = np.concatenate(rewards_no_cost)[:num_trials]
        steps = np.concatenate(steps)[:num_trials]

        return {
            "num_trials": int(num_trials),
            "greedy": bool(greedy),
            "reward_mean": float(np.mean(rewards)),
            "reward_sd": float(np.std(rewards)),
            "reward_no_cost_mean": float(np.mean(rewards_no_cost)),
            "reward_no_cost_sd": float(np.std(rewards_no_cost)),
            "n_steps_mean": float(np.mean(steps)),
            "n_steps_sd": float(np.std(steps)),
        }


def to_transformed_simulation_format(
    data: Dict[str, List[Any]],
    *,
    num_nodes: int,
    t_max: int,
    skip_timeout_trials: bool = True,
    detailed: bool = False,
) -> Dict[str, List[Any]]:
    """
    Convert simulator output to the JSON schema:
      {
        "adj_lists": [...],
        "starts": [...],
        "rewards": [...],
        "actions": [...],
      }
    """

    transformed = {
        "adj_lists": [],
        "starts": [],
        "rewards": [],
        "actions": [],
        "chosen_paths": [],
    }
    detail_keys = ["activations", "counts", "gs", "qs", "logits"]
    if detailed:
        for key in detail_keys:
            if key not in data:
                raise ValueError(f"Detailed simulation data missing key: {key}")
            transformed[key] = []

    child_dicts = data.get("child_dicts", [])
    root_nodes = data.get("root_nodes", [])
    points_list = data.get("points", [])
    action_seqs = data.get("action_seqs", [])
    choice_seqs = data.get("choice_seqs", [])

    for trial_idx, (child_dict, root_node, points, action_seq, choice_seq) in enumerate(
        zip(
            child_dicts,
            root_nodes,
            points_list,
            action_seqs,
            choice_seqs,
        )
    ):
        action_seq = [int(a) for a in action_seq]
        choice_seq = [int(a) for a in choice_seq]
        root_node = int(root_node)
        points = [float(v) for v in points]

        if skip_timeout_trials and len(action_seq) >= t_max:
            continue

        adj_list = [[] for _ in range(num_nodes)]
        for parent, children in child_dict.items():
            parent = int(parent)
            adj_list[parent] = [int(children[0]), int(children[1])]

        if len(action_seq) == 0:
            continue

        if action_seq[-1] != num_nodes:
            continue

        actions = [root_node] + action_seq

        transformed["adj_lists"].append(adj_list)
        transformed["starts"].append(root_node)
        transformed["rewards"].append(points)
        transformed["actions"].append(actions)
        transformed["chosen_paths"].append(choice_seq)
        if detailed:
            for key in detail_keys:
                transformed[key].append(data[key][trial_idx])

    return transformed
