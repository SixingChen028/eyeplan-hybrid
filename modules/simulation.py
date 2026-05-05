from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .environment import JaxDecisionTreeEnv
from .network import actor_critic_forward, apply_action_mask, sample_actions


DETAIL_KEYS = ["activations", "counts", "gs", "qs", "logits"]


def empty_simulation_data(*, detailed: bool = False) -> Dict[str, List[Any]]:
    data = {
        "adj_lists": [],
        "starts": [],
        "rewards": [],
        "actions": [],
        "chosen_paths": [],
    }
    if detailed:
        for key in DETAIL_KEYS:
            data[key] = []
    return data


def _adj_list_from_child_nodes(child_nodes: np.ndarray, num_nodes: int) -> List[List[int]]:
    adj_list = [[] for _ in range(num_nodes)]
    for parent, children in enumerate(child_nodes):
        if children[0] >= 0:
            adj_list[parent] = [int(children[0]), int(children[1])]
    return adj_list


def append_simulation_trial(
    data: Dict[str, List[Any]],
    *,
    child_nodes: np.ndarray,
    root_node: int,
    points: np.ndarray,
    action_seq: List[int],
    choice_seq: List[int],
    num_nodes: int,
    t_max: int,
    skip_timeout_trials: bool,
    details: Dict[str, Any] | None = None,
) -> bool:
    if skip_timeout_trials and len(action_seq) >= t_max:
        return False
    if len(action_seq) == 0 or action_seq[-1] != num_nodes:
        return False

    data["adj_lists"].append(_adj_list_from_child_nodes(child_nodes, num_nodes))
    data["starts"].append(int(root_node))
    data["rewards"].append([float(value) for value in points])
    data["actions"].append([int(root_node)] + [int(action) for action in action_seq])
    data["chosen_paths"].append([int(choice) for choice in choice_seq])

    if details is not None:
        for key in DETAIL_KEYS:
            data[key].append(details[key])
    return True


class JaxSimulator:
    def __init__(self, env: JaxDecisionTreeEnv):
        self.env = env
        self._trial_jit = jax.jit(self._run_trial, static_argnames=("greedy",))
        self._trial_batch_jit = jax.jit(self._run_trial_batch, static_argnames=("greedy",))
        self._eval_batch_jit = jax.jit(self._run_eval_batch, static_argnames=("greedy",))

    def _run_trial(self, params: Any, rng_key: jax.Array, greedy: bool = False):
        env_params = self.env.params()
        state, obs, info = self.env.reset_with_params(rng_key, env_params)
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
            logits, _ = actor_critic_forward(params, obs[None, :], action_mask[None, :])
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
            raw_action = action
            state, obs, _, done, _, info = self.env.step_with_params(state, action, env_params)
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

        final_action_index = jnp.maximum(action_len - 1, 0)
        final_action = action_seq[final_action_index]

        def sample_choice_path(payload):
            state, rng_key = payload
            sample_state = state._replace(rng_key=rng_key)
            _, path, path_len, rng_key = self.env._sample_move_path(sample_state, env_params)
            return path, path_len, rng_key

        def empty_choice_path(payload):
            _, rng_key = payload
            return self.env.empty_path, jnp.int32(0), rng_key

        choice_path, choice_path_len, rng_key = jax.lax.cond(
            final_action == self.env.num_nodes,
            sample_choice_path,
            empty_choice_path,
            (state, rng_key),
        )

        return (
            state,
            action_seq,
            choice_path,
            activation_seq,
            count_seq,
            g_seq,
            q_seq,
            logits_seq,
            action_len,
            choice_path_len,
            rng_key,
        )

    def _run_trial_metrics(self, params: Any, rng_key: jax.Array, greedy: bool = False):
        env_params = self.env.params()
        state, obs, info = self.env.reset_with_params(rng_key, env_params)
        action_mask = info["mask"]

        carry = (
            state,
            obs,
            action_mask,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(False),
            jnp.array(False),
            rng_key,
        )

        def cond_fn(carry):
            _, _, _, step_count, _, _, done, _ = carry
            return (~done) & (step_count < self.env.t_max)

        def body_fn(carry):
            state, obs, action_mask, step_count, episode_reward, moved, _, rng_key = carry

            logits, _ = actor_critic_forward(params, obs[None, :], action_mask[None, :])
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

            state, obs, reward, done, _, info = self.env.step_with_params(state, action, env_params)
            action_mask = info["mask"]
            step_count = step_count + 1
            episode_reward = episode_reward + reward
            moved = action == self.env.num_nodes

            return state, obs, action_mask, step_count, episode_reward, moved, done, rng_key

        state, _, _, step_count, episode_reward, moved, _, rng_key = jax.lax.while_loop(cond_fn, body_fn, carry)

        no_cost_reward = jax.lax.cond(
            moved,
            lambda _: self.env._expected_move_reward(state, env_params) * env_params.scale_factor,
            lambda _: jnp.array(0.0, dtype=jnp.float32),
            operand=None,
        )

        return episode_reward, no_cost_reward, step_count, rng_key

    def _run_eval_batch(self, params: Any, trial_keys: jax.Array, greedy: bool = False):
        rewards, rewards_no_cost, steps, _ = jax.vmap(
            lambda key: self._run_trial_metrics(params, key, greedy=greedy)
        )(trial_keys)
        return rewards, rewards_no_cost, steps

    def _run_trial_batch(self, params: Any, trial_keys: jax.Array, greedy: bool = False):
        (
            states,
            action_seqs,
            choice_paths,
            activation_seqs,
            count_seqs,
            g_seqs,
            q_seqs,
            logits_seqs,
            action_lens,
            choice_path_lens,
            _,
        ) = jax.vmap(
            lambda key: self._run_trial(params, key, greedy=greedy)
        )(trial_keys)
        return (
            states,
            action_seqs,
            choice_paths,
            activation_seqs,
            count_seqs,
            g_seqs,
            q_seqs,
            logits_seqs,
            action_lens,
            choice_path_lens,
        )

    def simulate(
        self,
        params: Any,
        seed: int,
        num_trials: int,
        greedy: bool = False,
        batch_size: int = 512,
        detailed: bool = False,
        skip_timeout_trials: bool = True,
    ):
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rng_key = jax.random.PRNGKey(seed)
        num_batches = int(np.ceil(num_trials / batch_size))

        data = empty_simulation_data(detailed=detailed)

        for batch_idx in range(num_batches):
            rng_key, batch_key = jax.random.split(rng_key)
            trial_keys = jax.random.split(batch_key, batch_size)
            (
                states,
                action_seqs,
                choice_paths,
                activation_seqs,
                count_seqs,
                g_seqs,
                q_seqs,
                logits_seqs,
                action_lens,
                choice_path_lens,
            ) = self._trial_batch_jit(params, trial_keys, greedy=greedy)

            states = jax.device_get(states)
            action_seqs = np.asarray(action_seqs)
            choice_paths = np.asarray(choice_paths)
            action_lens = np.asarray(action_lens)
            choice_path_lens = np.asarray(choice_path_lens)
            if detailed:
                activation_seqs = np.asarray(activation_seqs)
                count_seqs = np.asarray(count_seqs)
                g_seqs = np.asarray(g_seqs)
                q_seqs = np.asarray(q_seqs)
                logits_seqs = np.asarray(logits_seqs)

            child_nodes_batch = np.asarray(states.child_nodes)
            points_batch = np.asarray(states.points)
            root_nodes_batch = np.asarray(states.root_node)

            trials_remaining = num_trials - (batch_idx * batch_size)
            trials_in_batch = min(batch_size, trials_remaining)

            for trial_idx in range(trials_in_batch):
                child_nodes = child_nodes_batch[trial_idx]
                points = points_batch[trial_idx]
                root_node = int(root_nodes_batch[trial_idx])

                action_len = int(action_lens[trial_idx])
                action_seq = np.asarray(action_seqs[trial_idx, :action_len], dtype=np.int32).tolist()

                choice_len = int(choice_path_lens[trial_idx])
                choice_seq = np.asarray(choice_paths[trial_idx, :choice_len], dtype=np.int32).tolist()

                details = None
                if detailed:
                    details = {
                        "activations": activation_seqs[trial_idx, :action_len].tolist(),
                        "counts": count_seqs[trial_idx, :action_len].tolist(),
                        "gs": g_seqs[trial_idx, :action_len].tolist(),
                        "qs": q_seqs[trial_idx, :action_len].tolist(),
                    }
                    logits_seq = np.asarray(logits_seqs[trial_idx, :action_len], dtype=np.float32).tolist()
                    details["logits"] = logits_seq

                append_simulation_trial(
                    data,
                    child_nodes=child_nodes,
                    root_node=root_node,
                    points=points,
                    action_seq=action_seq,
                    choice_seq=choice_seq,
                    num_nodes=self.env.num_nodes,
                    t_max=self.env.t_max,
                    skip_timeout_trials=skip_timeout_trials,
                    details=details,
                )

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
