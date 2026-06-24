from __future__ import annotations

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .environment import DecisionTreeEnv, DecisionTreeParams
from .simulation import append_simulation_trial, empty_simulation_data

# Hard-coded from fit_random_looks() in ../analysis/julia/src/agent_registry.jl.
# That function fits the number of human looks with Gamma(shape, scale) and caps
# the native random baseline at 50 looks.
RANDOM_SEARCH_STOP_GAMMA_SHAPE = 1.931259599212531
RANDOM_SEARCH_STOP_GAMMA_SCALE = 7.168940848414317
RANDOM_SEARCH_STOP_MAX_FIXATIONS = 50


class RandomSearchSimulator:
    def __init__(self, env: DecisionTreeEnv, env_params: DecisionTreeParams):
        self.env = env
        self.env_params = env_params
        self._trial_batch_jit = jax.jit(self._run_trial_batch)

    def _sample_fixation_target(self, key: jax.Array) -> jax.Array:
        total_fixations = jnp.floor(
            jax.random.gamma(key, RANDOM_SEARCH_STOP_GAMMA_SHAPE) * RANDOM_SEARCH_STOP_GAMMA_SCALE
        ).astype(jnp.int32)
        total_fixations = jnp.minimum(total_fixations, RANDOM_SEARCH_STOP_MAX_FIXATIONS)
        return jnp.maximum(total_fixations - 1, 0)

    def _sample_random_fixation(self, key: jax.Array, action_mask: jax.Array) -> jax.Array:
        fixation_mask = action_mask[: self.env.num_nodes]
        legal_count = jnp.sum(fixation_mask.astype(jnp.int32))
        probs = fixation_mask.astype(jnp.float32) / jnp.maximum(legal_count, 1)
        fixation = jax.random.choice(key, self.env.num_nodes, p=probs)
        return jnp.where(legal_count > 0, fixation, self.env.num_nodes)

    def _run_trial(self, rng_key: jax.Array):
        env_params = self.env_params
        rng_key, reset_key, stop_key = jax.random.split(rng_key, 3)
        state, _, info = self.env.reset(reset_key, env_params)
        action_mask = info["mask"]
        target_fixations = self._sample_fixation_target(stop_key)

        action_seq = -jnp.ones((self.env.t_max,), dtype=jnp.int32)

        carry = (
            state,
            action_mask,
            action_seq,
            self.env.empty_path,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            rng_key,
        )

        def cond_fn(carry):
            _, _, _, _, step_count, done, _ = carry
            return (~done) & (step_count < self.env.t_max)

        def body_fn(carry):
            state, action_mask, action_seq, choice_path, step_count, _, rng_key = carry
            rng_key, action_key = jax.random.split(rng_key)

            random_fixation = self._sample_random_fixation(action_key, action_mask)
            action = jnp.where(step_count < target_fixations, random_fixation, self.env.num_nodes)

            state, _, _, done, info = self.env.step(state, action, env_params)
            action_mask = info["mask"]
            choice_path = jnp.where(action == self.env.num_nodes, info["choice_path"], choice_path)
            action_seq = action_seq.at[step_count].set(action)
            step_count = step_count + 1

            return state, action_mask, action_seq, choice_path, step_count, done, rng_key

        state, _, action_seq, choice_path, action_len, _, rng_key = jax.lax.while_loop(
            cond_fn,
            body_fn,
            carry,
        )
        return state, action_seq, choice_path, action_len, rng_key

    def _run_trial_batch(self, trial_keys: jax.Array):
        states, action_seqs, choice_paths, action_lens, _ = jax.vmap(self._run_trial)(trial_keys)
        return states, action_seqs, choice_paths, action_lens

    def simulate(
        self,
        *,
        seed: int,
        num_trials: int,
        batch_size: int = 512,
        skip_timeout_trials: bool = True,
    ) -> Dict[str, List[Any]]:
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rng_key = jax.random.PRNGKey(seed)
        num_batches = int(np.ceil(num_trials / batch_size))
        data = empty_simulation_data(detailed=False)

        for batch_idx in range(num_batches):
            rng_key, batch_key = jax.random.split(rng_key)
            trial_keys = jax.random.split(batch_key, batch_size)
            states, action_seqs, choice_paths, action_lens = self._trial_batch_jit(trial_keys)

            states = jax.device_get(states)
            action_seqs = np.asarray(action_seqs)
            choice_paths = np.asarray(choice_paths)
            action_lens = np.asarray(action_lens)
            child_nodes_batch = np.asarray(states.child_nodes)
            points_batch = np.asarray(states.points)
            root_nodes_batch = np.asarray(states.root_node)

            trials_remaining = num_trials - (batch_idx * batch_size)
            trials_in_batch = min(batch_size, trials_remaining)

            for trial_idx in range(trials_in_batch):
                action_len = int(action_lens[trial_idx])
                action_seq = np.asarray(action_seqs[trial_idx, :action_len], dtype=np.int32).tolist()
                choice_path = np.asarray(choice_paths[trial_idx], dtype=np.int32)
                choice_seq = choice_path[choice_path >= 0].tolist()

                append_simulation_trial(
                    data,
                    child_nodes=child_nodes_batch[trial_idx],
                    root_node=int(root_nodes_batch[trial_idx]),
                    points=points_batch[trial_idx],
                    action_seq=action_seq,
                    choice_seq=choice_seq,
                    num_nodes=self.env.num_nodes,
                    t_max=self.env.t_max,
                    skip_timeout_trials=skip_timeout_trials,
                )

        return data
