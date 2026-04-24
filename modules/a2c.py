import os
import pickle
import time
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .environment import JaxDecisionTreeEnv
from .network import actor_critic_forward, init_mlp_actor_critic_params, sample_actions


class AdamState(NamedTuple):
    step: jax.Array
    m: Any
    v: Any


class JaxTrainState(NamedTuple):
    params: Any
    optimizer: AdamState
    rng_key: jax.Array


class StepMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy_loss: jax.Array
    episode_reward: jax.Array
    episode_length: jax.Array
    grad_norm: jax.Array
    param_norm: jax.Array


class RolloutBatch(NamedTuple):
    masks: jax.Array
    rewards: jax.Array
    log_probs: jax.Array
    entropies: jax.Array
    values: jax.Array


def _zero_step_metrics(dtype=jnp.float32):
    zero = jnp.array(0.0, dtype=dtype)
    return StepMetrics(
        loss=zero,
        policy_loss=zero,
        value_loss=zero,
        entropy_loss=zero,
        episode_reward=zero,
        episode_length=zero,
        grad_norm=zero,
        param_norm=zero,
    )


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _tree_from_numpy(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def save_jax_tree(tree: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(_tree_to_numpy(tree), file)


def load_jax_tree(path: str):
    with open(path, "rb") as file:
        tree = pickle.load(file)
    return _tree_from_numpy(tree)


def save_jax_params(params: Any, path: str):
    save_jax_tree(params, path)


def load_jax_params(path: str):
    return load_jax_tree(path)


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def _select_not_done(done: jax.Array, new: jax.Array, old: jax.Array):
    selector = done
    while selector.ndim < new.ndim:
        selector = selector[..., None]
    return jnp.where(selector, old, new)


class JaxBatchMaskA2C:
    def __init__(
        self,
        env: JaxDecisionTreeEnv,
        feature_size: int,
        action_size: int,
        hidden_size: int,
        batch_size: int,
        lr: float,
        gamma: float,
        lamda: float,
        beta_v: float,
        beta_e: float,
        max_grad_norm: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        self.env = env
        self.feature_size = int(feature_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)

        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.lamda = float(lamda)
        self.beta_v = float(beta_v)
        self.beta_e = float(beta_e)
        self.max_grad_norm = float(max_grad_norm)

        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)

        self._train_step_jit = jax.jit(self._train_step)
        self._train_many_jit = jax.jit(self._train_many)
        self._train_many_mean_metrics_jit = jax.jit(self._train_many_mean_metrics)

    def init_state(self, seed: int = 0) -> JaxTrainState:
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)

        params = init_mlp_actor_critic_params(
            init_key,
            feature_size=self.feature_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
        )

        optimizer = AdamState(
            step=jnp.array(0, dtype=jnp.int32),
            m=_zeros_like_tree(params),
            v=_zeros_like_tree(params),
        )

        return JaxTrainState(params=params, optimizer=optimizer, rng_key=key)

    def _rollout(self, params: Any, rng_key: jax.Array):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, self.batch_size)

        env_state, obs, info = jax.vmap(self.env.reset)(reset_keys)
        action_mask = info["mask"]
        done = jnp.zeros((self.batch_size,), dtype=jnp.bool_)
        zero_output = jnp.zeros((self.batch_size,), dtype=jnp.float32)

        def body_fn(carry, _):
            env_state, obs, action_mask, done, rng_key = carry

            def done_branch(carry):
                env_state, obs, action_mask, done, rng_key = carry
                rng_key, _ = jax.random.split(rng_key)

                output = RolloutBatch(
                    masks=zero_output,
                    rewards=zero_output,
                    log_probs=zero_output,
                    entropies=zero_output,
                    values=zero_output,
                )
                return (env_state, obs, action_mask, done, rng_key), output

            def active_branch(carry):
                env_state, obs, action_mask, done, rng_key = carry

                mask = 1.0 - done.astype(jnp.float32)

                logits, values = actor_critic_forward(params, obs)

                rng_key, action_key = jax.random.split(rng_key)
                actions, log_probs, entropies = sample_actions(action_key, logits, action_mask)

                next_env_state, next_obs, rewards, dones, _, info = jax.vmap(self.env.step)(env_state, actions)
                next_action_mask = info["mask"]

                env_state = jax.tree_util.tree_map(
                    lambda new, old: _select_not_done(done, new, old),
                    next_env_state,
                    env_state,
                )
                obs = _select_not_done(done, next_obs, obs)
                action_mask = _select_not_done(done, next_action_mask, action_mask)
                done = jnp.logical_or(done, dones)

                rewards = rewards.astype(jnp.float32) * mask
                log_probs = log_probs * mask
                entropies = entropies * mask
                values = values * mask

                output = RolloutBatch(
                    masks=mask,
                    rewards=rewards,
                    log_probs=log_probs,
                    entropies=entropies,
                    values=values,
                )

                return (env_state, obs, action_mask, done, rng_key), output

            return jax.lax.cond(
                jnp.all(done),
                done_branch,
                active_branch,
                carry,
            )

        carry, rollout = jax.lax.scan(
            body_fn,
            (env_state, obs, action_mask, done, rng_key),
            xs=None,
            length=self.env.t_max,
        )

        new_key = carry[4]
        return rollout, new_key

    def _discounted_returns_and_advantages(self, rewards: jax.Array, values: jax.Array):
        batch_size = rewards.shape[1]
        padded_values = jnp.concatenate(
            [values, jnp.zeros((1, batch_size), dtype=values.dtype)],
            axis=0,
        )

        rewards_rev = rewards[::-1]
        values_rev = padded_values[:-1][::-1]
        next_values_rev = padded_values[1:][::-1]

        def body_fn(carry, xs):
            R, advantage = carry
            reward, value, value_next = xs

            R = reward + self.gamma * R
            delta = reward + self.gamma * value_next - value
            advantage = delta + self.gamma * self.lamda * advantage

            return (R, advantage), (R, advantage)

        init = (
            jnp.zeros((batch_size,), dtype=rewards.dtype),
            jnp.zeros((batch_size,), dtype=rewards.dtype),
        )

        (_, _), (returns_rev, advantages_rev) = jax.lax.scan(
            body_fn,
            init,
            (rewards_rev, values_rev, next_values_rev),
        )

        return returns_rev[::-1], advantages_rev[::-1]

    def _loss_and_metrics(self, params: Any, rng_key: jax.Array, beta_e: jax.Array):
        rollout, new_key = self._rollout(params, rng_key)

        returns, advantages = self._discounted_returns_and_advantages(
            rewards=rollout.rewards,
            values=rollout.values,
        )

        detached_advantages = jax.lax.stop_gradient(advantages)

        policy_loss = -jnp.mean(
            jnp.sum(rollout.log_probs * detached_advantages * rollout.masks, axis=0)
        )
        value_loss = jnp.mean(
            jnp.sum(((rollout.values - returns) ** 2) * rollout.masks, axis=0)
        )
        entropy_loss = -jnp.mean(
            jnp.sum(rollout.entropies * rollout.masks, axis=0)
        )

        loss = policy_loss + self.beta_v * value_loss + beta_e * entropy_loss

        metrics = StepMetrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            episode_reward=jnp.mean(jnp.sum(rollout.rewards, axis=0)),
            episode_length=jnp.mean(jnp.sum(rollout.masks, axis=0)),
            grad_norm=jnp.array(0.0, dtype=jnp.float32),
            param_norm=jnp.array(0.0, dtype=jnp.float32),
        )

        return loss, (metrics, new_key)

    def _optimizer_update(self, params: Any, grads: Any, optimizer: AdamState, grad_norm: jax.Array | None = None):
        if grad_norm is None:
            grad_norm = _global_norm(grads)
        clip_coef = jnp.minimum(1.0, self.max_grad_norm / (grad_norm + 1e-6))
        grads = jax.tree_util.tree_map(lambda g: g * clip_coef, grads)

        step = optimizer.step + 1

        m = jax.tree_util.tree_map(
            lambda m_val, g: self.adam_beta1 * m_val + (1.0 - self.adam_beta1) * g,
            optimizer.m,
            grads,
        )
        v = jax.tree_util.tree_map(
            lambda v_val, g: self.adam_beta2 * v_val + (1.0 - self.adam_beta2) * (g * g),
            optimizer.v,
            grads,
        )

        step_float = step.astype(jnp.float32)
        bias_correction1 = 1.0 - self.adam_beta1**step_float
        bias_correction2 = 1.0 - self.adam_beta2**step_float

        params = jax.tree_util.tree_map(
            lambda p, m_val, v_val: p
            - self.lr
            * (m_val / bias_correction1)
            / (jnp.sqrt(v_val / bias_correction2) + self.adam_eps),
            params,
            m,
            v,
        )

        return params, AdamState(step=step, m=m, v=v)

    def _train_step(self, state: JaxTrainState, beta_e: jax.Array):
        (loss, (metrics, new_key)), grads = jax.value_and_grad(
            self._loss_and_metrics,
            has_aux=True,
        )(state.params, state.rng_key, beta_e)

        del loss

        grad_norm = _global_norm(grads)
        param_norm = _global_norm(state.params)
        params, optimizer = self._optimizer_update(
            state.params,
            grads,
            state.optimizer,
            grad_norm=grad_norm,
        )
        metrics = metrics._replace(
            grad_norm=grad_norm,
            param_norm=param_norm,
        )
        new_state = JaxTrainState(params=params, optimizer=optimizer, rng_key=new_key)

        return new_state, metrics

    def _train_many(self, state: JaxTrainState, entropy_schedule: jax.Array):
        def body_fn(carry, beta_e):
            next_state, metrics = self._train_step(carry, beta_e)
            return next_state, metrics

        return jax.lax.scan(body_fn, state, entropy_schedule)

    def _train_many_mean_metrics(self, state: JaxTrainState, entropy_schedule: jax.Array):
        def body_fn(carry, beta_e):
            state, metric_sums = carry
            state, metrics = self._train_step(state, beta_e)
            metric_sums = jax.tree_util.tree_map(jnp.add, metric_sums, metrics)
            return (state, metric_sums), None

        init = (state, _zero_step_metrics(dtype=jnp.float32))
        (state, metric_sums), _ = jax.lax.scan(body_fn, init, entropy_schedule)
        num_steps = jnp.asarray(entropy_schedule.shape[0], dtype=jnp.float32)
        mean_metrics = jax.tree_util.tree_map(lambda x: x / num_steps, metric_sums)
        return state, mean_metrics

    def train_step(self, state: JaxTrainState, beta_e: float | None = None):
        if beta_e is None:
            beta_e = self.beta_e
        beta_e = jnp.asarray(beta_e, dtype=jnp.float32)

        return self._train_step_jit(state, beta_e)

    def train(self, state: JaxTrainState, num_updates: int, entropy_schedule=None):
        if entropy_schedule is None:
            entropy_schedule = np.full((num_updates,), self.beta_e, dtype=np.float32)
        else:
            entropy_schedule = np.asarray(entropy_schedule, dtype=np.float32)
            if entropy_schedule.shape[0] != num_updates:
                raise ValueError("entropy_schedule length must match num_updates")

        data = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "episode_length": [],
            "episode_reward": [],
            "grad_norm": [],
            "param_norm": [],
            "step_time_s": [],
            "cumulative_time_s": [],
        }

        start_time = time.perf_counter()
        for index in range(num_updates):
            step_start = time.perf_counter()
            state, metrics = self.train_step(state, beta_e=float(entropy_schedule[index]))
            step_time = time.perf_counter() - step_start

            data["loss"].append(float(metrics.loss))
            data["policy_loss"].append(float(metrics.policy_loss))
            data["value_loss"].append(float(metrics.value_loss))
            data["entropy_loss"].append(float(metrics.entropy_loss))
            data["episode_length"].append(float(metrics.episode_length))
            data["episode_reward"].append(float(metrics.episode_reward))
            data["grad_norm"].append(float(metrics.grad_norm))
            data["param_norm"].append(float(metrics.param_norm))
            data["step_time_s"].append(step_time)
            data["cumulative_time_s"].append(time.perf_counter() - start_time)

        return state, data

    def train_compiled(self, state: JaxTrainState, entropy_schedule):
        entropy_schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_many_jit(state, entropy_schedule)

    def train_compiled_mean_metrics(self, state: JaxTrainState, entropy_schedule):
        entropy_schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_many_mean_metrics_jit(state, entropy_schedule)
