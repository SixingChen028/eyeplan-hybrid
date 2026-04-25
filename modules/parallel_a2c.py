from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .a2c import (
    AdamState,
    JaxTrainState,
    RolloutBatch,
    StepMetrics,
    _global_norm,
    _select_not_done,
    _zeros_like_tree,
)
from .environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from .network import actor_critic_forward, init_mlp_actor_critic_params, sample_actions


class A2CHyperParams(NamedTuple):
    env: JaxDecisionTreeParams
    lr: jax.Array
    gamma: jax.Array
    lamda: jax.Array
    beta_v: jax.Array
    beta_e_init: jax.Array
    beta_e_final: jax.Array
    max_grad_norm: jax.Array


class ParallelA2CResult(NamedTuple):
    states: JaxTrainState
    metrics: StepMetrics


class ParallelJaxBatchMaskA2C:
    def __init__(
        self,
        env: JaxDecisionTreeEnv,
        feature_size: int,
        action_size: int,
        hidden_size: int,
        batch_size: int,
        num_updates: int,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        self.env = env
        self.feature_size = int(feature_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = int(batch_size)
        self.num_updates = int(num_updates)

        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)

        self._train_one_jit = jax.jit(self._train_one)
        self._train_sweep_jit = jax.jit(self._train_sweep)

    def init_state(self, seed: jax.Array) -> JaxTrainState:
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

    def _rollout(self, params: Any, rng_key: jax.Array, env_params: JaxDecisionTreeParams):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, self.batch_size)

        env_state, obs, info = jax.vmap(self.env.reset_with_params, in_axes=(0, None))(
            reset_keys,
            env_params,
        )
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

                next_env_state, next_obs, rewards, dones, _, info = jax.vmap(
                    self.env.step_with_params,
                    in_axes=(0, 0, None),
                )(env_state, actions, env_params)
                next_action_mask = info["mask"]

                env_state = jax.tree_util.tree_map(
                    lambda new, old: _select_not_done(done, new, old),
                    next_env_state,
                    env_state,
                )
                obs = _select_not_done(done, next_obs, obs)
                action_mask = _select_not_done(done, next_action_mask, action_mask)
                done = jnp.logical_or(done, dones)

                output = RolloutBatch(
                    masks=mask,
                    rewards=rewards.astype(jnp.float32) * mask,
                    log_probs=log_probs * mask,
                    entropies=entropies * mask,
                    values=values * mask,
                )
                return (env_state, obs, action_mask, done, rng_key), output

            return jax.lax.cond(jnp.all(done), done_branch, active_branch, carry)

        carry, rollout = jax.lax.scan(
            body_fn,
            (env_state, obs, action_mask, done, rng_key),
            xs=None,
            length=self.env.t_max,
        )

        return rollout, carry[4]

    def _discounted_returns_and_advantages(
        self,
        rewards: jax.Array,
        values: jax.Array,
        hyper: A2CHyperParams,
    ):
        batch_size = rewards.shape[1]
        padded_values = jnp.concatenate(
            [values, jnp.zeros((1, batch_size), dtype=values.dtype)],
            axis=0,
        )

        def body_fn(carry, xs):
            running_return, advantage = carry
            reward, value, value_next = xs

            running_return = reward + hyper.gamma * running_return
            delta = reward + hyper.gamma * value_next - value
            advantage = delta + hyper.gamma * hyper.lamda * advantage
            return (running_return, advantage), (running_return, advantage)

        init = (
            jnp.zeros((batch_size,), dtype=rewards.dtype),
            jnp.zeros((batch_size,), dtype=rewards.dtype),
        )
        (_, _), (returns_rev, advantages_rev) = jax.lax.scan(
            body_fn,
            init,
            (rewards[::-1], padded_values[:-1][::-1], padded_values[1:][::-1]),
        )
        return returns_rev[::-1], advantages_rev[::-1]

    def _loss_and_metrics(
        self,
        params: Any,
        rng_key: jax.Array,
        beta_e: jax.Array,
        hyper: A2CHyperParams,
    ):
        rollout, new_key = self._rollout(params, rng_key, hyper.env)
        returns, advantages = self._discounted_returns_and_advantages(
            rewards=rollout.rewards,
            values=rollout.values,
            hyper=hyper,
        )

        detached_advantages = jax.lax.stop_gradient(advantages)
        policy_loss = -jnp.mean(
            jnp.sum(rollout.log_probs * detached_advantages * rollout.masks, axis=0)
        )
        value_loss = jnp.mean(
            jnp.sum(((rollout.values - returns) ** 2) * rollout.masks, axis=0)
        )
        entropy_loss = -jnp.mean(jnp.sum(rollout.entropies * rollout.masks, axis=0))
        loss = policy_loss + hyper.beta_v * value_loss + beta_e * entropy_loss

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

    def _optimizer_update(
        self,
        params: Any,
        grads: Any,
        optimizer: AdamState,
        hyper: A2CHyperParams,
        grad_norm: jax.Array,
    ):
        clip_coef = jnp.minimum(1.0, hyper.max_grad_norm / (grad_norm + 1e-6))
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
            - hyper.lr
            * (m_val / bias_correction1)
            / (jnp.sqrt(v_val / bias_correction2) + self.adam_eps),
            params,
            m,
            v,
        )

        return params, AdamState(step=step, m=m, v=v)

    def _train_step(self, state: JaxTrainState, beta_e: jax.Array, hyper: A2CHyperParams):
        (loss, (metrics, new_key)), grads = jax.value_and_grad(
            self._loss_and_metrics,
            has_aux=True,
        )(state.params, state.rng_key, beta_e, hyper)
        del loss

        grad_norm = _global_norm(grads)
        param_norm = _global_norm(state.params)
        params, optimizer = self._optimizer_update(
            state.params,
            grads,
            state.optimizer,
            hyper,
            grad_norm=grad_norm,
        )
        metrics = metrics._replace(grad_norm=grad_norm, param_norm=param_norm)
        return JaxTrainState(params=params, optimizer=optimizer, rng_key=new_key), metrics

    def _train_one(self, hyper: A2CHyperParams, seed: jax.Array):
        state = self.init_state(seed)
        entropy_schedule = jnp.linspace(
            hyper.beta_e_init,
            hyper.beta_e_final,
            self.num_updates,
            dtype=jnp.float32,
        )

        def body_fn(carry, beta_e):
            return self._train_step(carry, beta_e, hyper)

        state, metrics = jax.lax.scan(body_fn, state, entropy_schedule)
        return state, metrics

    def _train_sweep(self, hypers: A2CHyperParams, seeds: jax.Array):
        train_seeds = jax.vmap(self._train_one, in_axes=(None, 0))
        train_hypers = jax.vmap(train_seeds, in_axes=(0, None))
        states, metrics = train_hypers(hypers, seeds)
        return ParallelA2CResult(states=states, metrics=metrics)

    def train_one(self, hyper: A2CHyperParams, seed: int):
        return self._train_one_jit(hyper, jnp.asarray(seed, dtype=jnp.int32))

    def train_sweep(self, hypers: A2CHyperParams, seeds):
        seeds = jnp.asarray(seeds, dtype=jnp.int32)
        return self._train_sweep_jit(hypers, seeds)
