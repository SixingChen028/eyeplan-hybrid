from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .a2c import AdamState, JaxTrainState, _global_norm, _select_not_done, _zeros_like_tree
from .environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from .network import NETWORK_MLP, actor_critic_forward, apply_action_mask, init_actor_critic_params, sample_actions
from .ppo import PPORolloutBatch, PPOStepMetrics


class PPOHyperParams(NamedTuple):
    env: JaxDecisionTreeParams
    lr: jax.Array
    gamma: jax.Array
    lamda: jax.Array
    beta_v: jax.Array
    beta_e_init: jax.Array
    beta_e_final: jax.Array
    max_grad_norm: jax.Array


class ParallelPPOResult(NamedTuple):
    states: JaxTrainState
    metrics: PPOStepMetrics


class ParallelJaxBatchMaskPPO:
    def __init__(
        self,
        env: JaxDecisionTreeEnv,
        feature_size: int,
        action_size: int,
        hidden_size: int,
        batch_size: int,
        num_updates: int,
        rollout_length: int | None = None,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        normalize_advantages: bool = True,
        network_type: str = NETWORK_MLP,
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
        self.rollout_length = int(self.env.t_max if rollout_length is None else rollout_length)
        if self.rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.normalize_advantages = bool(normalize_advantages)
        self.network_type = str(network_type)

        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)

        self._train_one_jit = jax.jit(self._train_one)
        self._train_sweep_jit = jax.jit(self._train_sweep)
        self._init_sweep_states_jit = jax.jit(self._init_sweep_states)
        self._train_sweep_chunk_jit = jax.jit(self._train_sweep_chunk)

    def init_state(self, seed: jax.Array) -> JaxTrainState:
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        params = init_actor_critic_params(
            init_key,
            feature_size=self.feature_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            network_type=self.network_type,
        )
        optimizer = AdamState(
            step=jnp.array(0, dtype=jnp.int32),
            m=_zeros_like_tree(params),
            v=_zeros_like_tree(params),
        )
        return JaxTrainState(params=params, optimizer=optimizer, rollout_state=None, rng_key=key)

    def _rollout(self, params: Any, rng_key: jax.Array, env_params: JaxDecisionTreeParams):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, self.batch_size)
        env_state, obs, info = jax.vmap(self.env.reset_with_params, in_axes=(0, None))(reset_keys, env_params)
        action_mask = info["mask"]
        done = jnp.zeros((self.batch_size,), dtype=jnp.bool_)
        one_mask = jnp.ones((self.batch_size,), dtype=jnp.float32)
        zero_mask = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        reward_sum = jnp.array(0.0, dtype=jnp.float32)
        length_sum = jnp.array(0.0, dtype=jnp.float32)

        def body_fn(carry, _):
            env_state, obs, action_mask, done, reward_sum, length_sum, rng_key = carry

            def done_branch(c):
                env_state, obs, action_mask, done, reward_sum, length_sum, rng_key = c
                rng_key, _ = jax.random.split(rng_key)
                output = PPORolloutBatch(
                    obs=jnp.zeros((self.batch_size, self.feature_size), dtype=obs.dtype),
                    action_mask=jnp.zeros((self.batch_size, self.action_size), dtype=jnp.bool_),
                    actions=jnp.zeros((self.batch_size,), dtype=jnp.int32),
                    old_log_probs=zero_mask,
                    masks=zero_mask,
                    not_done_masks=zero_mask,
                    rewards=zero_mask,
                    values=zero_mask,
                )
                return (env_state, obs, action_mask, done, reward_sum, length_sum, rng_key), output

            def active_branch(c):
                env_state, obs, action_mask, done, reward_sum, length_sum, rng_key = c
                mask = 1.0 - done.astype(jnp.float32)
                logits, values = actor_critic_forward(params, obs, action_mask)
                rng_key, action_key = jax.random.split(rng_key)
                actions, log_probs, _ = sample_actions(action_key, logits, action_mask)
                next_env_state, next_obs, rewards, dones, _, info = jax.vmap(
                    self.env.step_with_params,
                    in_axes=(0, 0, None),
                )(env_state, actions, env_params)
                next_action_mask = info["mask"]
                reward_sum = reward_sum + jnp.sum(rewards.astype(jnp.float32) * mask)
                length_sum = length_sum + jnp.sum(mask)
                env_state = jax.tree_util.tree_map(
                    lambda new, old: _select_not_done(done, new, old),
                    next_env_state,
                    env_state,
                )
                obs = _select_not_done(done, next_obs, obs)
                action_mask = _select_not_done(done, next_action_mask, action_mask)
                next_done = jnp.logical_or(done, dones)
                output = PPORolloutBatch(
                    obs=obs,
                    action_mask=action_mask,
                    actions=actions,
                    old_log_probs=log_probs * mask,
                    masks=mask,
                    not_done_masks=(1.0 - next_done.astype(jnp.float32)) * one_mask,
                    rewards=rewards.astype(jnp.float32) * mask,
                    values=values * mask,
                )
                return (env_state, obs, action_mask, next_done, reward_sum, length_sum, rng_key), output

            return jax.lax.cond(jnp.all(done), done_branch, active_branch, carry)

        carry, rollout = jax.lax.scan(
            body_fn,
            (env_state, obs, action_mask, done, reward_sum, length_sum, rng_key),
            xs=None,
            length=self.rollout_length,
        )
        final_env_state, final_obs, final_action_mask, _, reward_sum, length_sum, rng_key = carry
        _, bootstrap_values = actor_critic_forward(params, final_obs, final_action_mask)
        return rollout, bootstrap_values, reward_sum, length_sum, rng_key

    def _discounted_returns_and_advantages(
        self,
        rewards: jax.Array,
        values: jax.Array,
        not_done_masks: jax.Array,
        bootstrap_values: jax.Array,
        masks: jax.Array,
        hyper: PPOHyperParams,
    ):
        next_values = jnp.concatenate([values[1:], bootstrap_values[None, :]], axis=0)

        def body_fn(carry, xs):
            returns, advantage = carry
            reward, value, value_next, not_done_mask = xs
            returns = reward + hyper.gamma * returns * not_done_mask
            delta = reward + hyper.gamma * value_next * not_done_mask - value
            advantage = delta + hyper.gamma * hyper.lamda * advantage * not_done_mask
            return (returns, advantage), (returns, advantage)

        init = (
            jnp.zeros((rewards.shape[1],), dtype=rewards.dtype),
            jnp.zeros((rewards.shape[1],), dtype=rewards.dtype),
        )
        (_, _), (returns_rev, advantages_rev) = jax.lax.scan(
            body_fn,
            init,
            (rewards[::-1], values[::-1], next_values[::-1], not_done_masks[::-1]),
        )
        returns = returns_rev[::-1]
        advantages = advantages_rev[::-1]
        if self.normalize_advantages:
            adv_flat = advantages.reshape((-1,))
            valid = masks.reshape((-1,))
            valid_count = jnp.maximum(jnp.sum(valid), 1.0)
            mean = jnp.sum(adv_flat * valid) / valid_count
            var = jnp.sum(((adv_flat - mean) ** 2) * valid) / valid_count
            advantages = (advantages - mean) / jnp.sqrt(var + 1e-8)
        return returns, advantages

    def _optimizer_update(self, params: Any, grads: Any, optimizer: AdamState, hyper: PPOHyperParams):
        grad_norm = _global_norm(grads)
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

    def _ppo_loss(self, params: Any, data: dict[str, jax.Array], beta_e: jax.Array, hyper: PPOHyperParams):
        logits, values = actor_critic_forward(params, data["obs"], data["action_mask"])
        masked_logits = apply_action_mask(logits, data["action_mask"])
        log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
        probs_all = jax.nn.softmax(masked_logits, axis=-1)
        new_log_probs = log_probs_all[jnp.arange(log_probs_all.shape[0]), data["actions"]]
        entropy = -jnp.sum(jnp.where(data["action_mask"], probs_all * log_probs_all, 0.0), axis=-1)
        ratio = jnp.exp(new_log_probs - data["old_log_probs"])
        clipped_ratio = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        pg_loss_unclipped = ratio * data["advantages"]
        pg_loss_clipped = clipped_ratio * data["advantages"]
        denom = jnp.maximum(jnp.sum(data["masks"]), 1.0)
        policy_loss = -jnp.sum(jnp.minimum(pg_loss_unclipped, pg_loss_clipped) * data["masks"]) / denom
        value_loss = jnp.sum(((values - data["returns"]) ** 2) * data["masks"]) / denom
        entropy_loss = -jnp.sum(entropy * data["masks"]) / denom
        loss = policy_loss + hyper.beta_v * value_loss + beta_e * entropy_loss
        clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > self.clip_eps).astype(jnp.float32) * data["masks"]) / denom
        approx_kl = jnp.sum((data["old_log_probs"] - new_log_probs) * data["masks"]) / denom
        return loss, (policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl)

    def _ppo_epoch_update(self, state: JaxTrainState, data: dict[str, jax.Array], beta_e: jax.Array, hyper: PPOHyperParams):
        (loss, aux), grads = jax.value_and_grad(self._ppo_loss, has_aux=True)(state.params, data, beta_e, hyper)
        del loss
        params, optimizer = self._optimizer_update(state.params, grads, state.optimizer, hyper)
        next_state = JaxTrainState(params=params, optimizer=optimizer, rollout_state=None, rng_key=state.rng_key)
        policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl = aux
        return next_state, jnp.array([policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl])

    def _train_step(self, state: JaxTrainState, beta_e: jax.Array, hyper: PPOHyperParams):
        rollout, bootstrap_values, reward_sum, length_sum, new_key = self._rollout(state.params, state.rng_key, hyper.env)
        returns, advantages = self._discounted_returns_and_advantages(
            rollout.rewards,
            rollout.values,
            rollout.not_done_masks,
            bootstrap_values,
            rollout.masks,
            hyper,
        )
        flat = {
            "obs": rollout.obs.reshape((-1, self.feature_size)),
            "action_mask": rollout.action_mask.reshape((-1, self.action_size)),
            "actions": rollout.actions.reshape((-1,)),
            "old_log_probs": rollout.old_log_probs.reshape((-1,)),
            "masks": rollout.masks.reshape((-1,)),
            "returns": returns.reshape((-1,)),
            "advantages": advantages.reshape((-1,)),
        }
        work_state = JaxTrainState(
            params=state.params,
            optimizer=state.optimizer,
            rollout_state=None,
            rng_key=new_key,
        )

        def epoch_body(carry, _):
            return self._ppo_epoch_update(carry, flat, beta_e, hyper)

        work_state, metrics_by_epoch = jax.lax.scan(epoch_body, work_state, xs=None, length=self.ppo_epochs)
        metrics_mean = jnp.mean(metrics_by_epoch, axis=0)
        episode_count = jnp.asarray(self.batch_size, dtype=jnp.float32)
        metrics = PPOStepMetrics(
            loss=metrics_mean[0] + hyper.beta_v * metrics_mean[1] + beta_e * metrics_mean[2],
            policy_loss=metrics_mean[0],
            value_loss=metrics_mean[1],
            entropy_loss=metrics_mean[2],
            clip_fraction=metrics_mean[3],
            approx_kl=metrics_mean[4],
            episode_reward=reward_sum / episode_count,
            episode_length=length_sum / episode_count,
            episode_count=episode_count,
            episode_reward_sum=reward_sum,
            episode_length_sum=length_sum,
        )
        return work_state, metrics

    def _train_one(self, hyper: PPOHyperParams, seed: jax.Array):
        state = self.init_state(seed)
        entropy_schedule = jnp.linspace(hyper.beta_e_init, hyper.beta_e_final, self.num_updates, dtype=jnp.float32)

        def body_fn(carry, beta_e):
            return self._train_step(carry, beta_e, hyper)

        return jax.lax.scan(body_fn, state, entropy_schedule)

    def _train_one_from_state(self, state: JaxTrainState, hyper: PPOHyperParams, entropy_schedule: jax.Array):
        def body_fn(carry, beta_e):
            return self._train_step(carry, beta_e, hyper)

        return jax.lax.scan(body_fn, state, entropy_schedule)

    def _train_sweep(self, hypers: PPOHyperParams, seeds: jax.Array):
        train_seeds = jax.vmap(self._train_one, in_axes=(None, 0))
        train_hypers = jax.vmap(train_seeds, in_axes=(0, None))
        states, metrics = train_hypers(hypers, seeds)
        return ParallelPPOResult(states=states, metrics=metrics)

    def _init_sweep_states(self, hypers: PPOHyperParams, seeds: jax.Array):
        def init_hyper(_):
            return jax.vmap(self.init_state)(seeds)

        return jax.vmap(init_hyper)(jnp.arange(hypers.lr.shape[0]))

    def _train_sweep_chunk(self, states: JaxTrainState, hypers: PPOHyperParams, entropy_schedule: jax.Array):
        train_seeds = jax.vmap(self._train_one_from_state, in_axes=(0, None, None))
        train_hypers = jax.vmap(train_seeds, in_axes=(0, 0, 0))
        states, metrics = train_hypers(states, hypers, entropy_schedule)
        return ParallelPPOResult(states=states, metrics=metrics)

    def train_one(self, hyper: PPOHyperParams, seed: int):
        return self._train_one_jit(hyper, jnp.asarray(seed, dtype=jnp.int32))

    def train_sweep(self, hypers: PPOHyperParams, seeds):
        return self._train_sweep_jit(hypers, jnp.asarray(seeds, dtype=jnp.int32))

    def init_sweep_states(self, hypers: PPOHyperParams, seeds):
        return self._init_sweep_states_jit(hypers, jnp.asarray(seeds, dtype=jnp.int32))

    def train_sweep_chunk(self, states: JaxTrainState, hypers: PPOHyperParams, entropy_schedule):
        schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_sweep_chunk_jit(states, hypers, schedule)

    def compile_train_sweep_chunk(self, states: JaxTrainState, hypers: PPOHyperParams, entropy_schedule) -> None:
        schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        self._train_sweep_chunk_jit.lower(states, hypers, schedule).compile()
