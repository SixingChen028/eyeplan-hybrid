import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .a2c import AdamState, JaxTrainState, _global_norm, _select_not_done, _zeros_like_tree
from .environment import JaxDecisionTreeEnv
from .network import NETWORK_MLP, actor_critic_forward, apply_action_mask, init_actor_critic_params, sample_actions


class PPORolloutBatch(NamedTuple):
    obs: jax.Array
    action_mask: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    masks: jax.Array
    not_done_masks: jax.Array
    rewards: jax.Array
    values: jax.Array


class PPOStepMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy_loss: jax.Array
    clip_fraction: jax.Array
    approx_kl: jax.Array
    episode_reward: jax.Array
    episode_length: jax.Array


def _zero_ppo_step_metrics(dtype=jnp.float32):
    zero = jnp.array(0.0, dtype=dtype)
    return PPOStepMetrics(
        loss=zero,
        policy_loss=zero,
        value_loss=zero,
        entropy_loss=zero,
        clip_fraction=zero,
        approx_kl=zero,
        episode_reward=zero,
        episode_length=zero,
    )


class JaxBatchMaskPPO:
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
        rollout_steps: int | None = None,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        normalize_advantages: bool = True,
        max_grad_norm: float = 1.0,
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
        self.rollout_steps = int(self.env.t_max if rollout_steps is None else rollout_steps)
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.lamda = float(lamda)
        self.beta_v = float(beta_v)
        self.beta_e = float(beta_e)
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.normalize_advantages = bool(normalize_advantages)
        self.max_grad_norm = float(max_grad_norm)
        self.network_type = str(network_type)

        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)

        self._train_step_jit = jax.jit(self._train_step)
        self._train_many_jit = jax.jit(self._train_many)
        self._train_many_mean_metrics_jit = jax.jit(self._train_many_mean_metrics)

    def init_state(self, seed: int = 0) -> JaxTrainState:
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

        return JaxTrainState(params=params, optimizer=optimizer, rng_key=key)

    def _rollout(self, params: Any, rng_key: jax.Array):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, self.batch_size)

        env_state, obs, info = jax.vmap(self.env.reset)(reset_keys)
        action_mask = info["mask"]
        one_mask = jnp.ones((self.batch_size,), dtype=jnp.float32)

        def body_fn(carry, _):
            env_state, obs, action_mask, rng_key = carry

            logits, values = actor_critic_forward(params, obs, action_mask)

            rng_key, action_key, reset_key = jax.random.split(rng_key, 3)
            actions, log_probs, _ = sample_actions(action_key, logits, action_mask)

            next_env_state, next_obs, rewards, dones, _, info = jax.vmap(self.env.step)(env_state, actions)
            next_action_mask = info["mask"]

            reset_keys = jax.random.split(reset_key, self.batch_size)
            reset_env_state, reset_obs, reset_info = jax.vmap(self.env.reset)(reset_keys)
            reset_action_mask = reset_info["mask"]

            env_state = jax.tree_util.tree_map(
                lambda stepped, reset: _select_not_done(dones, stepped, reset),
                next_env_state,
                reset_env_state,
            )
            next_obs = _select_not_done(dones, next_obs, reset_obs)
            next_action_mask = _select_not_done(dones, next_action_mask, reset_action_mask)

            output = PPORolloutBatch(
                obs=obs,
                action_mask=action_mask,
                actions=actions,
                old_log_probs=log_probs,
                masks=one_mask,
                not_done_masks=1.0 - dones.astype(jnp.float32),
                rewards=rewards.astype(jnp.float32),
                values=values,
            )

            return (env_state, next_obs, next_action_mask, rng_key), output

        carry, rollout = jax.lax.scan(
            body_fn,
            (env_state, obs, action_mask, rng_key),
            xs=None,
            length=self.rollout_steps,
        )

        _, final_obs, final_action_mask, new_key = carry
        _, bootstrap_values = actor_critic_forward(params, final_obs, final_action_mask)

        episode_reward = jnp.mean(jnp.sum(rollout.rewards, axis=0))
        episode_length = jnp.mean(jnp.sum(rollout.not_done_masks, axis=0))

        return rollout, bootstrap_values, episode_reward, episode_length, new_key

    def _discounted_returns_and_advantages(
        self,
        rewards: jax.Array,
        values: jax.Array,
        not_done_masks: jax.Array,
        bootstrap_values: jax.Array,
        masks: jax.Array,
    ):
        next_values = jnp.concatenate([values[1:], bootstrap_values[None, :]], axis=0)
        rewards_rev = rewards[::-1]
        values_rev = values[::-1]
        next_values_rev = next_values[::-1]
        not_done_masks_rev = not_done_masks[::-1]

        def body_fn(carry, xs):
            R, advantage = carry
            reward, value, value_next, not_done_mask = xs

            R = reward + self.gamma * R * not_done_mask
            delta = reward + self.gamma * value_next * not_done_mask - value
            advantage = delta + self.gamma * self.lamda * advantage * not_done_mask

            return (R, advantage), (R, advantage)

        init = (
            jnp.zeros((rewards.shape[1],), dtype=rewards.dtype),
            jnp.zeros((rewards.shape[1],), dtype=rewards.dtype),
        )

        (_, _), (returns_rev, advantages_rev) = jax.lax.scan(
            body_fn,
            init,
            (rewards_rev, values_rev, next_values_rev, not_done_masks_rev),
        )

        returns = returns_rev[::-1]
        advantages = advantages_rev[::-1]

        if self.normalize_advantages:
            adv_flat = advantages.reshape(-1)
            valid = masks.reshape(-1)
            valid_count = jnp.maximum(jnp.sum(valid), 1.0)
            mean = jnp.sum(adv_flat * valid) / valid_count
            var = jnp.sum(((adv_flat - mean) ** 2) * valid) / valid_count
            advantages = (advantages - mean) / jnp.sqrt(var + 1e-8)

        return returns, advantages

    def _optimizer_update(self, params: Any, grads: Any, optimizer: AdamState):
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

    def _ppo_loss(self, params: Any, data: dict[str, jax.Array], beta_e: jax.Array):
        obs = data["obs"]
        action_mask = data["action_mask"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        masks = data["masks"]
        returns = data["returns"]
        advantages = data["advantages"]

        logits, values = actor_critic_forward(params, obs, action_mask)
        masked_logits = apply_action_mask(logits, action_mask)

        log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
        probs_all = jax.nn.softmax(masked_logits, axis=-1)
        new_log_probs = log_probs_all[jnp.arange(log_probs_all.shape[0]), actions]
        entropy = -jnp.sum(jnp.where(action_mask, probs_all * log_probs_all, 0.0), axis=-1)

        ratio = jnp.exp(new_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        pg_loss_unclipped = ratio * advantages
        pg_loss_clipped = clipped_ratio * advantages
        policy_loss = -jnp.sum(jnp.minimum(pg_loss_unclipped, pg_loss_clipped) * masks) / jnp.maximum(jnp.sum(masks), 1.0)

        value_loss = jnp.sum(((values - returns) ** 2) * masks) / jnp.maximum(jnp.sum(masks), 1.0)
        entropy_loss = -jnp.sum(entropy * masks) / jnp.maximum(jnp.sum(masks), 1.0)

        loss = policy_loss + self.beta_v * value_loss + beta_e * entropy_loss

        clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > self.clip_eps).astype(jnp.float32) * masks) / jnp.maximum(jnp.sum(masks), 1.0)
        approx_kl = jnp.sum((old_log_probs - new_log_probs) * masks) / jnp.maximum(jnp.sum(masks), 1.0)

        return loss, (policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl)

    def _ppo_epoch_update(self, state: JaxTrainState, data: dict[str, jax.Array], beta_e: jax.Array):
        (loss, aux), grads = jax.value_and_grad(self._ppo_loss, has_aux=True)(
            state.params,
            data,
            beta_e,
        )

        params, optimizer = self._optimizer_update(state.params, grads, state.optimizer)
        next_state = JaxTrainState(params=params, optimizer=optimizer, rng_key=state.rng_key)

        policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl = aux

        return next_state, jnp.array([loss, policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl])

    def _train_step(self, state: JaxTrainState, beta_e: jax.Array):
        rollout, bootstrap_values, episode_reward, episode_length, new_key = self._rollout(state.params, state.rng_key)

        returns, advantages = self._discounted_returns_and_advantages(
            rollout.rewards,
            rollout.values,
            rollout.not_done_masks,
            bootstrap_values,
            rollout.masks,
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

        work_state = JaxTrainState(params=state.params, optimizer=state.optimizer, rng_key=new_key)

        def epoch_body(carry, _):
            next_state, metrics_vec = self._ppo_epoch_update(carry, flat, beta_e)
            return next_state, metrics_vec

        work_state, metrics_by_epoch = jax.lax.scan(
            epoch_body,
            work_state,
            xs=None,
            length=self.ppo_epochs,
        )

        metrics_mean = jnp.mean(metrics_by_epoch, axis=0)

        metrics = PPOStepMetrics(
            loss=metrics_mean[0],
            policy_loss=metrics_mean[1],
            value_loss=metrics_mean[2],
            entropy_loss=metrics_mean[3],
            clip_fraction=metrics_mean[4],
            approx_kl=metrics_mean[5],
            episode_reward=episode_reward,
            episode_length=episode_length,
        )

        return work_state, metrics

    def train_step(self, state: JaxTrainState, beta_e: float | None = None):
        if beta_e is None:
            beta_e = self.beta_e
        beta_e = jnp.asarray(beta_e, dtype=jnp.float32)

        return self._train_step_jit(state, beta_e)

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

        init = (state, _zero_ppo_step_metrics(dtype=jnp.float32))
        (state, metric_sums), _ = jax.lax.scan(body_fn, init, entropy_schedule)
        num_steps = jnp.asarray(entropy_schedule.shape[0], dtype=jnp.float32)
        mean_metrics = jax.tree_util.tree_map(lambda x: x / num_steps, metric_sums)
        return state, mean_metrics

    def train_compiled(self, state: JaxTrainState, entropy_schedule):
        entropy_schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_many_jit(state, entropy_schedule)

    def train_compiled_mean_metrics(self, state: JaxTrainState, entropy_schedule):
        entropy_schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_many_mean_metrics_jit(state, entropy_schedule)

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
            "clip_fraction": [],
            "approx_kl": [],
            "episode_length": [],
            "episode_reward": [],
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
            data["clip_fraction"].append(float(metrics.clip_fraction))
            data["approx_kl"].append(float(metrics.approx_kl))
            data["episode_length"].append(float(metrics.episode_length))
            data["episode_reward"].append(float(metrics.episode_reward))
            data["step_time_s"].append(step_time)
            data["cumulative_time_s"].append(time.perf_counter() - start_time)

        return state, data
