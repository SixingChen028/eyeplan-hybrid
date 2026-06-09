import os
import pickle
import time
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from .compat import (
    COMPAT_KEY,
    COMPAT_VERSION,
    PARAMS_FORMAT_VERSION,
    assert_compat_version,
    read_compat_version,
)
from .network import NETWORK_MLP, actor_critic_forward, init_actor_critic_params, sample_actions


class AdamState(NamedTuple):
    step: jax.Array
    m: Any
    v: Any


class JaxTrainState(NamedTuple):
    params: Any
    optimizer: AdamState
    rollout_state: Any
    rng_key: jax.Array


class A2CTrainParams(NamedTuple):
    env: JaxDecisionTreeParams
    lr: jax.Array
    gamma: jax.Array
    lamda: jax.Array
    beta_v: jax.Array
    max_grad_norm: jax.Array


class StepMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy_loss: jax.Array
    episode_reward: jax.Array
    episode_length: jax.Array
    episode_count: jax.Array
    episode_reward_sum: jax.Array
    episode_length_sum: jax.Array
    grad_norm: jax.Array
    param_norm: jax.Array


class RolloutBatch(NamedTuple):
    not_done_masks: jax.Array
    rewards: jax.Array
    log_probs: jax.Array
    entropies: jax.Array
    values: jax.Array


class RolloutState(NamedTuple):
    env_state: Any
    obs: jax.Array
    action_mask: jax.Array
    observation_mask: jax.Array
    running_return: jax.Array
    running_length: jax.Array


def _zero_step_metrics(dtype=jnp.float32):
    zero = jnp.array(0.0, dtype=dtype)
    return StepMetrics(
        loss=zero,
        policy_loss=zero,
        value_loss=zero,
        entropy_loss=zero,
        episode_reward=zero,
        episode_length=zero,
        episode_count=zero,
        episode_reward_sum=zero,
        episode_length_sum=zero,
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


def _is_params_payload(tree: Any) -> bool:
    return (
        isinstance(tree, dict)
        and "params_format_version" in tree
        and COMPAT_KEY in tree
        and "params" in tree
    )


def save_jax_params(params: Any, path: str):
    payload = {
        "params_format_version": PARAMS_FORMAT_VERSION,
        COMPAT_KEY: COMPAT_VERSION,
        "params": _tree_to_numpy(params),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def load_jax_params(
    path: str,
    *,
    allow_unversioned: bool = False,
    allow_compat_mismatch: bool = False,
    expected_compat_version=None,
):
    with open(path, "rb") as file:
        tree = pickle.load(file)

    if not _is_params_payload(tree):
        if not allow_unversioned:
            raise ValueError(
                f"Model params are missing compatibility metadata: {path}. "
                "Pass --allow-unversioned-params only for legacy runs."
            )
        return _tree_from_numpy(tree)

    params_format_version = int(tree["params_format_version"])
    if params_format_version != PARAMS_FORMAT_VERSION:
        raise ValueError(
            "Unsupported params format version: "
            f"{params_format_version}; expected {PARAMS_FORMAT_VERSION}."
        )

    recorded_version = read_compat_version(tree, source=path)
    if expected_compat_version is not None:
        expected_version = int(expected_compat_version)
        if int(recorded_version) != expected_version:
            raise ValueError(
                "Compatibility version mismatch: "
                f"{path} has {int(recorded_version)}, run metadata has {expected_version}."
            )
    if not allow_compat_mismatch:
        assert_compat_version(recorded_version, source=path)
    return _tree_from_numpy(tree["params"])


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def _select_reset_on_done(done: jax.Array, stepped: jax.Array, reset: jax.Array):
    if stepped is None or reset is None:
        return None
    selector = done
    while selector.ndim < stepped.ndim:
        selector = selector[..., None]
    return jnp.where(selector, reset, stepped)


class JaxBatchMaskA2C:
    def __init__(
        self,
        env: JaxDecisionTreeEnv,
        action_size: int,
        hidden_size: int,
        num_envs: int,
        lr: float,
        gamma: float,
        lamda: float,
        beta_v: float,
        beta_e: float,
        rollout_length: int | None = None,
        max_grad_norm: float = 1.0,
        network_type: str = NETWORK_MLP,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        self.env = env
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)

        self.num_envs = int(num_envs)
        self.rollout_length = int(self.env.t_max if rollout_length is None else rollout_length)
        if self.rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.lamda = float(lamda)
        self.beta_v = float(beta_v)
        self.beta_e = float(beta_e)
        self.max_grad_norm = float(max_grad_norm)
        self.network_type = str(network_type)

        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)

        self._train_step_jit = jax.jit(self._train_step)
        self._train_many_jit = jax.jit(self._train_many)
        self._train_many_mean_metrics_jit = jax.jit(self._train_many_mean_metrics)

    def init_state(self, seed: int, env_params: JaxDecisionTreeParams) -> JaxTrainState:
        return self.init_state_with_params(seed, env_params)

    def init_state_with_params(self, seed: int, env_params: JaxDecisionTreeParams) -> JaxTrainState:
        key = jax.random.PRNGKey(seed)
        key, init_key, reset_key = jax.random.split(key, 3)

        params = init_actor_critic_params(
            init_key,
            observation_template=self.env.observation_template,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            network_type=self.network_type,
        )

        optimizer = AdamState(
            step=jnp.array(0, dtype=jnp.int32),
            m=_zeros_like_tree(params),
            v=_zeros_like_tree(params),
        )
        reset_keys = jax.random.split(reset_key, self.num_envs)
        env_state, obs, info = jax.vmap(self.env.reset, in_axes=(0, None))(reset_keys, env_params)
        rollout_state = RolloutState(
            env_state=env_state,
            obs=obs,
            action_mask=info["mask"],
            observation_mask=info["observation_mask"],
            running_return=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            running_length=jnp.zeros((self.num_envs,), dtype=jnp.float32),
        )

        return JaxTrainState(params=params, optimizer=optimizer, rollout_state=rollout_state, rng_key=key)

    def _rollout(
        self,
        params: Any,
        rollout_state: RolloutState,
        rng_key: jax.Array,
        train_params: A2CTrainParams,
    ):
        env_state = rollout_state.env_state
        obs = rollout_state.obs
        action_mask = rollout_state.action_mask
        observation_mask = rollout_state.observation_mask
        running_return = rollout_state.running_return
        running_length = rollout_state.running_length
        one_mask = jnp.ones((self.num_envs,), dtype=jnp.float32)
        zeros = jnp.zeros((self.num_envs,), dtype=jnp.float32)

        def body_fn(carry, _):
            (
                env_state,
                obs,
                action_mask,
                observation_mask,
                running_return,
                running_length,
                episode_reward_sum,
                episode_length_sum,
                episode_count,
                rng_key,
            ) = carry

            logits, values = actor_critic_forward(params, obs, action_mask, observation_mask)

            rng_key, action_key, reset_key = jax.random.split(rng_key, 3)
            actions, log_probs, entropies = sample_actions(action_key, logits, action_mask)

            next_env_state, next_obs, rewards, dones, info = jax.vmap(
                self.env.step,
                in_axes=(0, 0, None),
            )(env_state, actions, train_params.env)
            next_action_mask = info["mask"]
            next_observation_mask = info["observation_mask"]

            reset_keys = jax.random.split(reset_key, self.num_envs)
            reset_env_state, reset_obs, reset_info = jax.vmap(
                self.env.reset,
                in_axes=(0, None),
            )(reset_keys, train_params.env)
            reset_action_mask = reset_info["mask"]
            reset_observation_mask = reset_info["observation_mask"]

            env_state = jax.tree_util.tree_map(
                lambda stepped, reset: _select_reset_on_done(dones, stepped, reset),
                next_env_state,
                reset_env_state,
            )
            obs = jax.tree_util.tree_map(
                lambda stepped, reset: _select_reset_on_done(dones, stepped, reset),
                next_obs,
                reset_obs,
            )
            action_mask = _select_reset_on_done(dones, next_action_mask, reset_action_mask)
            observation_mask = _select_reset_on_done(dones, next_observation_mask, reset_observation_mask)
            running_return = running_return + rewards.astype(jnp.float32)
            running_length = running_length + one_mask

            completed = dones.astype(jnp.float32)
            episode_reward_sum = episode_reward_sum + jnp.sum(running_return * completed)
            episode_length_sum = episode_length_sum + jnp.sum(running_length * completed)
            episode_count = episode_count + jnp.sum(completed)

            running_return = _select_reset_on_done(dones, running_return, zeros)
            running_length = _select_reset_on_done(dones, running_length, zeros)

            output = RolloutBatch(
                not_done_masks=1.0 - dones.astype(jnp.float32),
                rewards=rewards.astype(jnp.float32),
                log_probs=log_probs,
                entropies=entropies,
                values=values,
            )

            return (
                (
                    env_state,
                    obs,
                    action_mask,
                    observation_mask,
                    running_return,
                    running_length,
                    episode_reward_sum,
                    episode_length_sum,
                    episode_count,
                    rng_key,
                ),
                output,
            )

        carry, rollout = jax.lax.scan(
            body_fn,
            (
                env_state,
                obs,
                action_mask,
                observation_mask,
                running_return,
                running_length,
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                rng_key,
            ),
            xs=None,
            length=self.rollout_length,
        )

        (
            final_env_state,
            final_obs,
            final_action_mask,
            final_observation_mask,
            final_running_return,
            final_running_length,
            episode_reward_sum,
            episode_length_sum,
            episode_count,
            new_key,
        ) = carry
        _, bootstrap_values = actor_critic_forward(params, final_obs, final_action_mask, final_observation_mask)
        next_rollout_state = RolloutState(
            env_state=final_env_state,
            obs=final_obs,
            action_mask=final_action_mask,
            observation_mask=final_observation_mask,
            running_return=final_running_return,
            running_length=final_running_length,
        )
        return (
            rollout,
            bootstrap_values,
            episode_reward_sum,
            episode_length_sum,
            episode_count,
            next_rollout_state,
            new_key,
        )

    def _discounted_returns_and_advantages(
        self,
        rewards: jax.Array,
        values: jax.Array,
        not_done_masks: jax.Array,
        bootstrap_values: jax.Array,
        train_params: A2CTrainParams,
    ):
        next_values = jnp.concatenate([values[1:], bootstrap_values[None, :]], axis=0)
        rewards_rev = rewards[::-1]
        values_rev = values[::-1]
        next_values_rev = next_values[::-1]
        not_done_masks_rev = not_done_masks[::-1]

        def body_fn(carry, xs):
            R, advantage = carry
            reward, value, next_value, not_done_mask = xs

            R = reward + train_params.gamma * R * not_done_mask
            delta = reward + train_params.gamma * next_value * not_done_mask - value
            advantage = delta + train_params.gamma * train_params.lamda * advantage * not_done_mask

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

        return returns_rev[::-1], advantages_rev[::-1]

    def _loss_and_metrics(
        self,
        params: Any,
        rollout_state: RolloutState,
        rng_key: jax.Array,
        beta_e: jax.Array,
        train_params: A2CTrainParams,
    ):
        (
            rollout,
            bootstrap_values,
            episode_reward_sum,
            episode_length_sum,
            episode_count,
            next_rollout_state,
            new_key,
        ) = self._rollout(params, rollout_state, rng_key, train_params)

        returns, advantages = self._discounted_returns_and_advantages(
            rewards=rollout.rewards,
            values=rollout.values,
            not_done_masks=rollout.not_done_masks,
            bootstrap_values=bootstrap_values,
            train_params=train_params,
        )

        detached_advantages = jax.lax.stop_gradient(advantages)

        policy_loss = -jnp.mean(
            jnp.sum(rollout.log_probs * detached_advantages, axis=0)
        )
        value_loss = jnp.mean(
            jnp.sum((rollout.values - returns) ** 2, axis=0)
        )
        entropy_loss = -jnp.mean(
            jnp.sum(rollout.entropies, axis=0)
        )

        loss = policy_loss + train_params.beta_v * value_loss + beta_e * entropy_loss
        completed_count = jnp.maximum(episode_count, 1.0)
        episode_reward = episode_reward_sum / completed_count
        episode_length = episode_length_sum / completed_count

        metrics = StepMetrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            episode_reward=episode_reward,
            episode_length=episode_length,
            episode_count=episode_count,
            episode_reward_sum=episode_reward_sum,
            episode_length_sum=episode_length_sum,
            grad_norm=jnp.array(0.0, dtype=jnp.float32),
            param_norm=jnp.array(0.0, dtype=jnp.float32),
        )

        return loss, (metrics, next_rollout_state, new_key)

    def _optimizer_update(
        self,
        params: Any,
        grads: Any,
        optimizer: AdamState,
        train_params: A2CTrainParams,
        grad_norm: jax.Array | None = None,
    ):
        if grad_norm is None:
            grad_norm = _global_norm(grads)
        clip_coef = jnp.minimum(1.0, train_params.max_grad_norm / (grad_norm + 1e-6))
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
            - train_params.lr
            * (m_val / bias_correction1)
            / (jnp.sqrt(v_val / bias_correction2) + self.adam_eps),
            params,
            m,
            v,
        )

        return params, AdamState(step=step, m=m, v=v)

    def _train_step(self, state: JaxTrainState, beta_e: jax.Array, train_params: A2CTrainParams):
        (loss, (metrics, next_rollout_state, new_key)), grads = jax.value_and_grad(
            self._loss_and_metrics,
            has_aux=True,
        )(state.params, state.rollout_state, state.rng_key, beta_e, train_params)

        del loss

        grad_norm = _global_norm(grads)
        param_norm = _global_norm(state.params)
        params, optimizer = self._optimizer_update(
            state.params,
            grads,
            state.optimizer,
            train_params,
            grad_norm=grad_norm,
        )
        metrics = metrics._replace(
            grad_norm=grad_norm,
            param_norm=param_norm,
        )
        new_state = JaxTrainState(
            params=params,
            optimizer=optimizer,
            rollout_state=next_rollout_state,
            rng_key=new_key,
        )

        return new_state, metrics

    def _train_many(
        self,
        state: JaxTrainState,
        entropy_schedule: jax.Array,
        train_params: A2CTrainParams,
    ):
        def body_fn(carry, beta_e):
            next_state, metrics = self._train_step(carry, beta_e, train_params)
            return next_state, metrics

        return jax.lax.scan(body_fn, state, entropy_schedule)

    def _train_many_mean_metrics(
        self,
        state: JaxTrainState,
        entropy_schedule: jax.Array,
        train_params: A2CTrainParams,
    ):
        def body_fn(carry, beta_e):
            state, metric_sums = carry
            state, metrics = self._train_step(state, beta_e, train_params)
            metric_sums = jax.tree_util.tree_map(jnp.add, metric_sums, metrics)
            return (state, metric_sums), None

        init = (state, _zero_step_metrics(dtype=jnp.float32))
        (state, metric_sums), _ = jax.lax.scan(body_fn, init, entropy_schedule)
        num_steps = jnp.asarray(entropy_schedule.shape[0], dtype=jnp.float32)
        mean_metrics = jax.tree_util.tree_map(lambda x: x / num_steps, metric_sums)
        completed_count = jnp.maximum(metric_sums.episode_count, 1.0)
        mean_metrics = mean_metrics._replace(
            episode_reward=metric_sums.episode_reward_sum / completed_count,
            episode_length=metric_sums.episode_length_sum / completed_count,
            episode_count=metric_sums.episode_count,
            episode_reward_sum=metric_sums.episode_reward_sum,
            episode_length_sum=metric_sums.episode_length_sum,
        )
        return state, mean_metrics

    def train_step(
        self,
        state: JaxTrainState,
        train_params: A2CTrainParams,
        beta_e: float | None = None,
    ):
        if beta_e is None:
            beta_e = self.beta_e
        beta_e_array = jnp.asarray(beta_e, dtype=jnp.float32)

        return self._train_step_jit(state, beta_e_array, train_params)

    def train(
        self,
        state: JaxTrainState,
        num_updates: int,
        train_params: A2CTrainParams,
        entropy_schedule=None,
    ):
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
            "episode_count": [],
            "episode_reward_sum": [],
            "episode_length_sum": [],
            "grad_norm": [],
            "param_norm": [],
            "step_time_s": [],
            "cumulative_time_s": [],
        }

        start_time = time.perf_counter()
        for index in range(num_updates):
            step_start = time.perf_counter()
            state, metrics = self.train_step(
                state,
                train_params,
                beta_e=float(entropy_schedule[index]),
            )
            step_time = time.perf_counter() - step_start

            data["loss"].append(float(metrics.loss))
            data["policy_loss"].append(float(metrics.policy_loss))
            data["value_loss"].append(float(metrics.value_loss))
            data["entropy_loss"].append(float(metrics.entropy_loss))
            data["episode_length"].append(float(metrics.episode_length))
            data["episode_reward"].append(float(metrics.episode_reward))
            data["episode_count"].append(float(metrics.episode_count))
            data["episode_reward_sum"].append(float(metrics.episode_reward_sum))
            data["episode_length_sum"].append(float(metrics.episode_length_sum))
            data["grad_norm"].append(float(metrics.grad_norm))
            data["param_norm"].append(float(metrics.param_norm))
            data["step_time_s"].append(step_time)
            data["cumulative_time_s"].append(time.perf_counter() - start_time)

        return state, data

    def train_compiled(
        self,
        state: JaxTrainState,
        entropy_schedule,
        train_params: A2CTrainParams,
    ):
        entropy_schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_many_jit(state, entropy_schedule, train_params)

    def train_compiled_mean_metrics(
        self,
        state: JaxTrainState,
        entropy_schedule,
        train_params: A2CTrainParams,
    ):
        entropy_schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        return self._train_many_mean_metrics_jit(state, entropy_schedule, train_params)
