from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from modules.a2c import A2CTrainParams, JaxBatchMaskA2C, JaxTrainState, StepMetrics
from modules.config import ENV_DYNAMIC_PARAM_KEYS, TRAIN_SWEEP_KEYS
from modules.environment import JaxDecisionTreeEnv, JaxDecisionTreeParams


class A2CHyperParams(NamedTuple):
    env: JaxDecisionTreeParams
    seed: jax.Array
    lr: jax.Array
    gamma: jax.Array
    lamda: jax.Array
    beta_v: jax.Array
    beta_e_init: jax.Array
    beta_e_final: jax.Array
    max_grad_norm: jax.Array


class A2CSweepResult(NamedTuple):
    states: JaxTrainState
    metrics: StepMetrics


def build_hypers(combos: list[dict]) -> A2CHyperParams:
    def array(key: str, dtype=jnp.float32):
        return jnp.asarray([combo[key] for combo in combos], dtype=dtype)

    int_keys = {"backup_steps", "seed"}
    bool_keys = {"persist_terminal"}
    env_values = {
        key: array(key, dtype=jnp.bool_ if key in bool_keys else jnp.int32 if key in int_keys else jnp.float32)
        for key in ENV_DYNAMIC_PARAM_KEYS
    }
    env = JaxDecisionTreeParams(**env_values)

    train_values = {
        key: array(key, dtype=jnp.int32 if key in int_keys else jnp.float32)
        for key in TRAIN_SWEEP_KEYS
    }
    return A2CHyperParams(env=env, **train_values)


class VmappedA2CTrainer:
    def __init__(
        self,
        env: JaxDecisionTreeEnv,
        action_size: int,
        hidden_size: int,
        num_envs: int,
        num_updates: int,
        rollout_length: int | None = None,
        network_type: str = "mlp",
    ):
        self.num_updates = int(num_updates)
        self.trainer = JaxBatchMaskA2C(
            env=env,
            action_size=action_size,
            hidden_size=hidden_size,
            num_envs=num_envs,
            rollout_length=rollout_length,
            lr=1.0,
            gamma=1.0,
            lamda=1.0,
            beta_v=1.0,
            beta_e=0.0,
            network_type=network_type,
        )
        self._train_one_jit = jax.jit(self._train_one)
        self._train_sweep_jit = jax.jit(self._train_sweep)
        self._init_sweep_states_jit = jax.jit(self._init_sweep_states)
        self._train_sweep_chunk_jit = jax.jit(self._train_sweep_chunk)

    @staticmethod
    def _train_params(hyper: A2CHyperParams) -> A2CTrainParams:
        return A2CTrainParams(
            env=hyper.env,
            lr=hyper.lr,
            gamma=hyper.gamma,
            lamda=hyper.lamda,
            beta_v=hyper.beta_v,
            max_grad_norm=hyper.max_grad_norm,
        )

    def _train_one(self, hyper: A2CHyperParams):
        state = self.trainer.init_state_with_params(hyper.seed, hyper.env)
        entropy_schedule = jnp.linspace(
            hyper.beta_e_init,
            hyper.beta_e_final,
            self.num_updates,
            dtype=jnp.float32,
        )
        return self.trainer._train_many(state, entropy_schedule, self._train_params(hyper))

    def _train_one_from_state(
        self,
        state: JaxTrainState,
        hyper: A2CHyperParams,
        entropy_schedule: jax.Array,
    ):
        return self.trainer._train_many(state, entropy_schedule, self._train_params(hyper))

    def _train_sweep(self, hypers: A2CHyperParams):
        states, metrics = jax.vmap(self._train_one)(hypers)
        return A2CSweepResult(states=states, metrics=metrics)

    def _init_sweep_states(self, hypers: A2CHyperParams):
        return jax.vmap(lambda hyper: self.trainer.init_state_with_params(hyper.seed, hyper.env))(hypers)

    def _train_sweep_chunk(
        self,
        states: JaxTrainState,
        hypers: A2CHyperParams,
        entropy_schedule: jax.Array,
    ):
        states, metrics = jax.vmap(self._train_one_from_state, in_axes=(0, 0, 0))(
            states,
            hypers,
            entropy_schedule,
        )
        return A2CSweepResult(states=states, metrics=metrics)

    def train_sweep(self, hypers: A2CHyperParams):
        return self._train_sweep_jit(hypers)

    def init_sweep_states(self, hypers: A2CHyperParams):
        return self._init_sweep_states_jit(hypers)

    def train_sweep_chunk(
        self,
        states: JaxTrainState,
        hypers: A2CHyperParams,
        entropy_schedule,
    ):
        return self._train_sweep_chunk_jit(states, hypers, jnp.asarray(entropy_schedule, dtype=jnp.float32))

    def compile_train_sweep_chunk(
        self,
        states: JaxTrainState,
        hypers: A2CHyperParams,
        entropy_schedule,
    ) -> None:
        schedule = jnp.asarray(entropy_schedule, dtype=jnp.float32)
        self._train_sweep_chunk_jit.lower(states, hypers, schedule).compile()
