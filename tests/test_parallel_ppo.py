import jax.numpy as jnp
import numpy as np

from modules.environment import JaxDecisionTreeEnv, JaxDecisionTreeParams
from modules.parallel_ppo import PPOHyperParams, ParallelJaxBatchMaskPPO


def _build_hypers():
    env = JaxDecisionTreeParams(
        beta_move=jnp.asarray([4.0, 4.0], dtype=jnp.float32),
        eps_move=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        learning_rate=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        lamda_backup=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        backup_steps=jnp.asarray([100, 100], dtype=jnp.int32),
        wm_decay=jnp.asarray([0.5, 1.0], dtype=jnp.float32),
        q_drop_rate=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        recency_decay=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        cost=jnp.asarray([0.01, 0.01], dtype=jnp.float32),
        scale_factor=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        shuffle_nodes=jnp.asarray([False, False], dtype=np.bool_),
        wm_backup=jnp.asarray([False, False], dtype=np.bool_),
    )
    return PPOHyperParams(
        env=env,
        lr=jnp.asarray([1e-3, 1e-3], dtype=jnp.float32),
        gamma=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        lamda=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        beta_v=jnp.asarray([0.05, 0.05], dtype=jnp.float32),
        beta_e_init=jnp.asarray([0.05, 0.05], dtype=jnp.float32),
        beta_e_final=jnp.asarray([0.01, 0.01], dtype=jnp.float32),
        max_grad_norm=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )


def _build_single_hyper():
    env = JaxDecisionTreeParams(
        beta_move=jnp.asarray([4.0], dtype=jnp.float32),
        eps_move=jnp.asarray([0.0], dtype=jnp.float32),
        learning_rate=jnp.asarray([1.0], dtype=jnp.float32),
        lamda_backup=jnp.asarray([0.5], dtype=jnp.float32),
        backup_steps=jnp.asarray([100], dtype=jnp.int32),
        wm_decay=jnp.asarray([1.0], dtype=jnp.float32),
        q_drop_rate=jnp.asarray([0.0], dtype=jnp.float32),
        recency_decay=jnp.asarray([0.0], dtype=jnp.float32),
        cost=jnp.asarray([0.01], dtype=jnp.float32),
        scale_factor=jnp.asarray([1.0], dtype=jnp.float32),
        shuffle_nodes=jnp.asarray([False], dtype=np.bool_),
        wm_backup=jnp.asarray([False], dtype=np.bool_),
    )
    return PPOHyperParams(
        env=env,
        lr=jnp.asarray([1e-3], dtype=jnp.float32),
        gamma=jnp.asarray([1.0], dtype=jnp.float32),
        lamda=jnp.asarray([1.0], dtype=jnp.float32),
        beta_v=jnp.asarray([0.05], dtype=jnp.float32),
        beta_e_init=jnp.asarray([0.05], dtype=jnp.float32),
        beta_e_final=jnp.asarray([0.01], dtype=jnp.float32),
        max_grad_norm=jnp.asarray([1.0], dtype=jnp.float32),
    )


def test_parallel_ppo_sweep_compiles_and_returns_expected_shapes():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=4,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = ParallelJaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        batch_size=4,
        num_updates=2,
        ppo_epochs=2,
    )
    result = trainer.train_sweep(_build_hypers(), seeds=[0, 1])

    assert result.metrics.loss.shape == (2, 2, 2)
    assert result.metrics.episode_reward.shape == (2, 2, 2)
    assert result.states.optimizer.step.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(result.states.optimizer.step), np.full((2, 2), 4))


def test_parallel_ppo_sweep_compiles_node_shared_network():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=4,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = ParallelJaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        batch_size=4,
        num_updates=2,
        ppo_epochs=2,
        network_type="node_shared",
    )
    result = trainer.train_sweep(_build_single_hyper(), seeds=[0])

    assert result.metrics.loss.shape == (1, 1, 2)
    np.testing.assert_array_equal(np.asarray(result.states.optimizer.step), np.full((1, 1), 4))


def test_parallel_ppo_loss_handles_fully_masked_rows():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=4,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = ParallelJaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        batch_size=2,
        num_updates=1,
        ppo_epochs=1,
    )
    hyper = _build_single_hyper()
    state = trainer.init_state(jnp.asarray(0, dtype=jnp.int32))
    data = {
        "obs": jnp.zeros((2, env.observation_shape[0]), dtype=jnp.float32),
        "action_mask": jnp.zeros((2, env.action_size), dtype=jnp.bool_),
        "actions": jnp.zeros((2,), dtype=jnp.int32),
        "old_log_probs": jnp.zeros((2,), dtype=jnp.float32),
        "masks": jnp.zeros((2,), dtype=jnp.float32),
        "returns": jnp.zeros((2,), dtype=jnp.float32),
        "advantages": jnp.zeros((2,), dtype=jnp.float32),
    }
    loss, aux = trainer._ppo_loss(state.params, data, jnp.asarray(0.01, dtype=jnp.float32), hyper)
    assert np.all(np.isfinite(np.asarray(loss)))
    assert all(np.all(np.isfinite(np.asarray(value))) for value in aux)
