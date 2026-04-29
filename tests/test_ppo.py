import numpy as np

from modules.environment import JaxDecisionTreeEnv
from modules.ppo import JaxBatchMaskPPO


def test_jax_ppo_train_step_runs():
    env = JaxDecisionTreeEnv(
        num_nodes=5,
        beta_move=4.0,
        eps_move=0.0,
        learning_rate=1.0,
        wm_decay=1.0,
        t_max=7,
        cost=0.0,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        batch_size=8,
        lr=1e-3,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.01,
        clip_eps=0.2,
        ppo_epochs=3,
        normalize_advantages=True,
        max_grad_norm=1.0,
    )

    state = trainer.init_state(seed=0)
    state, metrics = trainer.train_step(state, beta_e=0.01)

    assert int(state.optimizer.step) == 3
    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.policy_loss))
    assert np.isfinite(float(metrics.value_loss))
    assert np.isfinite(float(metrics.entropy_loss))
    assert np.isfinite(float(metrics.clip_fraction))
    assert np.isfinite(float(metrics.approx_kl))
    assert np.isfinite(float(metrics.episode_reward))
    assert np.isfinite(float(metrics.episode_length))


def test_jax_ppo_train_step_runs_node_shared_network():
    env = JaxDecisionTreeEnv(
        num_nodes=5,
        beta_move=4.0,
        eps_move=0.0,
        learning_rate=1.0,
        wm_decay=1.0,
        t_max=7,
        cost=0.0,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskPPO(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        batch_size=8,
        lr=1e-3,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.01,
        clip_eps=0.2,
        ppo_epochs=2,
        normalize_advantages=True,
        max_grad_norm=1.0,
        network_type="node_shared",
    )

    state = trainer.init_state(seed=0)
    state, metrics = trainer.train_step(state, beta_e=0.01)

    assert int(state.optimizer.step) == 2
    assert np.isfinite(float(metrics.loss))
