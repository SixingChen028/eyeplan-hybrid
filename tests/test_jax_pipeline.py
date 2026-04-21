import numpy as np

from modules.jax_a2c import JaxBatchMaskA2C
from modules.jax_environment import JaxDecisionTreeEnv
from modules.jax_simulation import JaxSimulator


def test_jax_train_step_compiles_and_runs():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        beta_move=4.0,
        eps_move=0.0,
        learning_rate=1.0,
        wm_decay=1.0,
        t_max=5,
        cost=0.01,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        batch_size=4,
        lr=1e-3,
        max_grad_norm=1.0,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.05,
    )

    state = trainer.init_state(seed=0)
    state, metrics = trainer.train_step(state, beta_e=0.05)

    assert int(state.optimizer.step) == 1
    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.episode_reward))
    assert np.isfinite(float(metrics.episode_length))


def test_jax_simulator_runs_trials():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        beta_move=4.0,
        eps_move=0.0,
        learning_rate=1.0,
        wm_decay=1.0,
        t_max=5,
        cost=0.01,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        batch_size=4,
        lr=1e-3,
        max_grad_norm=1.0,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.05,
    )

    state = trainer.init_state(seed=1)
    simulator = JaxSimulator(env)
    data = simulator.simulate(
        params=state.params,
        seed=1,
        num_trials=5,
        greedy=False,
    )

    assert len(data["action_seqs"]) == 5
    assert len(data["choice_seqs"]) == 5
    assert all(len(seq) <= env.t_max for seq in data["action_seqs"])
