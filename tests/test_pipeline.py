import numpy as np

from modules.a2c import JaxBatchMaskA2C
from modules.environment import JaxDecisionTreeEnv
from modules.simulation import JaxSimulator, to_transformed_simulation_format


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


def test_jax_simulator_runs_detailed_trials():
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
        detailed=True,
    )

    for key in ["activations", "counts", "gs", "qs", "logits"]:
        assert len(data[key]) == 5
        assert len(data[key][0]) == len(data["action_seqs"][0])
    assert len(data["activations"][0][0]) == env.num_nodes
    assert len(data["counts"][0][0]) == env.num_nodes
    assert len(data["gs"][0][0]) == env.num_nodes
    assert len(data["qs"][0][0]) == env.num_nodes
    assert len(data["logits"][0]) == len(data["action_seqs"][0])
    assert len(data["logits"][0][0]) == env.action_size


def test_jax_simulator_evaluate_policy_returns_summary_stats():
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

    state = trainer.init_state(seed=2)
    simulator = JaxSimulator(env)
    summary = simulator.evaluate_policy(
        params=state.params,
        seed=2,
        num_trials=17,
        greedy=True,
        batch_size=8,
    )

    assert summary["num_trials"] == 17
    assert summary["greedy"] is True
    assert np.isfinite(summary["reward_mean"])
    assert np.isfinite(summary["reward_sd"])
    assert np.isfinite(summary["reward_no_cost_mean"])
    assert np.isfinite(summary["reward_no_cost_sd"])
    assert np.isfinite(summary["n_steps_mean"])
    assert np.isfinite(summary["n_steps_sd"])


def test_transformed_simulation_format_encodes_actions():
    data = {
        "child_dicts": [
            {0: [1, 2]},
            {0: [1, 2]},
        ],
        "root_nodes": [0, 1],
        "points": [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, 3.0],
        ],
        "action_seqs": [
            [1, 2, 3],
            [1, 1, 3],
        ],
        "choice_seqs": [
            [2, 1],
            [2],
        ],
    }

    transformed = to_transformed_simulation_format(
        data,
        num_nodes=3,
        t_max=5,
        skip_timeout_trials=True,
    )

    assert list(transformed.keys()) == ["adj_lists", "starts", "rewards", "actions", "chosen_paths"]
    assert transformed["starts"] == [0, 1]
    assert transformed["adj_lists"][0] == [[1, 2], [], []]
    assert transformed["actions"][0] == [0, 1, 2, 3]
    assert transformed["chosen_paths"][0] == [2, 1]


def test_transformed_simulation_format_includes_details():
    data = {
        "child_dicts": [{0: [1, 2]}],
        "root_nodes": [0],
        "points": [[0.0, 1.0, -1.0]],
        "action_seqs": [[1, 2, 3]],
        "choice_seqs": [[2, 1]],
        "activations": [[[1.0, 0.5, 0.25], [1.0, 0.0, 0.25], [1.0, 0.0, 1.0]]],
        "counts": [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
        "gs": [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0]]],
        "qs": [[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 1.0]]],
        "logits": [[[0.0, 0.1, 0.2, 0.3], [0.0, 0.2, 0.1, 0.3], [0.0, 0.3, 0.2, 0.1]]],
    }

    transformed = to_transformed_simulation_format(
        data,
        num_nodes=3,
        t_max=5,
        skip_timeout_trials=True,
        detailed=True,
    )

    assert list(transformed.keys()) == [
        "adj_lists",
        "starts",
        "rewards",
        "actions",
        "chosen_path",
        "activations",
        "counts",
        "gs",
        "qs",
        "logits",
    ]
    assert transformed["activations"] == data["activations"]
    assert transformed["counts"] == data["counts"]
    assert transformed["gs"] == data["gs"]
    assert transformed["qs"] == data["qs"]
    assert transformed["logits"] == data["logits"]
    assert transformed["actions"] == [[0, 1, 2, 3]]
    assert transformed["chosen_path"] == data["choice_seqs"]


def test_transformed_simulation_format_skips_timeouts_when_requested():
    data = {
        "child_dicts": [
            {0: [1, 2]},
            {0: [1, 2]},
        ],
        "root_nodes": [0, 0],
        "points": [
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        "action_seqs": [
            [1, 1, 1, 3],
            [1, 2, 3],
        ],
        "choice_seqs": [
            [2],
            [1],
        ],
    }

    transformed_skip = to_transformed_simulation_format(
        data,
        num_nodes=3,
        t_max=4,
        skip_timeout_trials=True,
    )
    transformed_keep = to_transformed_simulation_format(
        data,
        num_nodes=3,
        t_max=4,
        skip_timeout_trials=False,
    )

    assert len(transformed_skip["actions"]) == 1
    assert len(transformed_keep["actions"]) == 2
