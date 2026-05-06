import numpy as np

from modules.a2c import JaxBatchMaskA2C
from modules.environment import JaxDecisionTreeEnv
from modules.simulation import JaxSimulator, append_simulation_trial, empty_simulation_data


def test_jax_train_step_compiles_and_runs():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=5,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        num_envs=4,
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


def test_jax_train_step_runs_node_shared_network():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=5,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        num_envs=4,
        lr=1e-3,
        max_grad_norm=1.0,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.05,
        network_type="node_shared",
    )

    state = trainer.init_state(seed=0)
    state, metrics = trainer.train_step(state, beta_e=0.05)

    assert int(state.optimizer.step) == 1
    assert np.isfinite(float(metrics.loss))


def test_jax_simulator_runs_trials():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=5,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        num_envs=4,
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
        skip_timeout_trials=False,
    )

    assert set(data) == {"adj_lists", "starts", "rewards", "actions", "chosen_paths"}
    assert len(data["actions"]) <= 5
    assert all(len(seq) <= env.t_max + 1 for seq in data["actions"])


def test_jax_simulator_runs_node_shared_trials():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=5,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=16,
        num_envs=4,
        lr=1e-3,
        max_grad_norm=1.0,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.05,
        network_type="node_shared",
    )

    state = trainer.init_state(seed=1)
    simulator = JaxSimulator(env)
    data = simulator.simulate(
        params=state.params,
        seed=1,
        num_trials=5,
        greedy=False,
        skip_timeout_trials=False,
    )

    assert set(data) == {"adj_lists", "starts", "rewards", "actions", "chosen_paths"}
    assert len(data["actions"]) <= 5


def test_jax_simulator_runs_detailed_trials():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=5,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        num_envs=4,
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
        skip_timeout_trials=False,
    )

    for key in ["activations", "counts", "gs", "qs", "logits"]:
        assert len(data[key]) == 5
        assert len(data[key][0]) == len(data["actions"][0]) - 1
    assert len(data["activations"][0][0]) == env.num_nodes
    assert len(data["counts"][0][0]) == env.num_nodes
    assert len(data["gs"][0][0]) == env.num_nodes
    assert len(data["qs"][0][0]) == env.num_nodes
    assert len(data["logits"][0]) == len(data["actions"][0]) - 1
    assert len(data["logits"][0][0]) == env.action_size


def test_jax_simulator_records_forced_terminal_action():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=1,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        num_envs=4,
        lr=1e-3,
        max_grad_norm=1.0,
        gamma=1.0,
        lamda=1.0,
        beta_v=0.05,
        beta_e=0.05,
    )

    state = trainer.init_state(seed=3)
    simulator = JaxSimulator(env)
    data = simulator.simulate(
        params=state.params,
        seed=3,
        num_trials=4,
        greedy=False,
        skip_timeout_trials=False,
    )

    for actions in data["actions"]:
        assert actions[-1] == env.num_nodes


def test_jax_simulator_evaluate_policy_returns_summary_stats():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        t_max=5,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=32,
        num_envs=4,
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


def test_append_simulation_trial_encodes_export_shape():
    data = empty_simulation_data(detailed=False)
    appended = append_simulation_trial(
        data,
        child_nodes=np.array([[1, 2], [-1, -1], [-1, -1]], dtype=np.int32),
        root_node=0,
        points=np.array([0.0, 1.0, -1.0], dtype=np.float32),
        action_seq=[1, 2, 3],
        choice_seq=[2, 1],
        num_nodes=3,
        t_max=5,
        skip_timeout_trials=True,
    )

    assert appended is True
    assert list(data.keys()) == ["adj_lists", "starts", "rewards", "actions", "chosen_paths"]
    assert data["starts"] == [0]
    assert data["adj_lists"] == [[[1, 2], [], []]]
    assert data["actions"] == [[0, 1, 2, 3]]
    assert data["chosen_paths"] == [[2, 1]]


def test_append_simulation_trial_includes_details():
    data = empty_simulation_data(detailed=True)
    details = {
        "activations": [[1.0, 0.5, 0.25], [1.0, 0.0, 0.25], [1.0, 0.0, 1.0]],
        "counts": [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        "gs": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        "qs": [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 1.0]],
        "logits": [[0.0, 0.1, 0.2, 0.3], [0.0, 0.2, 0.1, 0.3], [0.0, 0.3, 0.2, 0.1]],
    }
    append_simulation_trial(
        data,
        child_nodes=np.array([[1, 2], [-1, -1], [-1, -1]], dtype=np.int32),
        root_node=0,
        points=np.array([0.0, 1.0, -1.0], dtype=np.float32),
        action_seq=[1, 2, 3],
        choice_seq=[2, 1],
        num_nodes=3,
        t_max=5,
        skip_timeout_trials=True,
        details=details,
    )

    assert list(data.keys()) == [
        "adj_lists",
        "starts",
        "rewards",
        "actions",
        "chosen_paths",
        "activations",
        "counts",
        "gs",
        "qs",
        "logits",
    ]
    assert data["activations"] == [details["activations"]]
    assert data["counts"] == [details["counts"]]
    assert data["gs"] == [details["gs"]]
    assert data["qs"] == [details["qs"]]
    assert data["logits"] == [details["logits"]]
    assert data["actions"] == [[0, 1, 2, 3]]
    assert data["chosen_paths"] == [[2, 1]]


def test_append_simulation_trial_skips_timeouts_when_requested():
    child_nodes = np.array([[1, 2], [-1, -1], [-1, -1]], dtype=np.int32)
    points = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    data = empty_simulation_data()

    appended_timeout = append_simulation_trial(
        data,
        child_nodes=child_nodes,
        root_node=0,
        points=points,
        action_seq=[1, 1, 1, 3],
        choice_seq=[2],
        num_nodes=3,
        t_max=4,
        skip_timeout_trials=True,
    )
    appended_complete = append_simulation_trial(
        data,
        child_nodes=child_nodes,
        root_node=0,
        points=points,
        action_seq=[1, 2, 3],
        choice_seq=[1],
        num_nodes=3,
        t_max=4,
        skip_timeout_trials=True,
    )

    assert appended_timeout is False
    assert appended_complete is True
    assert len(data["actions"]) == 1
