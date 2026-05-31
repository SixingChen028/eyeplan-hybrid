import pickle
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.a2c import A2CTrainParams, JaxBatchMaskA2C
from modules.a2c_sweep import VmappedA2CTrainer, build_hypers
from modules.config import expand_sweep
from modules.config import ENV_DYNAMIC_PARAM_KEYS, load_canonical_defaults
from modules.environment import JaxDecisionTreeEnv
from modules.evaluation import evaluate_run_dir
from modules.network import flatten_observation
from modules.train_progress import StartupTrainingTimeout, train_with_progress
from modules.train_results import save_results

_, _DEFAULT_PARAMS = load_canonical_defaults()


def _obs_size(env: JaxDecisionTreeEnv) -> int:
    return int(flatten_observation(env.observation_template).shape[0])


def _env(**overrides):
    params = dict(_DEFAULT_PARAMS)
    params.update(overrides)
    return JaxDecisionTreeEnv(
        num_nodes=int(params["num_nodes"]),
        t_max=int(params["t_max"]),
        scale_factor=float(params["scale_factor"]),
        shuffle_nodes=bool(params["shuffle_nodes"]),
        wm_only=bool(params["wm_only"]),
        persist_terminal=bool(params["persist_terminal"]),
        use_recency_obs=bool(params["use_recency_obs"]),
        use_best_open_value_obs=bool(params["use_best_open_value_obs"]),
        use_best_terminal_value_obs=bool(params["use_best_terminal_value_obs"]),
        use_g_values_obs=bool(params["use_g_values_obs"]),
        use_q_values_obs=bool(params["use_q_values_obs"]),
        use_n_visits_obs=bool(params["use_n_visits_obs"]),
        use_is_terminal_obs=bool(params["use_is_terminal_obs"]),
        use_time_elapsed_obs=bool(params["use_time_elapsed_obs"]),
        backup_mode=str(params["backup_mode"]),
        point_set=params["point_set"],
    )


def _env_params(env, **overrides):
    params = {key: _DEFAULT_PARAMS[key] for key in ENV_DYNAMIC_PARAM_KEYS}
    params.update(overrides)
    return env.make_params(**params)


def _small_params(**overrides):
    params = {
        "num_nodes": 3,
        "hidden_size": 16,
        "t_max": 4,
        "num_updates": 2,
        "num_envs": 4,
        "rollout_length": 4,
        "seed": [0, 1],
        "beta_move": 4.0,
        "eps_move": 0.0,
        "learning_rate": 1.0,
        "lamda_backup": 0.5,
        "wm_decay": [0.5, 1.0],
        "cost": 0.01,
        "scale_factor": 1.0,
        "shuffle_nodes": False,
        "lr": 1e-3,
        "gamma": 1.0,
        "lamda": 1.0,
        "beta_v": 0.05,
        "beta_e_init": 0.05,
        "beta_e_final": 0.01,
        "max_grad_norm": 1.0,
    }
    params.update(overrides)
    return params


def _a2c_train_params(env, config):
    return A2CTrainParams(
        env=_env_params(env, beta_move=config["beta_move"],
            eps_move=config["eps_move"],
            learning_rate=config["learning_rate"],
            lamda_backup=config["lamda_backup"],
            wm_decay=config["wm_decay"],
            cost=config["cost"],
        ),
        lr=config["lr"],
        gamma=config["gamma"],
        lamda=config["lamda"],
        beta_v=config["beta_v"],
        max_grad_norm=config["max_grad_norm"],
    )


@pytest.mark.slow
def test_dynamic_env_params_match_default_env_for_same_values():
    env = _env(
        num_nodes=3,
        t_max=4,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )
    key = jax.random.PRNGKey(2)
    params = _env_params(env, beta_move=4.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=0.5,
        wm_decay=1.0,
        cost=0.01,
    )

    state_default, obs_default, info_default = env.reset(key, params)
    state_dynamic, obs_dynamic, info_dynamic = env.reset(key, params)

    np.testing.assert_allclose(
        np.asarray(flatten_observation(obs_dynamic)),
        np.asarray(flatten_observation(obs_default)),
        atol=1e-6,
    )
    np.testing.assert_array_equal(np.asarray(info_dynamic["mask"]), np.asarray(info_default["mask"]))

    action = jnp.asarray(1, dtype=jnp.int32)
    default_step = env.step(state_default, action, params)
    dynamic_step = env.step(state_dynamic, action, params)

    for dynamic_leaf, default_leaf in zip(
        jax.tree_util.tree_leaves(dynamic_step[0]),
        jax.tree_util.tree_leaves(default_step[0]),
    ):
        np.testing.assert_allclose(np.asarray(dynamic_leaf), np.asarray(default_leaf), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(flatten_observation(dynamic_step[1])),
        np.asarray(flatten_observation(default_step[1])),
        atol=1e-6,
    )
    np.testing.assert_allclose(float(dynamic_step[2]), float(default_step[2]), atol=1e-6)
    assert bool(dynamic_step[3]) == bool(default_step[3])


@pytest.mark.slow
def test_parallel_sweep_compiles_and_returns_expected_shapes():
    fixed, runs, varied_keys = expand_sweep(_small_params())
    assert varied_keys == ["seed", "wm_decay"]
    assert len(runs) == 4
    assert [run["seed"] for run in runs] == [0, 0, 1, 1]

    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=fixed["num_updates"],
    )

    result = trainer.train_sweep(build_hypers(runs))

    assert result.metrics.loss.shape == (4, 2)
    assert result.metrics.episode_reward.shape == (4, 2)
    assert result.states.optimizer.step.shape == (4,)
    np.testing.assert_array_equal(np.asarray(result.states.optimizer.step), np.full((4,), 2))


@pytest.mark.slow
def test_parallel_sweep_compiles_node_shared_network():
    fixed, runs, _ = expand_sweep(
        _small_params(seed=[0], wm_decay=[1.0], network_type="node_shared")
    )
    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=fixed["num_updates"],
        network_type=fixed["network_type"],
    )

    result = trainer.train_sweep(build_hypers(runs))

    assert result.metrics.loss.shape == (1, 2)
    np.testing.assert_array_equal(np.asarray(result.states.optimizer.step), np.full((1,), 2))


def test_startup_training_timeout_exits_with_message(capsys):
    exit_codes = []
    diagnostics = []
    timeout = StartupTrainingTimeout(
        5,
        exit_code=124,
        exit_fn=exit_codes.append,
        diagnostic_fn=lambda: diagnostics.append("called"),
    )
    timeout.set_stage("init_sweep_states")

    timeout._expire()

    assert exit_codes == [124]
    assert diagnostics == ["called"]
    stderr = capsys.readouterr().err
    assert "parallel_train_startup_timeout seconds=5 stage=init_sweep_states" in stderr
    assert "reason=training_not_started" in stderr


def test_parallel_sweep_allows_shape_stable_recency_decay_arrays():
    fixed, runs, varied_keys = expand_sweep(
        _small_params(seed=0, wm_decay=0.5, recency_decay=[0, 0.5])
    )

    assert varied_keys == ["recency_decay"]
    assert len(runs) == 2

    hypers = build_hypers(runs)
    np.testing.assert_allclose(np.asarray(hypers.env.recency_decay), np.array([0.0, 0.5], dtype=np.float32))

    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        use_recency_obs=True,
        point_set=np.array([1.0], dtype=np.float32),
    )
    no_recency_env = _env(num_nodes=fixed["num_nodes"], use_recency_obs=False)
    assert _obs_size(env) == _obs_size(no_recency_env) + fixed["num_nodes"]


def test_parallel_sweep_allows_forget_rate_arrays():
    fixed, runs, varied_keys = expand_sweep(
        _small_params(seed=0, wm_decay=0.5, forget_rate=[0.0, 0.25])
    )

    assert varied_keys == ["forget_rate"]
    assert len(runs) == 2

    hypers = build_hypers(runs)
    np.testing.assert_allclose(np.asarray(hypers.env.forget_rate), np.array([0.0, 0.25], dtype=np.float32))


def test_parallel_sweep_allows_q_drift_arrays():
    fixed, runs, varied_keys = expand_sweep(
        _small_params(seed=0, wm_decay=1.0, q_drift=[0.0, 0.25])
    )

    assert varied_keys == ["q_drift"]
    assert len(runs) == 2

    hypers = build_hypers(runs)
    np.testing.assert_allclose(np.asarray(hypers.env.q_drift), np.array([0.0, 0.25], dtype=np.float32))


def test_parallel_sweep_keeps_q_decay_float():
    fixed, runs, varied_keys = expand_sweep(
        _small_params(seed=0, wm_decay=1.0, q_drift=[0.0, 0.5], q_decay=0.75, scale_factor=0.25)
    )

    assert varied_keys == ["q_drift"]
    assert len(runs) == 2

    expected = np.array([0.75, 0.75], dtype=np.float32)
    hypers = build_hypers(runs)
    np.testing.assert_allclose(np.asarray(hypers.env.q_decay), expected, atol=1e-6)


def test_parallel_sweep_allows_move_cost_scale_arrays():
    fixed, runs, varied_keys = expand_sweep(
        _small_params(seed=0, wm_decay=1.0, move_cost_scale=[0.0, 1.5])
    )

    assert varied_keys == ["move_cost_scale"]
    assert len(runs) == 2

    hypers = build_hypers(runs)
    expected = np.array([0.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(hypers.env.move_cost_scale), expected, atol=1e-6)


def test_parallel_sweep_rejects_non_numeric_recency_decay_arrays():
    with np.testing.assert_raises(ValueError):
        expand_sweep(_small_params(seed=0, recency_decay=["off", 0.5]))


@pytest.mark.slow
def test_train_with_progress_reports_numeric_rate(capsys):
    fixed, runs, _ = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    num_updates = fixed["num_updates"]
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=num_updates,
    )

    result, elapsed_seconds = train_with_progress(
        trainer,
        build_hypers(runs),
        num_updates=num_updates,
        print_frequency=1,
    )

    assert result.metrics.loss.shape == (1, 2)
    assert elapsed_seconds >= 0.0
    progress_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("parallel_train_progress ")
    ]
    assert len(progress_lines) == 2
    assert all("updates_per_second=" in line for line in progress_lines)
    for line in progress_lines:
        rate = float(line.rsplit("updates_per_second=", maxsplit=1)[1])
        assert rate > 0.0


@pytest.mark.slow
def test_train_with_progress_tracks_startup_timeout_compile_stages():
    class Timeout:
        def __init__(self):
            self.cancel_count = 0
            self.stages = []

        def cancel(self):
            self.cancel_count += 1

        def set_stage(self, stage):
            self.stages.append(stage)

    fixed, runs, _ = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=fixed["num_updates"],
    )
    timeout = Timeout()

    train_with_progress(
        trainer,
        build_hypers(runs),
        num_updates=fixed["num_updates"],
        print_frequency=0,
        startup_timeout=timeout,
    )

    assert timeout.cancel_count == 1
    assert timeout.stages == ["init_sweep_states", "compile_train_sweep_chunk"]


@pytest.mark.slow
def test_parallel_single_combo_matches_existing_a2c():
    params = _small_params(seed=0, wm_decay=1.0)
    fixed, runs, _ = expand_sweep(params)
    assert len(runs) == 1
    assert runs[0]["seed"] == 0

    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        scale_factor=fixed["scale_factor"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    num_updates = fixed["num_updates"]
    entropy_schedule = np.linspace(
        fixed["beta_e_init"],
        fixed["beta_e_final"],
        num_updates,
        dtype=np.float32,
    )

    reference_trainer = JaxBatchMaskA2C(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        lr=fixed["lr"],
        max_grad_norm=fixed["max_grad_norm"],
        gamma=fixed["gamma"],
        lamda=fixed["lamda"],
        beta_v=fixed["beta_v"],
        beta_e=fixed["beta_e_init"],
    )
    train_params = _a2c_train_params(env, runs[0])
    reference_state = reference_trainer.init_state(seed=0, env_params=train_params.env)
    reference_state, reference_metrics = reference_trainer.train_compiled(
        reference_state,
        entropy_schedule,
        train_params,
    )

    parallel_trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=num_updates,
    )
    result = parallel_trainer.train_sweep(build_hypers(runs))

    parallel_loss = np.asarray(result.metrics.loss[0])
    reference_loss = np.asarray(reference_metrics.loss)
    assert parallel_loss.shape == reference_loss.shape
    assert np.all(np.isfinite(parallel_loss))
    assert np.all(np.isfinite(reference_loss))
    assert np.all(np.isfinite(np.asarray(result.metrics.episode_reward[0])))


@pytest.mark.slow
def test_default_shape_compiled_a2c_materializes_metrics():
    code = """
import numpy as np
import jax

from modules.a2c import A2CTrainParams, JaxBatchMaskA2C
from modules.environment import JaxDecisionTreeEnv

env = JaxDecisionTreeEnv(
    num_nodes=15,
    t_max=100,
    scale_factor=1 / 8,
    shuffle_nodes=True,
    use_recency_obs=False,
    use_best_open_value_obs=True,
    use_best_terminal_value_obs=True,
    use_g_values_obs=True,
    use_q_values_obs=True,
    use_n_visits_obs=True,
    use_is_terminal_obs=True,
    use_time_elapsed_obs=True,
    backup_mode="full",
    point_set=(-8, -4, -2, -1, 1, 2, 4, 8),
)
trainer = JaxBatchMaskA2C(
    env=env,
    action_size=env.action_size,
    hidden_size=128,
    num_envs=64,
    lr=5e-4,
    gamma=1.0,
    lamda=0.9,
    beta_v=0.05,
    beta_e=0.05,
)
env_params = env.make_params(
    beta_move=40.0,
    eps_move=0.0,
    learning_rate=1.0,
    lamda_backup=1.0,
    backup_steps=100,
    wm_decay=1.0,
    wm_neighbor_activation=1.0,
    forget_rate=0.0,
    q_drift=0.0,
    q_decay=0.0,
    recency_decay=0.0,
    cost=0.01,
)
train_params = A2CTrainParams(
    env=env_params,
    lr=5e-4,
    gamma=1.0,
    lamda=0.9,
    beta_v=0.05,
    max_grad_norm=1.0,
)
state = trainer.init_state(seed=15, env_params=env_params)
_, metrics = trainer.train_compiled(state, np.array([0.05], dtype=np.float32), train_params)
jax.device_get(metrics.episode_reward)
"""
    try:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            timeout=20,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError("compiled A2C metrics did not materialize") from error


def test_expand_sweep_rejects_shape_changing_arrays():
    try:
        expand_sweep(_small_params(num_nodes=[3, 5]))
    except ValueError as error:
        assert "changes compiled shapes" in str(error)
        return
    assert False, "shape-changing arrays should be rejected"


def test_expand_sweep_rejects_backup_mode_arrays():
    try:
        expand_sweep(_small_params(backup_mode=["full", "wm_zero"]))
    except ValueError as error:
        assert "changes compiled shapes" in str(error)
        return
    assert False, "backup_mode arrays should be rejected"


def test_expand_sweep_rejects_invalid_backup_mode():
    try:
        expand_sweep(_small_params(backup_mode="invalid"))
    except ValueError as error:
        assert "backup_mode must be one of" in str(error)
        return
    assert False, "invalid backup_mode should be rejected"


def test_expand_sweep_rejects_unknown_params():
    try:
        expand_sweep(_small_params(num_episodes=8))
    except ValueError as error:
        assert "Unknown [params] keys: num_episodes" in str(error)
        return
    assert False, "unknown params should be rejected"


@pytest.mark.slow
def test_save_results_writes_existing_style_run_dirs(tmp_path):
    fixed, runs, varied_keys = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=fixed["num_updates"],
    )
    result = trainer.train_sweep(build_hypers(runs))

    run_dirs = save_results(
        result,
        runs,
        path=str(tmp_path),
        experiment="parallel-test",
        config_path=tmp_path / "config.toml",
        varied_keys=varied_keys,
        elapsed_seconds=0.25,
        run_eval=True,
        eval_episodes=3,
    )

    assert len(run_dirs) == 1
    run_dir = Path(run_dirs[0])
    assert run_dir.exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "net_jax.p").exists()
    assert (run_dir / "eval_summary_jax.json").exists()
    with open(run_dir / "data_training_jax.p", "rb") as file:
        data = pickle.load(file)
    assert len(data["loss"]) == 2


@pytest.mark.slow
def test_save_results_skips_eval_by_default(tmp_path):
    fixed, runs, varied_keys = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=fixed["num_updates"],
    )
    result = trainer.train_sweep(build_hypers(runs))

    run_dirs = save_results(
        result,
        runs,
        path=str(tmp_path),
        experiment="parallel-test",
        config_path=tmp_path / "config.toml",
        varied_keys=varied_keys,
        elapsed_seconds=0.25,
    )

    run_dir = Path(run_dirs[0])
    assert (run_dir / "net_jax.p").exists()
    assert (run_dir / "data_training_jax.p").exists()
    assert not (run_dir / "eval_summary_jax.json").exists()
    assert "eval_skipped=true" in (run_dir / "training.log").read_text()


@pytest.mark.slow
def test_evaluate_run_dir_uses_recorded_eval_episodes(tmp_path):
    fixed, runs, varied_keys = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = _env(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = VmappedA2CTrainer(
        env=env,
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        num_envs=fixed["num_envs"],
        num_updates=fixed["num_updates"],
    )
    result = trainer.train_sweep(build_hypers(runs))
    run_dirs = save_results(
        result,
        runs,
        path=str(tmp_path),
        experiment="parallel-test",
        config_path=tmp_path / "config.toml",
        varied_keys=varied_keys,
        elapsed_seconds=0.25,
        eval_episodes=3,
    )

    path, summary = evaluate_run_dir(run_dirs[0])

    assert path == str(Path(run_dirs[0]) / "eval_summary_jax.json")
    assert summary["num_trials"] == 3
    assert summary["num_updates"] == 2
    assert summary["train_elapsed_seconds"] == 0.25
    assert Path(path).exists()
