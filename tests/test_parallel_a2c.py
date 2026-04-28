import pickle
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from modules.a2c import JaxBatchMaskA2C
from modules.environment import JaxDecisionTreeEnv
from modules.parallel_a2c import ParallelJaxBatchMaskA2C
from train_parallel_a2c import build_hypers, expand_sweep, save_results, train_with_progress


def _small_params(**overrides):
    params = {
        "num_nodes": 3,
        "hidden_size": 16,
        "batch_size": 4,
        "t_max": 4,
        "num_episodes": 8,
        "eval_episodes": 3,
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


def test_dynamic_env_params_match_default_env_for_same_values():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        beta_move=4.0,
        eps_move=0.0,
        learning_rate=1.0,
        lamda_backup=0.5,
        wm_decay=1.0,
        t_max=4,
        cost=0.01,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )
    key = jax.random.PRNGKey(2)

    state_default, obs_default, info_default = env.reset(key)
    state_dynamic, obs_dynamic, info_dynamic = env.reset_with_params(key, env.default_params())

    np.testing.assert_allclose(np.asarray(obs_dynamic), np.asarray(obs_default), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(info_dynamic["mask"]), np.asarray(info_default["mask"]))

    action = jnp.asarray(1, dtype=jnp.int32)
    default_step = env.step(state_default, action)
    dynamic_step = env.step_with_params(state_dynamic, action, env.default_params())

    for dynamic_leaf, default_leaf in zip(
        jax.tree_util.tree_leaves(dynamic_step[0]),
        jax.tree_util.tree_leaves(default_step[0]),
    ):
        np.testing.assert_allclose(np.asarray(dynamic_leaf), np.asarray(default_leaf), atol=1e-6)
    np.testing.assert_allclose(np.asarray(dynamic_step[1]), np.asarray(default_step[1]), atol=1e-6)
    np.testing.assert_allclose(float(dynamic_step[2]), float(default_step[2]), atol=1e-6)
    assert bool(dynamic_step[3]) == bool(default_step[3])


def test_parallel_sweep_compiles_and_returns_expected_shapes():
    fixed, combos, seeds, varied_keys = expand_sweep(_small_params())
    assert varied_keys == ["wm_decay"]
    assert len(combos) == 2
    assert seeds == [0, 1]

    env = JaxDecisionTreeEnv(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = ParallelJaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        batch_size=fixed["batch_size"],
        num_updates=int(fixed["num_episodes"] / fixed["batch_size"]),
    )

    result = trainer.train_sweep(build_hypers(combos), seeds)

    assert result.metrics.loss.shape == (2, 2, 2)
    assert result.metrics.episode_reward.shape == (2, 2, 2)
    assert result.states.optimizer.step.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(result.states.optimizer.step), np.full((2, 2), 2))


def test_train_with_progress_reports_numeric_rate(capsys):
    fixed, combos, seeds, _ = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = JaxDecisionTreeEnv(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    num_updates = int(fixed["num_episodes"] / fixed["batch_size"])
    trainer = ParallelJaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        batch_size=fixed["batch_size"],
        num_updates=num_updates,
    )

    result, elapsed_seconds = train_with_progress(
        trainer,
        build_hypers(combos),
        seeds,
        num_updates=num_updates,
        print_frequency=1,
    )

    assert result.metrics.loss.shape == (1, 1, 2)
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


def test_parallel_single_combo_matches_existing_a2c():
    params = _small_params(seed=0, wm_decay=1.0)
    fixed, combos, seeds, _ = expand_sweep(params)
    assert len(combos) == 1
    assert seeds == [0]

    env = JaxDecisionTreeEnv(
        num_nodes=fixed["num_nodes"],
        beta_move=fixed["beta_move"],
        eps_move=fixed["eps_move"],
        learning_rate=fixed["learning_rate"],
        lamda_backup=fixed["lamda_backup"],
        wm_decay=fixed["wm_decay"],
        t_max=fixed["t_max"],
        cost=fixed["cost"],
        scale_factor=fixed["scale_factor"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    num_updates = int(fixed["num_episodes"] / fixed["batch_size"])
    entropy_schedule = np.linspace(
        fixed["beta_e_init"],
        fixed["beta_e_final"],
        num_updates,
        dtype=np.float32,
    )

    reference_trainer = JaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        batch_size=fixed["batch_size"],
        lr=fixed["lr"],
        max_grad_norm=fixed["max_grad_norm"],
        gamma=fixed["gamma"],
        lamda=fixed["lamda"],
        beta_v=fixed["beta_v"],
        beta_e=fixed["beta_e_init"],
    )
    reference_state = reference_trainer.init_state(seed=0)
    reference_state, reference_metrics = reference_trainer.train_compiled(
        reference_state,
        entropy_schedule,
    )

    parallel_trainer = ParallelJaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        batch_size=fixed["batch_size"],
        num_updates=num_updates,
    )
    result = parallel_trainer.train_sweep(build_hypers(combos), seeds)

    np.testing.assert_allclose(
        np.asarray(result.metrics.loss[0, 0]),
        np.asarray(reference_metrics.loss),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.metrics.episode_reward[0, 0]),
        np.asarray(reference_metrics.episode_reward),
        atol=1e-6,
    )
    for parallel_leaf, reference_leaf in zip(
        jax.tree_util.tree_leaves(result.states.params),
        jax.tree_util.tree_leaves(reference_state.params),
    ):
        np.testing.assert_allclose(np.asarray(parallel_leaf[0, 0]), np.asarray(reference_leaf), atol=1e-6)


def test_default_shape_compiled_a2c_materializes_metrics():
    code = """
import numpy as np
import jax

from modules.a2c import JaxBatchMaskA2C
from modules.environment import JaxDecisionTreeEnv

env = JaxDecisionTreeEnv(
    num_nodes=15,
    beta_move=40.0,
    eps_move=0.0,
    learning_rate=1.0,
    lamda_backup=1.0,
    wm_decay=1.0,
    t_max=100,
    cost=0.01,
    scale_factor=1 / 8,
    shuffle_nodes=True,
)
trainer = JaxBatchMaskA2C(
    env=env,
    feature_size=env.observation_shape[0],
    action_size=env.action_size,
    hidden_size=128,
    batch_size=64,
    lr=5e-4,
    gamma=1.0,
    lamda=0.9,
    beta_v=0.05,
    beta_e=0.05,
)
state = trainer.init_state(seed=15)
_, metrics = trainer.train_compiled(state, np.array([0.05], dtype=np.float32))
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


def test_save_results_writes_existing_style_run_dirs(tmp_path):
    fixed, combos, seeds, varied_keys = expand_sweep(_small_params(seed=[0], wm_decay=[1.0]))
    env = JaxDecisionTreeEnv(
        num_nodes=fixed["num_nodes"],
        t_max=fixed["t_max"],
        shuffle_nodes=fixed["shuffle_nodes"],
        point_set=np.array([1.0], dtype=np.float32),
    )
    trainer = ParallelJaxBatchMaskA2C(
        env=env,
        feature_size=env.observation_shape[0],
        action_size=env.action_size,
        hidden_size=fixed["hidden_size"],
        batch_size=fixed["batch_size"],
        num_updates=int(fixed["num_episodes"] / fixed["batch_size"]),
    )
    result = trainer.train_sweep(build_hypers(combos), seeds)

    run_dirs = save_results(
        result,
        combos,
        seeds,
        path=str(tmp_path),
        experiment="parallel-test",
        config_path=tmp_path / "config.toml",
        varied_keys=varied_keys,
        elapsed_seconds=0.25,
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
