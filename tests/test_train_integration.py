import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def train_test_config_run(tmp_path_factory: pytest.TempPathFactory):
    repo_root = Path(__file__).resolve().parents[1]
    tmp_path = tmp_path_factory.mktemp("train-integration")
    result_path = tmp_path / "results"
    experiment = "train-integration"

    completed = subprocess.run(
        [
            sys.executable,
            "train.py",
            "config/test.toml",
            "--path",
            str(result_path),
            "--experiment",
            experiment,
            "--num_envs",
            "4",
            "--num_updates",
            "1",
            "--eval_episodes",
            "1",
            "--wm_decay",
            "0.0",
            "--cost",
            "0.01",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )

    run_root = result_path / "runs" / experiment
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    return {
        "completed": completed,
        "run_dir": run_dirs[0],
    }


def test_train_py_runs_test_config_with_small_training_geometry(train_test_config_run):
    stdout = train_test_config_run["completed"].stdout

    assert "writing results to " in stdout
    assert "jax_backend=cpu" in stdout
    assert "parallel_run_config runs=1 num_updates=1 num_envs=4" in stdout
    assert "parallel_train_started" in stdout
    assert "save_results runs=1 skip_eval=False" in stdout


def test_train_integration_writes_expected_artifacts(train_test_config_run):
    run_dir = train_test_config_run["run_dir"]

    assert (run_dir / "net_jax.p").exists()
    assert (run_dir / "data_training_jax.p").exists()
    assert (run_dir / "eval_summary_jax.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "training.log").exists()


def test_train_integration_metadata_records_overrides(train_test_config_run):
    run_dir = train_test_config_run["run_dir"]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["args"]["parallel_config"] == "config/test.toml"
    assert metadata["args"]["parallel_varied_keys"] == []
    assert metadata["args"]["num_envs"] == 4
    assert metadata["args"]["num_updates"] == 1
    assert metadata["args"]["eval_episodes"] == 1
    assert metadata["args"]["wm_decay"] == 0.0
    assert metadata["args"]["cost"] == 0.01
    assert metadata["args"]["network_type"] == "node_shared"


def test_train_integration_training_data_has_one_update(train_test_config_run):
    run_dir = train_test_config_run["run_dir"]

    with (run_dir / "data_training_jax.p").open("rb") as file:
        training_data = pickle.load(file)

    assert len(training_data["loss"]) == 1
    assert len(training_data["episode_reward"]) == 1
    assert len(training_data["episode_length"]) == 1


def test_train_integration_eval_summary_records_policy_stats(train_test_config_run):
    run_dir = train_test_config_run["run_dir"]

    eval_summary = json.loads((run_dir / "eval_summary_jax.json").read_text(encoding="utf-8"))
    assert eval_summary["num_trials"] == 1
    assert eval_summary["num_updates"] == 1
    assert "reward_mean" in eval_summary
    assert "reward_no_cost_mean" in eval_summary
    assert "n_steps_mean" in eval_summary


def test_train_integration_training_log_records_progress(train_test_config_run):
    run_dir = train_test_config_run["run_dir"]
    training_log = (run_dir / "training.log").read_text(encoding="utf-8")

    assert "run_summary run_index=0 seed=5" in training_log
    assert "eval_summary episodes=1" in training_log
    assert "reward_mean=" in training_log
    assert "eval_skipped=true" not in training_log
