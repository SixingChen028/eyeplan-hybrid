import json
import os
import subprocess
import sys
from pathlib import Path


def test_train_py_runs_test_config_with_small_training_geometry(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
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

    assert "parallel_run_config runs=1 num_updates=1 num_envs=4" in completed.stdout
    assert "save_results runs=1 skip_eval=False" in completed.stdout

    run_root = result_path / "runs" / experiment
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    assert (run_dir / "net_jax.p").exists()
    assert (run_dir / "data_training_jax.p").exists()
    assert (run_dir / "training.log").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["args"]["parallel_config"] == "config/test.toml"
    assert metadata["args"]["parallel_varied_keys"] == []
    assert metadata["args"]["num_envs"] == 4
    assert metadata["args"]["num_updates"] == 1
    assert metadata["args"]["eval_episodes"] == 1
    assert metadata["args"]["wm_decay"] == 0.0
    assert metadata["args"]["cost"] == 0.01

    eval_summary = json.loads((run_dir / "eval_summary_jax.json").read_text(encoding="utf-8"))
    assert eval_summary["num_trials"] == 1
    assert eval_summary["num_updates"] == 1
