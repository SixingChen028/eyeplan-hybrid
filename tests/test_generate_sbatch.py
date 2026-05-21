import os
from pathlib import Path
import subprocess

import pytest

from generate_sbatch import _render_script


def test_render_script_keeps_point_set_tuple_as_single_param():
    config = {
        "meta": {
            "experiment": "tuple-point-set",
            "result_path": "./results",
        },
        "params": {
            "point_set": (-3, -1, 1, 3),
        },
    }

    script = _render_script(config, config_path=Path("config/test.toml"))

    assert "#SBATCH --array=0-0" in script
    assert "POINT_SET_VALUES=" not in script
    assert '--point_set="${POINT_SET_VALUE}"' not in script


def test_render_script_passes_skip_existing_from_meta():
    config = {
        "meta": {
            "skip_existing": True,
        },
    }

    script = _render_script(config, config_path=Path("config/test.toml"))

    assert "--skip-existing" in script


@pytest.mark.slow
def test_generated_script_executes_train_py_for_one_array_task(tmp_path: Path):
    config_path = tmp_path / "slurm_exec_test.toml"
    config_path.write_text(
        (
            "[meta]\n"
            f"result_path = {str(tmp_path / 'results')!r}\n"
            "experiment = 'sbatch-train-exec'\n"
            "array_vars = ['seed']\n"
            "\n"
            "[training]\n"
            "seed = [7, 9]\n"
            "num_updates = 1\n"
            "num_envs = 1\n"
            "rollout_length = 1\n"
            "eval_episodes = 1\n"
            "\n"
            "[sbatch]\n"
            "cpus_per_task = 1\n"
            "time = '00:05:00'\n"
            "mem_per_cpu = '1G'\n"
        ),
        encoding="utf-8",
    )

    config = {
        "meta": {
            "result_path": str(tmp_path / "results"),
            "experiment": "sbatch-train-exec",
            "array_vars": ["seed"],
        },
        "training": {
            "seed": [7, 9],
            "num_updates": 1,
            "num_envs": 1,
            "rollout_length": 1,
            "eval_episodes": 1,
        },
        "sbatch": {
            "cpus_per_task": 1,
            "time": "00:05:00",
            "mem_per_cpu": "1G",
        },
    }
    script_path = tmp_path / "job.sh"
    script_path.write_text(_render_script(config, config_path=config_path), encoding="utf-8")
    script_path.chmod(0o755)

    subprocess.run(
        ["bash", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
    )

    run_root = tmp_path / "results" / "runs" / "sbatch-train-exec"
    assert run_root.exists()
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
