import os
from pathlib import Path
import subprocess

import pytest

from generate_local import _render_script
from modules.config import normalize_config


def test_normalize_config_allows_launcher_local_table():
    normalized = normalize_config(
        {
            "local": {
                "gpus": [0, 1],
                "processes_per_gpu": 1,
            }
        }
    )

    assert "params" in normalized


def test_normalize_config_allows_condition_tables():
    normalized = normalize_config(
        {
            "params": {
                "seed": [1, 2],
            },
            "conditions": [
                {
                    "cost": 0.01,
                    "wm_decay": 0.5,
                }
            ],
        }
    )

    assert normalized["params"]["seed"] == [1, 2]
    assert len(normalized["conditions"]) == 1


def test_render_script_assigns_array_tasks_to_gpus():
    config = {
        "meta": {
            "experiment": "local-render",
            "result_path": "./results",
            "array_vars": ["seed"],
        },
        "training": {
            "seed": [7, 9],
            "num_updates": 1,
            "num_envs": 1,
            "rollout_length": 1,
        },
        "local": {
            "gpus": [0, 1],
            "processes_per_gpu": 1,
        },
    }

    script = _render_script(config, config_path=Path("config/test.toml"))

    assert "GPUS=(0 1)" in script
    assert "TOTAL=2" in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU}"' in script
    assert '--seed="${SEED_VALUE}"' in script


def test_render_script_passes_label_from_meta():
    config = {
        "meta": {
            "label": "obs-basic",
        },
    }

    script = _render_script(config, config_path=Path("config/test.toml"))
    assert "LABEL=obs-basic" in script
    assert '--label="${LABEL}"' in script


def test_render_script_expands_condition_tables_for_local_grid():
    config = {
        "meta": {
            "array_vars": ["seed"],
        },
        "params": {
            "seed": [7, 9],
            "cost": [0.01, 0.02],
        },
        "conditions": [
            {
                "wm_decay": 0.0,
                "label": "value",
            },
            {
                "wm_decay": 0.5,
                "seed": 11,
                "label": "mcts",
            },
        ],
        "local": {
            "gpus": [0],
        },
    }

    script = _render_script(config, config_path=Path("config/test.toml"))

    assert "TOTAL=3" in script
    assert "--condition=0 --seed=7" in script
    assert "--condition=0 --seed=9" in script
    assert "--condition=1" in script
    assert "--cost=" not in script
    assert "--wm_decay=" not in script
    assert "--label=" not in script


def test_render_script_defaults_to_four_cpus_and_one_process_per_gpu():
    script = _render_script({}, config_path=Path("config/test.toml"))

    assert "THREADS=4" in script
    assert "PROCESSES_PER_GPU=1" in script


@pytest.mark.slow
def test_generated_script_executes_train_py_for_local_grid_task(tmp_path: Path):
    config_path = tmp_path / "local_exec_test.toml"
    config_path.write_text(
        (
            "[meta]\n"
            f"result_path = {str(tmp_path / 'results')!r}\n"
            "experiment = 'local-train-exec'\n"
            "array_vars = ['seed']\n"
            "\n"
            "[training]\n"
            "seed = [7, 9]\n"
            "num_updates = 1\n"
            "num_envs = 1\n"
            "rollout_length = 1\n"
            "\n"
            "[local]\n"
            f"log = {str(tmp_path / 'logs')!r}\n"
            "gpus = ['']\n"
            "processes_per_gpu = 1\n"
            "cpus_per_task = 1\n"
        ),
        encoding="utf-8",
    )

    config = {
        "meta": {
            "result_path": str(tmp_path / "results"),
            "experiment": "local-train-exec",
            "array_vars": ["seed"],
        },
        "training": {
            "seed": [7, 9],
            "num_updates": 1,
            "num_envs": 1,
            "rollout_length": 1,
        },
        "local": {
            "log": str(tmp_path / "logs"),
            "gpus": [""],
            "processes_per_gpu": 1,
            "cpus_per_task": 1,
        },
    }
    script_path = tmp_path / "local.sh"
    script_path.write_text(_render_script(config, config_path=config_path), encoding="utf-8")
    script_path.chmod(0o755)

    subprocess.run(
        ["bash", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )

    run_root = tmp_path / "results" / "runs" / "local-train-exec"
    assert run_root.exists()
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 2
