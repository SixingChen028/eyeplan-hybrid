import subprocess
import sys
from pathlib import Path

from modules.config import normalize_config
from preflight_rollout_invariants import unique_environment_runs


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unique_environment_runs_ignores_non_environment_params():
    config = normalize_config(
        {
            "environment": {
                "num_nodes": 7,
                "t_max": 31,
            },
            "training": {
                "seed": [1, 2],
                "lr": [0.001, 0.002],
            },
        }
    )

    runs = unique_environment_runs(config)

    assert len(runs) == 1
    assert runs[0]["num_nodes"] == 7
    assert runs[0]["t_max"] == 31


def test_unique_environment_runs_keeps_distinct_environment_params():
    config = normalize_config(
        {
            "environment": {
                "num_nodes": 7,
                "t_max": 31,
                "wm_decay": [0.5, 1.0],
            },
            "training": {
                "seed": [1, 2],
            },
        }
    )

    runs = unique_environment_runs(config)

    assert [run["wm_decay"] for run in runs] == [0.5, 1.0]


def test_preflight_rollout_invariants_script_runs_config(tmp_path):
    config_path = tmp_path / "preflight.toml"
    config_path.write_text(
        """
[environment]
num_nodes = 7
t_max = 31
wm_decay = [0.5, 1.0]

[training]
seed = [1, 2]
""".lstrip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "preflight_rollout_invariants.py",
            str(config_path),
            "--num-rollouts",
            "2",
            "--noisy-steps",
            "2",
            "--max-consistency-steps",
            "2",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "unique_environments=2" in result.stdout
    assert "preflight_rollout_invariants=passed" in result.stdout
