import os
from pathlib import Path
import subprocess
import sys

from generate_sbatch import DEFAULT_META, _build_job_summary_lines, _render_script


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


def test_generated_script_executes_command_and_passes_only_selected_axes(tmp_path: Path, monkeypatch):
    entrypoint = tmp_path / "echo_args.py"
    entrypoint.write_text(
        (
            "import json\n"
            "import sys\n"
            "print(json.dumps(sys.argv[1:]))\n"
        ),
        encoding="utf-8",
    )

    config = {
        "meta": {
            "experiment": "exec-test",
            "result_path": str(tmp_path / "results"),
            "array_vars": ["seed"],
        },
        "params": {
            "seed": [7, 9],
            "point_set": (-3, -1, 1, 3),
        },
    }

    monkeypatch.setitem(DEFAULT_META, "entrypoint", str(entrypoint))
    monkeypatch.setitem(DEFAULT_META, "python", f"{sys.executable} -u")

    script_path = tmp_path / "job.sh"
    script_path.write_text(_render_script(config, config_path=Path("config/test.toml")), encoding="utf-8")
    script_path.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
    )

    assert "grid_task task_id=0 seed=7" in result.stdout
    assert '["config/test.toml", "--path=' in result.stdout
    assert '"--seed=7"' in result.stdout
    assert "--point_set=" not in result.stdout


def test_build_job_summary_lines_marks_array_axes_and_resources():
    config = {
        "meta": {
            "experiment": "summary",
            "result_path": "./results",
            "array_vars": ["lr"],
        },
        "params": {
            "lr": [0.001, 0.002],
            "seed": [1, 2],
        },
        "sbatch": {
            "gpu": True,
            "time": "01:30:00",
            "mem_per_cpu": "3G",
            "cpus_per_task": 4,
        },
    }

    lines = _build_job_summary_lines(config, config_path=Path("config/test.toml"))

    assert "Job summary:" in lines
    assert "  - lr (array): [0.001, 0.002]" in lines
    assert "  - seed (not array): [1, 2]" in lines
    assert "Resources: gpu:1, time=01:30:00, mem-per-cpu=3G" in lines
