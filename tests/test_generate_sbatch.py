from pathlib import Path

from generate_sbatch import _build_job_summary_lines, _render_script


def test_render_script_accepts_tuple_array_params():
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

    assert "#SBATCH --array=0-3" in script
    assert 'POINT_SET_VALUES=(-3 -1 1 3)' in script
    assert '--point_set="${POINT_SET_VALUE}"' in script


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
