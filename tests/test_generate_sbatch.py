from pathlib import Path

from generate_sbatch import _render_script


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
