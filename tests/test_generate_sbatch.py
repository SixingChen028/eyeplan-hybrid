from pathlib import Path

import pytest

from generate_sbatch import _render_script


def _config(array_start: int = 0, array_end: int | None = None) -> dict:
    sbatch = {"array_start": array_start}
    if array_end is not None:
        sbatch["array_end"] = array_end

    return {
        "sbatch": sbatch,
        "params": {
            "cost": [0.01, 0.02],
            "seed": [1, 2, 3],
            "num_episodes": 100,
            "batch_size": 10,
        },
    }


def test_render_script_can_start_array_after_existing_tasks():
    script = _render_script(_config(array_start=3), config_path=Path("cost-0425.toml"))

    assert "#SBATCH --array=3-5" in script
    assert "TOTAL=$((COST_N * SEED_N))" in script
    assert "TASK_ID=${SLURM_ARRAY_TASK_ID}" in script


def test_render_script_rejects_array_start_past_grid():
    with pytest.raises(ValueError, match="exceeds generated grid size"):
        _render_script(_config(array_start=6), config_path=Path("cost-0425.toml"))
