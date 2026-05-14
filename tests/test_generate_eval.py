import json
import subprocess
import sys
from pathlib import Path


def test_generate_eval_skips_run_without_params(tmp_path: Path):
    results_root = tmp_path / "results"
    run_dir = results_root / "runs" / "experiment" / "run-without-params"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"args": {}}))

    result = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "experiment",
            "--results_root",
            str(results_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "skip incomplete run=" in result.stdout
    assert str(run_dir) in result.stdout
    assert not (run_dir / "eval_summary_jax.json").exists()
