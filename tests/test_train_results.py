import json
from pathlib import Path

from modules.train_results import (
    EVAL_SUMMARY_NAME,
    PARAMS_NAME,
    TRAINING_DATA_NAME,
    filter_pending_runs,
)


def _write_run(
    root: Path,
    run_name: str,
    args: dict,
    *,
    complete: bool = True,
    eval_complete: bool = True,
) -> Path:
    run_dir = root / "runs" / "skip-existing" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"args": args}), encoding="utf-8")
    if complete:
        (run_dir / PARAMS_NAME).write_bytes(b"params")
        (run_dir / TRAINING_DATA_NAME).write_bytes(b"training")
        if eval_complete:
            (run_dir / EVAL_SUMMARY_NAME).write_text("{}", encoding="utf-8")
    return run_dir


def test_filter_pending_runs_skips_complete_matching_args(tmp_path: Path):
    completed = {
        "seed": 1,
        "wm_decay": 0.5,
        "point_set": [-8, -4, 4, 8],
    }
    _write_run(tmp_path, "done", completed)

    runs = [
        {"seed": 1, "wm_decay": 0.5, "point_set": (-8, -4, 4, 8)},
        {"seed": 2, "wm_decay": 0.5, "point_set": (-8, -4, 4, 8)},
    ]

    pending, skipped = filter_pending_runs(
        runs,
        path=str(tmp_path),
        experiment="skip-existing",
        require_eval=True,
    )

    assert pending == [runs[1]]
    assert len(skipped) == 1
    assert skipped[0].endswith("done")


def test_filter_pending_runs_requires_eval_when_requested(tmp_path: Path):
    run = {"seed": 1, "wm_decay": 0.5}
    _write_run(tmp_path, "no-eval", run, eval_complete=False)

    pending, skipped = filter_pending_runs(
        [run],
        path=str(tmp_path),
        experiment="skip-existing",
        require_eval=True,
    )

    assert pending == [run]
    assert skipped == []


def test_filter_pending_runs_can_skip_without_eval_requirement(tmp_path: Path):
    run = {"seed": 1, "wm_decay": 0.5}
    _write_run(tmp_path, "no-eval", run, eval_complete=False)

    pending, skipped = filter_pending_runs(
        [run],
        path=str(tmp_path),
        experiment="skip-existing",
        require_eval=False,
    )

    assert pending == []
    assert len(skipped) == 1
