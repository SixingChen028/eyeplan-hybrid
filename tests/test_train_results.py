import json
import pickle
from pathlib import Path

import pytest

from modules.environment_compat import ENVIRONMENT_COMPAT_VERSION
from modules.evaluation import evaluate_run_dir
from modules.train_results import (
    EVAL_SUMMARY_NAME,
    PARAMS_NAME,
    TRAINING_DATA_NAME,
    filter_pending_runs,
    prepare_run_dirs,
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


def test_prepare_run_dirs_writes_label_to_metadata(tmp_path: Path):
    run_dirs = prepare_run_dirs(
        [{"seed": 1, "wm_decay": 0.5}],
        path=str(tmp_path),
        experiment="labeled",
        config_path=Path("config/test.toml"),
        varied_keys=["seed"],
        label="obs-basic",
    )

    metadata_path = Path(run_dirs[0]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["args"]["label"] == "obs-basic"
    assert metadata["environment_compat_version"] == ENVIRONMENT_COMPAT_VERSION


def test_evaluate_run_dir_rejects_unversioned_params_by_default(tmp_path: Path):
    run_dir = tmp_path / "results" / "runs" / "test" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"args": {}}), encoding="utf-8")
    with (run_dir / PARAMS_NAME).open("wb") as file:
        pickle.dump({"w": [1.0]}, file)

    with pytest.raises(ValueError, match="missing environment compatibility metadata"):
        evaluate_run_dir(str(run_dir))
