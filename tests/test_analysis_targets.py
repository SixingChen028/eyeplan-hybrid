import os
from pathlib import Path

from modules.analysis_targets import (
    resolve_analysis_target,
    select_most_recent_run,
)


def _make_run_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text("{}")


def test_resolve_experiment_shorthand_legacy_layout(tmp_path: Path):
    results = tmp_path / "results"
    _make_run_dir(results / "exp-a" / "run-1")
    _make_run_dir(results / "exp-a" / "run-2")

    target = resolve_analysis_target("exp-a", str(results))

    assert target.experiment == "exp-a"
    assert target.kind == "experiment"
    assert len(target.run_dirs) == 2


def test_resolve_run_shorthand(tmp_path: Path):
    results = tmp_path / "results"
    run_path = results / "runs" / "exp-b" / "run-7"
    _make_run_dir(run_path)

    target = resolve_analysis_target("exp-b/run-7", str(results))

    assert target.experiment == "exp-b"
    assert target.kind == "run"
    assert target.run_dirs == [str(run_path)]


def test_resolve_wildcard_shorthand(tmp_path: Path):
    results = tmp_path / "results"
    _make_run_dir(results / "runs" / "exp-c" / "run-1")
    _make_run_dir(results / "runs" / "exp-c" / "run-2")

    target = resolve_analysis_target("exp-c/*", str(results))

    assert target.experiment == "exp-c"
    assert target.kind == "wildcard"
    assert len(target.run_dirs) == 2


def test_resolve_full_run_path(tmp_path: Path):
    results = tmp_path / "results"
    run_path = results / "runs" / "exp-d" / "run-9"
    _make_run_dir(run_path)

    target = resolve_analysis_target(str(run_path), str(results))

    assert target.experiment == "exp-d"
    assert target.kind == "run"
    assert target.run_dirs == [str(run_path)]


def test_resolve_analysis_experiment_path(tmp_path: Path):
    results = tmp_path / "results"
    _make_run_dir(results / "runs" / "exp-e" / "run-1")
    _make_run_dir(results / "runs" / "exp-e" / "run-2")
    (results / "analysis" / "exp-e" / "summary").mkdir(parents=True, exist_ok=True)

    target = resolve_analysis_target(str(results / "analysis" / "exp-e"), str(results))

    assert target.experiment == "exp-e"
    assert target.kind == "experiment"
    assert len(target.run_dirs) == 2


def test_resolve_analysis_run_path(tmp_path: Path):
    results = tmp_path / "results"
    run_path = results / "runs" / "exp-f" / "run-9"
    _make_run_dir(run_path)
    (results / "analysis" / "exp-f" / "runs" / "run-9").mkdir(parents=True, exist_ok=True)

    target = resolve_analysis_target(
        str(results / "analysis" / "exp-f" / "runs" / "run-9"),
        str(results),
    )

    assert target.experiment == "exp-f"
    assert target.kind == "run"
    assert target.run_dirs == [str(run_path)]


def test_select_most_recent_run_uses_mtime(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _make_run_dir(run_a)
    _make_run_dir(run_b)

    os.utime(run_a, (1, 1))
    os.utime(run_b, (2, 2))

    selected = select_most_recent_run([str(run_a), str(run_b)])
    assert selected == str(run_b)
