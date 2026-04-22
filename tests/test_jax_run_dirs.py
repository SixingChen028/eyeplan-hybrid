from pathlib import Path

from modules.jax_run_dirs import build_timestamped_run_dir, resolve_timestamped_run_dir


def test_build_timestamped_run_dir_without_jobid():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        jobid="0",
        timestamp="20260422_101500",
    )

    assert run_dir == "/tmp/results/20260422_101500"


def test_build_timestamped_run_dir_with_jobid():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        jobid="42",
        timestamp="20260422_101500",
    )

    assert run_dir == "/tmp/results/42_20260422_101500"


def test_resolve_timestamped_run_dir_picks_latest(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "20260422_090000").mkdir()
    latest = results_dir / "20260422_101500"
    latest.mkdir()
    (results_dir / "abc").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir))

    assert resolved == str(latest)


def test_resolve_timestamped_run_dir_filters_by_jobid(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "1_20260422_090000").mkdir()
    latest = results_dir / "2_20260422_101500"
    latest.mkdir()
    (results_dir / "2_20260422_091000").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), jobid="2")

    assert resolved == str(latest)
