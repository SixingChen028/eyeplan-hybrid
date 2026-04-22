from argparse import Namespace
from pathlib import Path

from modules.jax_run_dirs import (
    build_timestamped_run_dir,
    resolve_timestamped_run_dir,
    write_run_metadata,
)


def test_build_timestamped_run_dir_without_jobid():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        jobid="0",
        timestamp="20260422_101500",
        suffix="a1b2",
    )

    assert run_dir == "/tmp/results/20260422_101500_a1b2"


def test_build_timestamped_run_dir_with_jobid():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        jobid="42",
        timestamp="20260422_101500",
        suffix="z9x8",
    )

    assert run_dir == "/tmp/results/42_20260422_101500_z9x8"


def test_build_timestamped_run_dir_generates_4_char_suffix():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        jobid="0",
        timestamp="20260422_101500",
    )
    run_name = Path(run_dir).name
    suffix = run_name.rsplit("_", 1)[-1]

    assert len(suffix) == 4


def test_resolve_timestamped_run_dir_picks_latest(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "20260422_090000_ab12").mkdir()
    latest = results_dir / "20260422_101500_cd34"
    latest.mkdir()
    (results_dir / "abc").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir))

    assert resolved == str(latest)


def test_resolve_timestamped_run_dir_filters_by_jobid(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "1_20260422_090000_ab12").mkdir()
    latest = results_dir / "2_20260422_101500_cd34"
    latest.mkdir()
    (results_dir / "2_20260422_091000_ef56").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), jobid="2")

    assert resolved == str(latest)


def test_resolve_timestamped_run_dir_supports_legacy_without_suffix(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    latest = results_dir / "3_20260422_101500"
    latest.mkdir()
    (results_dir / "3_20260422_091000").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), jobid="3")

    assert resolved == str(latest)


def test_write_run_metadata_writes_args_and_git_sha_field(tmp_path: Path):
    run_dir = tmp_path / "results" / "20260422_101500"
    run_dir.mkdir(parents=True)
    args = Namespace(jobid="7", learning_rate=0.2, seed=15)

    metadata_path = write_run_metadata(run_dir=str(run_dir), args=args, cwd=str(tmp_path))

    assert metadata_path == str(run_dir / "metadata.json")
    content = (run_dir / "metadata.json").read_text()
    assert '"args"' in content
    assert '"jobid": "7"' in content
    assert '"git_sha": null' in content
