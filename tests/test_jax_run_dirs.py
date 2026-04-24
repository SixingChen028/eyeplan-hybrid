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
        experiment="default",
        jobid="",
        timestamp="20260422_101500",
        suffix="a1b2",
    )

    assert run_dir == "/tmp/results/runs/default/20260422_101500_a1b2"


def test_build_timestamped_run_dir_with_jobid():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        experiment="default",
        jobid="42",
        timestamp="20260422_101500",
        suffix="z9x8",
    )

    assert run_dir == "/tmp/results/runs/default/42_20260422_101500_z9x8"


def test_build_timestamped_run_dir_generates_4_char_suffix():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        experiment="default",
        jobid="",
        timestamp="20260422_101500",
    )
    run_name = Path(run_dir).name
    suffix = run_name.rsplit("_", 1)[-1]

    assert len(suffix) == 4


def test_build_timestamped_run_dir_with_zero_jobid():
    run_dir = build_timestamped_run_dir(
        path="/tmp/results",
        experiment="default",
        jobid="0",
        timestamp="20260422_101500",
        suffix="m3n4",
    )

    assert run_dir == "/tmp/results/runs/default/0_20260422_101500_m3n4"


def test_resolve_timestamped_run_dir_picks_latest(tmp_path: Path):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs" / "default"
    runs_dir.mkdir(parents=True)
    (runs_dir / "20260422_090000_ab12").mkdir()
    latest = runs_dir / "20260422_101500_cd34"
    latest.mkdir()
    (runs_dir / "abc").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), experiment="default")

    assert resolved == str(latest)


def test_resolve_timestamped_run_dir_filters_by_jobid(tmp_path: Path):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs" / "default"
    runs_dir.mkdir(parents=True)
    (runs_dir / "1_20260422_090000_ab12").mkdir()
    latest = runs_dir / "2_20260422_101500_cd34"
    latest.mkdir()
    (runs_dir / "2_20260422_091000_ef56").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), experiment="default", jobid="2")

    assert resolved == str(latest)


def test_resolve_timestamped_run_dir_supports_legacy_without_suffix(tmp_path: Path):
    results_dir = tmp_path / "results"
    legacy_dir = results_dir / "default"
    legacy_dir.mkdir(parents=True)
    latest = legacy_dir / "3_20260422_101500"
    latest.mkdir()
    (legacy_dir / "3_20260422_091000").mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), experiment="default", jobid="3")

    assert resolved == str(latest)


def test_resolve_timestamped_run_dir_with_custom_experiment(tmp_path: Path):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs" / "exp-x"
    runs_dir.mkdir(parents=True)
    latest = runs_dir / "9_20260422_101500_a1b2"
    latest.mkdir()

    resolved = resolve_timestamped_run_dir(path=str(results_dir), experiment="exp-x", jobid="9")

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
