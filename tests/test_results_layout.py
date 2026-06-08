from argparse import Namespace
from pathlib import Path

import pytest

from modules.results_layout import (
    create_run_dir,
    resolve_analysis_target,
    write_run_metadata,
)
from modules.environment_compat import ENVIRONMENT_COMPAT_VERSION


def _make_run_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "metadata.json").write_text("{}")


def test_resolve_experiment_shorthand(tmp_path: Path):
    results = tmp_path / "results"
    _make_run_dir(results / "runs" / "test" / "h0_s1_20260425_122806_pfs1")
    _make_run_dir(results / "runs" / "test" / "h1_s1_20260425_122807_8g4i")
    (results / "runs" / "test" / "incomplete").mkdir(parents=True)

    target = resolve_analysis_target("test", str(results))

    assert target.experiment == "test"
    assert target.kind == "experiment"
    assert target.run_dirs == [
        str(results / "runs" / "test" / "h0_s1_20260425_122806_pfs1"),
        str(results / "runs" / "test" / "h1_s1_20260425_122807_8g4i"),
    ]


def test_resolve_experiment_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    results = tmp_path / "results"
    run_dir = results / "runs" / "test" / "h0_s1_20260425_122806_pfs1"
    _make_run_dir(run_dir)
    monkeypatch.chdir(tmp_path)

    target = resolve_analysis_target("results/runs/test", str(results))

    assert target.experiment == "test"
    assert target.kind == "experiment"
    assert target.run_dirs == [str(run_dir)]


def test_resolve_run_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    results = tmp_path / "results"
    run_dir = results / "runs" / "test" / "h0_s1_20260425_122806_pfs1"
    _make_run_dir(run_dir)
    monkeypatch.chdir(tmp_path)

    target = resolve_analysis_target("results/runs/test/h0_s1_20260425_122806_pfs1", str(results))

    assert target.experiment == "test"
    assert target.kind == "run"
    assert target.run_dirs == [str(run_dir)]


def test_old_target_formats_fail(tmp_path: Path):
    results = tmp_path / "results"
    _make_run_dir(results / "runs" / "test" / "h0_s1_20260425_122806_pfs1")

    with pytest.raises(ValueError):
        resolve_analysis_target("test/*", str(results))

    with pytest.raises(ValueError):
        resolve_analysis_target("test/h0_s1_20260425_122806_pfs1", str(results))


def test_create_run_dir_uses_canonical_layout(tmp_path: Path):
    run_dir = create_run_dir(
        results_root=str(tmp_path / "results"),
        experiment="test",
        prefix="seed1",
        timestamp="20260425_122806",
        suffix="pfs1",
    )

    assert run_dir == str(tmp_path / "results" / "runs" / "test" / "seed1_20260425_122806_pfs1")
    assert Path(run_dir).is_dir()


def test_create_run_dir_without_prefix(tmp_path: Path):
    run_dir = create_run_dir(
        results_root=str(tmp_path / "results"),
        experiment="test",
        prefix="",
        timestamp="20260425_122806",
        suffix="pfs1",
    )

    assert run_dir == str(tmp_path / "results" / "runs" / "test" / "20260425_122806_pfs1")


def test_write_run_metadata_writes_args_and_git_sha_field(tmp_path: Path):
    run_dir = tmp_path / "results" / "runs" / "test" / "seed1_20260425_122806_pfs1"
    run_dir.mkdir(parents=True)
    args = Namespace(learning_rate=0.2, seed=15)

    metadata_path = write_run_metadata(run_dir=str(run_dir), args=args, cwd=str(tmp_path))

    assert metadata_path == str(run_dir / "metadata.json")
    content = (run_dir / "metadata.json").read_text()
    assert '"args"' in content
    assert '"git_sha": null' in content
    assert f'"environment_compat_version": {ENVIRONMENT_COMPAT_VERSION}' in content
