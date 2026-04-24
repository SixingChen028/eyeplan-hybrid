import json
import pickle
from pathlib import Path

import simulate


def _write_metadata(run_dir: Path, *, num_episodes: int = 100, batch_size: int = 10) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": {
            "num_episodes": num_episodes,
            "batch_size": batch_size,
        }
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))


def _write_checkpoint(run_dir: Path, next_update: int) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "train_state_latest.json").write_text(json.dumps({"next_update": next_update}))


def _write_training_data(run_dir: Path, num_updates: int) -> None:
    with open(run_dir / "data_training_jax.p", "wb") as file:
        pickle.dump({"episode_reward": [0.0] * num_updates}, file)


def _write_model_file(run_dir: Path) -> None:
    (run_dir / "net_jax.p").write_bytes(b"placeholder")


def test_main_experiment_target_simulates_all_complete_runs(monkeypatch, tmp_path: Path):
    results_root = tmp_path / "results"
    experiment_root = results_root / "runs" / "backup_check"

    run_complete_ckpt = experiment_root / "run-complete-ckpt"
    run_incomplete = experiment_root / "run-incomplete"
    run_complete_data = experiment_root / "run-complete-data"

    for run_dir in (run_complete_ckpt, run_incomplete, run_complete_data):
        _write_metadata(run_dir)
        _write_model_file(run_dir)

    _write_checkpoint(run_complete_ckpt, next_update=10)
    _write_checkpoint(run_incomplete, next_update=3)
    _write_training_data(run_complete_data, num_updates=10)

    simulated_dirs: list[str] = []

    def fake_simulate_run(run_dir: str, **kwargs):
        del kwargs
        simulated_dirs.append(run_dir)
        return 1, 1

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        simulate.sys,
        "argv",
        ["simulate.py", "backup_check", "--results_root", str(results_root)],
    )

    simulate.main()

    assert set(simulated_dirs) == {str(run_complete_ckpt), str(run_complete_data)}


def test_main_run_target_keeps_explicit_run_without_completion_filter(monkeypatch, tmp_path: Path):
    results_root = tmp_path / "results"
    run_dir = results_root / "runs" / "backup_check" / "run-incomplete"
    _write_metadata(run_dir)
    _write_model_file(run_dir)
    _write_checkpoint(run_dir, next_update=3)

    simulated_dirs: list[str] = []

    def fake_simulate_run(run_dir: str, **kwargs):
        del kwargs
        simulated_dirs.append(run_dir)
        return 1, 1

    monkeypatch.setattr(simulate, "_simulate_run", fake_simulate_run)
    monkeypatch.setattr(
        simulate.sys,
        "argv",
        ["simulate.py", "backup_check/run-incomplete", "--results_root", str(results_root)],
    )

    simulate.main()

    assert simulated_dirs == [str(run_dir)]
