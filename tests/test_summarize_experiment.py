import csv
import json
import sys
from pathlib import Path

from summarize_experiment import EVAL_FIELDS, main


def _write_run(run_dir: Path, args: dict, reward: float = 1.0) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"args": args}), encoding="utf-8")
    eval_summary = {
        "n_steps_mean": 2.0,
        "n_steps_sd": 0.0,
        "reward_mean": reward,
        "reward_sd": 0.0,
        "reward_no_cost_mean": reward,
        "reward_no_cost_sd": 0.0,
        "train_elapsed_seconds": 3.0,
    }
    (run_dir / "eval_summary_jax.json").write_text(json.dumps(eval_summary), encoding="utf-8")


def test_summarize_experiment_ignores_toml_for_varying_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    experiment = "summary_test"
    _write_run(
        results_root / "runs" / experiment / "seed0_20260101_000000_abcd",
        {"experiment": experiment, "seed": 0, "wm_decay": 0.9},
        reward=1.0,
    )
    _write_run(
        results_root / "runs" / experiment / "seed1_20260101_000000_efgh",
        {"experiment": experiment, "seed": 1, "wm_decay": 0.9},
        reward=2.0,
    )

    config_path = tmp_path / "config" / f"{experiment}.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
[params]
seed = [0, 1]
wm_decay = [0.9, 0.8]
missing_from_metadata = [1, 2]
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_experiment.py", str(config_path), "--results_root", str(results_root)],
    )
    main()

    output_path = results_root / "analysis" / experiment / "summary" / "evaluation.csv"
    with output_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert list(rows[0]) == ["run_id", "seed", *EVAL_FIELDS]
    assert {row["seed"] for row in rows} == {"0", "1"}
