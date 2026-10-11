"""Exercise shard identity, output validation, and resumable execution without JAX."""

import argparse
import json
from pathlib import Path

import pytest

import action_fitting_jobs as jobs


def test_partition_covers_exact_range():
    shards = jobs.partition_trials(5, 23, 7)
    assert shards == [(5, 7), (12, 7), (19, 7), (26, 2)]
    assert [trial for start, count in shards for trial in range(start, start + count)] == list(range(5, 28))


def test_worker_resume_verifies_identity_and_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "revision", lambda: "frozen_revision")
    data = tmp_path / "data.json"
    jobs.write_json(data, [{"moves": [1]}])
    metadata = tmp_path / "run" / "metadata.json"
    jobs.write_json(metadata, {"args": {}})
    run = {"run_dir": "run", "metadata_sha256": jobs.file_hash(metadata), "params_sha256": None}
    context = {"code_sha": "frozen_revision", "source_sha256": {},
               "inputs": {"data": {"path": "data.json", "sha256": jobs.file_hash(data)}}}
    task = {"candidate_id": "candidate", "model": "main", "runs": [run], "parameters": {},
            "trial_start": 0, "trial_count": 1, "samples": 4, "repeats": 2, "batch_size": 2, "seed": 1}
    task["task_key"] = jobs.object_hash({"context": context, "task": task})
    task["output"] = f"cache/{task['task_key']}.json"
    manifest = {**context, "resources": {"cpus_per_task": 1}, "tasks": [task]}
    manifest_path = tmp_path / "jobs.json"
    jobs.write_json(manifest_path, manifest)
    commands = []

    def calculate(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.with_suffix(".npz").write_bytes(b"arrays")
        jobs.write_json(output, {"code_sha": context["code_sha"], "source_sha256": {},
                                "data_sha256": jobs.file_hash(data), "samples": 4, "repeats": 2,
                                "batch_size": 2, "seed": 1, "trial_start": 0, "trial_count": 1,
                                "runs": [{**run, "estimates": [[{}, {}]]}],
                                "arrays_file": output.with_suffix(".npz").name,
                                "arrays_sha256": jobs.file_hash(output.with_suffix(".npz"))})

    monkeypatch.setattr(jobs.subprocess, "run", calculate)
    args = argparse.Namespace(manifest=manifest_path, task_index=0)
    jobs.run_task(args)
    jobs.run_task(args)
    assert len(commands) == 1
    output = tmp_path / task["output"]
    receipt = output.with_suffix(".complete.json")
    assert jobs.read_json(receipt)["task_key"] == task["task_key"]
    receipt.unlink()  # A crash after cache writing must recover without new sampling.
    jobs.run_task(args)
    assert len(commands) == 1
    assert jobs.read_json(receipt)["recovered_completed_output"]
    output.with_suffix(".npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="array identity or hash"):
        jobs.run_task(args)
    assert len(commands) == 1
    changed = {**task, "samples": 8}
    with pytest.raises(ValueError, match="task configuration"):
        jobs.validate_task_key(manifest, changed)


def test_orphan_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    context = {"code_sha": "revision", "source_sha256": {}, "inputs": {}}
    task = {"trial_start": 1, "trial_count": 2}
    task["task_key"] = jobs.object_hash({"context": context, "task": task})
    task["output"] = task["task_key"] + ".json"
    arrays = (tmp_path / task["output"]).with_suffix(".npz")
    arrays.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete or unrecognized"):
        jobs.task_state(context, task)
    assert arrays.read_bytes() == b"partial"
