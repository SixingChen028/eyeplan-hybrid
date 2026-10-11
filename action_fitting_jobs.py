"""Freeze, run, and audit resumable CPU likelihood-cache shards on Torch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = [
    "action_fitting_jobs.py", "lapse_likelihood_cache.py", "undirected_likelihood_cache.py",
    "modules/path_likelihood.py", "modules/path_likelihood_cache.py", "modules/undirected_likelihood_cache.py",
    "modules/environment.py", "modules/random_search.py", "modules/network.py", "modules/evaluation.py",
    "modules/a2c.py", "modules/compat.py", "modules/config.py",
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def repo_path(path):
    """Use portable repo-relative paths in frozen manifests."""
    return str(Path(path).resolve().relative_to(ROOT))


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def partition_trials(start, count, shard_size):
    if start < 0 or count < 1 or shard_size < 1:
        raise ValueError("trial start must be nonnegative; count and shard size must be positive")
    return [(offset, min(shard_size, start + count - offset)) for offset in range(start, start + count, shard_size)]


def generate(args):
    candidates = read_json(args.candidates)
    records = read_json(args.data)
    if not isinstance(records, list) or not records:
        raise ValueError("data must be a nonempty JSON list")
    count = len(records) - args.trial_start if args.limit is None else args.limit
    if args.trial_start + count > len(records):
        raise ValueError("trial range exceeds input data")
    shards = partition_trials(args.trial_start, count, args.shard_size)
    if args.samples < 2 or args.undirected_samples < 2 or args.repeats < 2 or args.undirected_repeats < 2:
        raise ValueError("samples and independent repeats must be at least two")
    if args.batch_size < 1 or args.seed_count is not None and args.seed_count < 1:
        raise ValueError("batch-size and seed-count must be positive")
    inputs = {name: {"path": repo_path(path), "sha256": file_hash(path)} for name, path in (
        ("data", args.data), ("candidates", args.candidates), ("split", args.split),
    )}
    sources = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    resources = {"cpus_per_task": 1, "memory": args.memory, "walltime": args.walltime,
                 "concurrency": args.concurrency}
    context = {"code_sha": revision(), "source_sha256": sources, "inputs": inputs}
    tasks = []
    selected_ids = set(args.candidate_id or [])
    seen_ids = set()
    for candidate in candidates["candidates"]:
        candidate_id = candidate["id"]
        if selected_ids and candidate_id not in selected_ids:
            continue
        if args.model and candidate["model"] not in args.model:
            continue
        if args.scope != "all" and candidate.get("scope", "primary") != args.scope:
            continue
        if candidate_id in seen_ids:
            raise ValueError("candidate IDs must be unique")
        seen_ids.add(candidate_id)
        runs = candidate["runs"]
        if args.training_seed:
            runs = [run for run in runs if run["training_seed"] in args.training_seed]
        if args.seed_count is not None:
            runs = sorted(runs, key=lambda run: run["training_seed"])[:args.seed_count]
        if not runs:
            raise ValueError(f"candidate {candidate_id} has no selected runs")
        undirected = candidate["model"] == "undirected"
        for start, length in shards:
            task = {"candidate_id": candidate_id, "model": candidate["model"], "runs": runs,
                    "parameters": candidate["parameters"], "trial_start": start, "trial_count": length,
                    "samples": args.undirected_samples if undirected else args.samples,
                    "repeats": args.undirected_repeats if undirected else args.repeats,
                    "batch_size": args.batch_size, "seed": args.seed}
            task["task_key"] = object_hash({"context": context, "task": task})
            task["output"] = str(Path(repo_path(args.cache_dir)) / (task["task_key"] + ".json"))
            tasks.append(task)
    if not tasks or selected_ids - seen_ids:
        raise ValueError(f"no tasks or unmatched candidate IDs: {sorted(selected_ids - seen_ids)}")
    manifest = {"schema_version": 1, **context, "resources": resources,
                "created_utc": datetime.now(timezone.utc).isoformat(), "tasks": tasks}
    write_json(args.output, manifest)
    print(json.dumps({"manifest": str(args.output), "tasks": len(tasks), "resources": resources}))


def validate_inputs(manifest, task):
    validate_task_key(manifest, task)
    if revision() != manifest["code_sha"]:
        raise ValueError("Git revision differs from frozen work manifest")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"source changed: {name}")
    for item in manifest["inputs"].values():
        if file_hash(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"input changed: {item['path']}")
    for run in task["runs"]:
        if file_hash(ROOT / run["run_dir"] / "metadata.json") != run["metadata_sha256"]:
            raise ValueError(f"metadata changed: {run['run_dir']}")
        if run.get("params_file") and file_hash(ROOT / run["params_file"]) != run["params_sha256"]:
            raise ValueError(f"checkpoint changed: {run['params_file']}")


def validate_task_key(manifest, task):
    context = {key: manifest[key] for key in ("code_sha", "source_sha256", "inputs")}
    configuration = {key: value for key, value in task.items() if key not in ("task_key", "output")}
    if object_hash({"context": context, "task": configuration}) != task["task_key"]:
        raise ValueError("task configuration or context differs from frozen task key")
    if Path(task["output"]).name != task["task_key"] + ".json":
        raise ValueError("output filename differs from frozen task key")


def validate_cache(manifest, task):
    path = ROOT / task["output"]
    cache = read_json(path)
    checks = {"code_sha": manifest["code_sha"], "data_sha256": manifest["inputs"]["data"]["sha256"],
              "trial_start": task["trial_start"], "trial_count": task["trial_count"], "repeats": task["repeats"],
              "batch_size": task["batch_size"], "seed": task["seed"]}
    checks["samples_per_repeat" if task["model"] == "undirected" else "samples"] = task["samples"]
    for key, expected in checks.items():
        if cache.get(key) != expected:
            raise ValueError(f"cache {key} differs from task {task['task_key']}")
    expected_runs = task["runs"]
    if len(cache["runs"]) != len(expected_runs):
        raise ValueError("cache run count differs from task")
    for cached, expected in zip(cache["runs"], expected_runs, strict=True):
        if cached["metadata_sha256"] != expected["metadata_sha256"]:
            raise ValueError("cache metadata hash differs from task")
        if cached.get("params_sha256") != expected.get("params_sha256"):
            raise ValueError("cache checkpoint hash differs from task")
        if len(cached["estimates"]) != task["trial_count"]:
            raise ValueError("cache trial count differs from task")
    arrays_path = path.with_suffix(".npz")
    if cache["arrays_file"] != arrays_path.name or file_hash(arrays_path) != cache["arrays_sha256"]:
        raise ValueError("cache array identity or hash differs")
    expected_sources = manifest["source_sha256"]
    for name, expected in cache["source_sha256"].items():
        if expected_sources.get(name) != expected:
            raise ValueError(f"cache source hash differs: {name}")
    return {"manifest_sha256": file_hash(path), "arrays_sha256": cache["arrays_sha256"]}


def task_state(manifest, task):
    validate_task_key(manifest, task)
    path = ROOT / task["output"]
    request_path = path.with_suffix(".request.json")
    receipt_path = path.with_suffix(".complete.json")
    if request_path.exists() and read_json(request_path).get("task_key") != task["task_key"]:
        raise ValueError("existing request has a different task key")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        hashes = validate_cache(manifest, task)
        if receipt.get("task_key") != task["task_key"] or receipt.get("outputs") != hashes:
            raise ValueError("completion receipt differs from frozen task or output hashes")
        return "complete"
    if path.exists() and path.with_suffix(".npz").exists() and request_path.exists():
        validate_cache(manifest, task)
        return "needs_receipt"
    if path.exists() or path.with_suffix(".npz").exists():
        raise ValueError("incomplete or unrecognized output exists; retain or rename before retry")
    if path.with_suffix(".lock").exists():
        return "locked"
    return "pending"


def run_task(args):
    manifest = read_json(args.manifest)
    task = manifest["tasks"][args.task_index]
    if args.task_index < 0:
        raise ValueError("task index must be nonnegative")
    validate_inputs(manifest, task)
    state = task_state(manifest, task)
    if state == "complete":
        print(f"verified complete; skipped {task['task_key']}")
        return
    path = ROOT / task["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    with lock.open("x") as file:
        file.write(f"pid={os.getpid()} job={os.getenv('SLURM_JOB_ID', 'local')}\n")
    try:
        request_path = path.with_suffix(".request.json")
        if not request_path.exists():
            write_json(request_path, {"task_key": task["task_key"], "task": task,
                                      "work_manifest_sha256": file_hash(args.manifest)})
        started = time.perf_counter()
        command = [sys.executable, str(ROOT / ("undirected_likelihood_cache.py" if task["model"] == "undirected"
                                               else "lapse_likelihood_cache.py"))]
        command += [str(ROOT / run["run_dir"]) for run in task["runs"]]
        command += ["--data", str(ROOT / manifest["inputs"]["data"]["path"]), "--output", str(path)]
        for option, key in (("samples", "samples"), ("batch-size", "batch_size"), ("repeats", "repeats"),
                            ("seed", "seed"), ("trial-start", "trial_start"), ("limit", "trial_count")):
            command += ["--" + option, str(task[key])]
        if state != "needs_receipt":
            subprocess.run(command, cwd=ROOT, check=True)
        hashes = validate_cache(manifest, task)
        receipt = {"task_key": task["task_key"], "outputs": hashes, "command": command,
                   "work_manifest_sha256": file_hash(args.manifest), "elapsed_seconds": time.perf_counter() - started,
                   "recovered_completed_output": state == "needs_receipt", "requested_resources": manifest["resources"],
                   "slurm": {key: os.getenv(key) for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                             "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURMD_NODENAME")},
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(path.with_suffix(".complete.json"), receipt)
        print(f"verified complete {task['task_key']}")
    finally:
        lock.unlink()


def status(args):
    manifest = read_json(args.manifest)
    states = {"complete": [], "pending": [], "locked": [], "needs_receipt": [], "invalid": []}
    for index, task in enumerate(manifest["tasks"]):
        try:
            states[task_state(manifest, task)].append(index)
        except (OSError, ValueError, KeyError) as error:
            states["invalid"].append({"task_index": index, "error": str(error)})
    print(json.dumps({"counts": {key: len(value) for key, value in states.items()}, **states}, indent=2))


def slurm_script(args):
    manifest = read_json(args.manifest)
    resources = manifest["resources"]
    hours, minutes, seconds = map(int, resources["walltime"].split(":"))
    if not 0 < hours * 3600 + minutes * 60 + seconds < 4 * 3600:
        raise ValueError("walltime must be positive and below four hours")
    if resources["concurrency"] < 1:
        raise ValueError("concurrency must be positive")
    indices = args.task_index if args.task_index is not None else list(range(len(manifest["tasks"])))
    if not indices or min(indices) < 0 or max(indices) >= len(manifest["tasks"]):
        raise ValueError("invalid Slurm task indices")
    array = ",".join(map(str, indices)) if args.task_index is not None else f"0-{len(indices) - 1}"
    repo = args.repo.rstrip("/")
    for value in (repo, args.log_dir, resources["memory"], args.python):
        if "\n" in value or "\r" in value:
            raise ValueError("script values cannot contain newlines")
    script = f'''#!/bin/bash
#SBATCH --job-name=action_fit
#SBATCH --cpus-per-task=1
#SBATCH --mem={resources["memory"]}
#SBATCH --time={resources["walltime"]}
#SBATCH --array={array}%{resources["concurrency"]}
#SBATCH --chdir={shlex.quote(repo)}
#SBATCH --output={shlex.quote(args.log_dir + "/%A_%a.log")}
#SBATCH --error={shlex.quote(args.log_dir + "/%A_%a.log")}

set -euo pipefail
TASK_TMP=$(mktemp -d "${{SLURM_TMPDIR:-${{TMPDIR:-/tmp}}}}/action_fit.${{SLURM_JOB_ID:-local}}.${{SLURM_ARRAY_TASK_ID:-0}}.XXXXXX")
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1 intra_op_parallelism_threads=1"
{shlex.quote(args.python)} action_fitting_jobs.py run --manifest {shlex.quote(repo_path(args.manifest))} --task-index "$SLURM_ARRAY_TASK_ID"
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(script)
    print(f"wrote={args.output}; create remote log directory before sbatch: {args.log_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    for name in ("candidates", "data", "split", "output", "cache-dir"):
        generate_parser.add_argument("--" + name, type=Path, required=True)
    generate_parser.add_argument("--candidate-id", action="append")
    generate_parser.add_argument("--model", choices=("main", "unbounded", "undirected"), action="append")
    generate_parser.add_argument("--scope", choices=("primary", "sensitivity", "all"), default="primary")
    generate_parser.add_argument("--training-seed", type=int, action="append")
    generate_parser.add_argument("--seed-count", type=int)
    generate_parser.add_argument("--trial-start", type=int, default=0)
    generate_parser.add_argument("--limit", type=int)
    generate_parser.add_argument("--shard-size", type=int, default=1000)
    generate_parser.add_argument("--samples", type=int, default=1024)
    generate_parser.add_argument("--undirected-samples", type=int, default=256)
    generate_parser.add_argument("--repeats", type=int, default=2)
    generate_parser.add_argument("--undirected-repeats", type=int, default=4)
    generate_parser.add_argument("--batch-size", type=int, default=128)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--memory", default="8G")
    generate_parser.add_argument("--walltime", default="03:59:00")
    generate_parser.add_argument("--concurrency", type=int, default=32)
    generate_parser.set_defaults(function=generate)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.set_defaults(function=run_task)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--manifest", type=Path, required=True)
    status_parser.set_defaults(function=status)
    script_parser = commands.add_parser("slurm")
    script_parser.add_argument("--manifest", type=Path, required=True)
    script_parser.add_argument("--output", type=Path, required=True)
    script_parser.add_argument("--repo", required=True)
    script_parser.add_argument("--python", default=".venv/bin/python")
    script_parser.add_argument("--log-dir", required=True)
    script_parser.add_argument("--task-index", type=int, action="append")
    script_parser.set_defaults(function=slurm_script)
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, KeyError, IndexError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
