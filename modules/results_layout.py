import json
import os
import random
import string
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone

from modules.pipeline_compat import PIPELINE_COMPAT_KEY, PIPELINE_COMPAT_VERSION


@dataclass(frozen=True)
class ResolvedAnalysisTarget:
    experiment: str
    run_dirs: list[str]
    kind: str  # "experiment" or "run"


def _to_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _is_run_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.json"))


def _random_suffix(length: int = 4) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def _normalize_prefix(prefix: str | None) -> str | None:
    if prefix is None:
        return None
    value = str(prefix).strip()
    if value == "":
        return None
    return value


def get_experiment_runs_dir(results_root: str, experiment: str) -> str:
    return _to_abs(os.path.join(results_root, "runs", experiment))


def list_experiment_run_dirs(results_root: str, experiment: str) -> list[str]:
    root = get_experiment_runs_dir(results_root, experiment)
    if not os.path.isdir(root):
        return []

    run_dirs = [
        _to_abs(entry.path)
        for entry in os.scandir(root)
        if entry.is_dir() and _is_run_dir(entry.path)
    ]
    run_dirs.sort()
    return run_dirs


def create_run_dir(
    results_root: str,
    experiment: str,
    prefix: str | None = None,
    timestamp: str | None = None,
    suffix: str | None = None,
) -> str:
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    experiment_runs_dir = get_experiment_runs_dir(results_root, experiment)
    os.makedirs(experiment_runs_dir, exist_ok=True)

    for _ in range(16):
        run_name = _run_dir_name(prefix=prefix, timestamp=timestamp, suffix=suffix or _random_suffix())
        run_dir = os.path.join(experiment_runs_dir, run_name)
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            if suffix is not None:
                raise

    raise RuntimeError("Unable to create a unique run directory after multiple attempts.")


def _run_dir_name(prefix: str | None, timestamp: str, suffix: str) -> str:
    normalized_prefix = _normalize_prefix(prefix)
    stem = timestamp if normalized_prefix is None else f"{normalized_prefix}_{timestamp}"
    return f"{stem}_{suffix}"


def resolve_analysis_target(target: str, results_root: str) -> ResolvedAnalysisTarget:
    results_root = _to_abs(results_root)
    if not target or not target.strip():
        raise ValueError("target must be non-empty")

    target = target.strip()
    if target.endswith("/*"):
        raise ValueError("Wildcard targets are not supported. Use the experiment name or runs path.")

    expanded_target = os.path.expanduser(target)
    if os.path.exists(expanded_target):
        abs_target = _to_abs(expanded_target)
        rel_parts = _relative_results_parts(abs_target, results_root)
        if rel_parts is None:
            raise ValueError(f"Target path must be under the results root: {abs_target}")

        if len(rel_parts) == 2 and rel_parts[0] == "runs":
            experiment = rel_parts[1]
            run_dirs = list_experiment_run_dirs(results_root, experiment)
            if not run_dirs:
                raise FileNotFoundError(f"No run directories found for experiment: {experiment}")
            return ResolvedAnalysisTarget(experiment=experiment, run_dirs=run_dirs, kind="experiment")

        if len(rel_parts) == 3 and rel_parts[0] == "runs":
            experiment = rel_parts[1]
            if not _is_run_dir(abs_target):
                raise FileNotFoundError(f"Target path is not a run directory: {abs_target}")
            return ResolvedAnalysisTarget(experiment=experiment, run_dirs=[abs_target], kind="run")

        raise ValueError(
            "Target path must be results/runs/<experiment> or results/runs/<experiment>/<run_id>."
        )

    if os.sep in target or "/" in target:
        raise ValueError(
            "Unsupported target format. Use <experiment>, results/runs/<experiment>, "
            "or results/runs/<experiment>/<run_id>."
        )

    run_dirs = list_experiment_run_dirs(results_root, target)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for experiment: {target}")
    return ResolvedAnalysisTarget(experiment=target, run_dirs=run_dirs, kind="experiment")


def _relative_results_parts(path: str, results_root: str) -> list[str] | None:
    try:
        if os.path.commonpath([path, results_root]) != results_root:
            return None
        rel = os.path.relpath(path, results_root)
    except ValueError:
        return None
    return rel.split(os.sep)


def get_run_analysis_dir(results_root: str, experiment: str, run_id: str) -> str:
    return os.path.join(_to_abs(results_root), "analysis", experiment, "runs", run_id)


def get_summary_analysis_dir(results_root: str, experiment: str) -> str:
    return os.path.join(_to_abs(results_root), "analysis", experiment, "summary")


def _get_git_sha(cwd: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def write_run_metadata(run_dir: str, args: Namespace, cwd: str | None = None) -> str:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "started_at_utc": timestamp_utc,
        "git_sha": _get_git_sha(cwd=cwd),
        PIPELINE_COMPAT_KEY: PIPELINE_COMPAT_VERSION,
        "argv": sys.argv,
        "args": vars(args),
    }

    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    return metadata_path
