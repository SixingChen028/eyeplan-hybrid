import os
import re
import json
import sys
import time
import random
import string
import subprocess
from datetime import datetime, timezone
from argparse import Namespace

from .analysis_targets import get_experiment_runs_dir, list_experiment_candidate_dirs

_TIMESTAMP_PATTERN = re.compile(
    r"^(?:(?P<prefix>.+)_)?(?P<timestamp>\d{8}_\d{6})(?:_(?P<suffix>[a-z0-9]{4}))?$"
)


def _normalize_prefix(jobid: str | None) -> str | None:
    if jobid is None:
        return None
    value = str(jobid).strip()
    if value in {"", "0"}:
        return None
    return value


def _random_suffix(length: int = 4) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def build_timestamped_run_dir(
    path: str,
    experiment: str = "default",
    jobid: str | None = None,
    timestamp: str | None = None,
    suffix: str | None = None,
) -> str:
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    if suffix is None:
        suffix = _random_suffix()

    prefix = _normalize_prefix(jobid)
    stem = timestamp if prefix is None else f"{prefix}_{timestamp}"
    run_name = f"{stem}_{suffix}"
    experiment_runs_dir = get_experiment_runs_dir(path, experiment)
    return os.path.join(experiment_runs_dir, run_name)


def create_timestamped_run_dir(
    path: str,
    experiment: str = "default",
    jobid: str | None = None,
    timestamp: str | None = None,
) -> str:
    experiment_runs_dir = get_experiment_runs_dir(path, experiment)
    os.makedirs(experiment_runs_dir, exist_ok=True)
    for _ in range(16):
        run_dir = build_timestamped_run_dir(
            path=path,
            experiment=experiment,
            jobid=jobid,
            timestamp=timestamp,
        )
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError("Unable to create a unique run directory after multiple attempts.")


def resolve_timestamped_run_dir(
    path: str,
    experiment: str = "default",
    run_dir: str | None = None,
    jobid: str | None = None,
) -> str:
    if run_dir is not None:
        return run_dir

    prefix = _normalize_prefix(jobid)
    candidates: list[tuple[str, str]] = []
    for full_path in list_experiment_candidate_dirs(path, experiment):
        name = os.path.basename(full_path)

        match = _TIMESTAMP_PATTERN.match(name)
        if match is None:
            continue

        if prefix is not None and match.group("prefix") != prefix:
            continue

        candidates.append((match.group("timestamp"), full_path))

    if not candidates:
        if prefix is None:
            raise FileNotFoundError(
                f"No timestamped run directories found under: {get_experiment_runs_dir(path, experiment)}"
            )
        raise FileNotFoundError(
            f"No timestamped run directories found under {get_experiment_runs_dir(path, experiment)} "
            f"for jobid={prefix}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][1]


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
        "argv": sys.argv,
        "args": vars(args),
    }

    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    return metadata_path
