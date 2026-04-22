import os
import re
import json
import sys
import time
import subprocess
from datetime import datetime, timezone
from argparse import Namespace

_TIMESTAMP_PATTERN = re.compile(r"^(?:(?P<prefix>.+)_)?(?P<timestamp>\d{8}_\d{6})$")


def _normalize_prefix(jobid: str | None) -> str | None:
    if jobid is None:
        return None
    value = str(jobid).strip()
    if value in {"", "0"}:
        return None
    return value


def build_timestamped_run_dir(path: str, jobid: str | None = None, timestamp: str | None = None) -> str:
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    prefix = _normalize_prefix(jobid)
    run_name = timestamp if prefix is None else f"{prefix}_{timestamp}"
    return os.path.join(path, run_name)


def create_timestamped_run_dir(path: str, jobid: str | None = None, timestamp: str | None = None) -> str:
    run_dir = build_timestamped_run_dir(path=path, jobid=jobid, timestamp=timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def resolve_timestamped_run_dir(path: str, run_dir: str | None = None, jobid: str | None = None) -> str:
    if run_dir is not None:
        return run_dir

    prefix = _normalize_prefix(jobid)
    candidates: list[tuple[str, str]] = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if not os.path.isdir(full_path):
            continue

        match = _TIMESTAMP_PATTERN.match(name)
        if match is None:
            continue

        if prefix is not None and match.group("prefix") != prefix:
            continue

        candidates.append((match.group("timestamp"), full_path))

    if not candidates:
        if prefix is None:
            raise FileNotFoundError(f"No timestamped run directories found under: {path}")
        raise FileNotFoundError(
            f"No timestamped run directories found under {path} for jobid={prefix}"
        )

    candidates.sort(key=lambda item: item[0])
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
