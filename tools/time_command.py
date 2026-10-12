"""Run a command and save portable wall time, CPU time, and peak child RSS."""

import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_receipt(path, receipt):
    """Publish a complete receipt without replacing an existing destination."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(receipt, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink()
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def run_timed(command, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Timing output already exists: {output}")
    lock = output.with_name(output.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(descriptor)
        if output.exists():
            raise FileExistsError(f"Timing output already exists: {output}")
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started_utc = utc_now()
        started = time.monotonic()
        error = None
        try:
            returncode = subprocess.run(command, check=False).returncode
        except OSError as exception:
            returncode = 127 if isinstance(exception, FileNotFoundError) else 126
            error = str(exception)
        elapsed = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt = {
            "schema_version": 1,
            "command": list(command),
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "user_cpu_seconds": after.ru_utime - before.ru_utime,
            "system_cpu_seconds": after.ru_stime - before.ru_stime,
            "max_rss_kib": after.ru_maxrss / 1024 if sys.platform == "darwin" else after.ru_maxrss,
            "returncode": returncode,
        }
        if error is not None:
            receipt["error"] = error
        atomic_receipt(output, receipt)
        return returncode
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    try:
        returncode = run_timed(command, args.output)
    except FileExistsError as exception:
        parser.exit(1, f"{exception}\n")
    if returncode < 0:
        child_signal = -returncode
        if child_signal not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(child_signal, signal.SIG_DFL)
        os.kill(os.getpid(), child_signal)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
