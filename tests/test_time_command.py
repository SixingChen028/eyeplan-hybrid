import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest


WRAPPER = Path(__file__).resolve().parents[1] / "tools" / "time_command.py"


def invoke(output, source):
    return subprocess.run(
        [sys.executable, str(WRAPPER), "--output", str(output), "--", sys.executable, "-c", source],
        text=True, capture_output=True,
    )


def test_command_output_resources_and_failure_receipt(tmp_path):
    output = tmp_path / "timing.json"
    result = invoke(output, "import sys; data=bytearray(4_000_000); print('child output'); sys.exit(7)")
    assert result.returncode == 7
    assert result.stdout == "child output\n"
    receipt = json.loads(output.read_text())
    assert receipt["returncode"] == 7
    assert receipt["elapsed_seconds"] > 0
    assert receipt["user_cpu_seconds"] >= 0
    assert receipt["system_cpu_seconds"] >= 0
    assert 3906 <= receipt["max_rss_kib"] < 1024 * 1024
    assert receipt["completed_utc"] >= receipt["started_utc"]
    assert not output.with_name(output.name + ".lock").exists()


@pytest.mark.parametrize("existing", ["output", "lock"])
def test_fresh_guard_prevents_child_execution(tmp_path, existing):
    output = tmp_path / "timing.json"
    guard = output if existing == "output" else output.with_name(output.name + ".lock")
    guard.write_text("original")
    side_effect = tmp_path / "child-ran"
    result = invoke(output, f"from pathlib import Path; Path({str(side_effect)!r}).touch()")
    assert result.returncode == 1
    assert not side_effect.exists()
    assert guard.read_text() == "original"


@pytest.mark.parametrize("child_signal", [signal.SIGTERM, signal.SIGKILL])
def test_signal_is_recorded_and_propagated(tmp_path, child_signal):
    output = tmp_path / "timing.json"
    result = invoke(output, f"import os; os.kill(os.getpid(),{int(child_signal)})")
    assert result.returncode == -child_signal
    assert json.loads(output.read_text())["returncode"] == -child_signal


def test_missing_executable_records_failure(tmp_path):
    output = tmp_path / "timing.json"
    result = subprocess.run([sys.executable, str(WRAPPER), "--output", str(output), "--", "/no/such/command"])
    assert result.returncode == 127
    receipt = json.loads(output.read_text())
    assert receipt["returncode"] == 127
    assert receipt["error"]
