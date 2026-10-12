"""Allocation accounting with Slurm public array IDs and numeric raw task IDs."""

import csv

import pytest

from tools.report_action_fits import accounting_summary


def write_accounting(path, rows, *, public_ids=True):
    fields = ["JobIDRaw", "State", "ElapsedRaw", "AllocCPUS", "AllocTRES", "MaxRSS", "ReqMem"]
    if public_ids:
        fields.insert(0, "JobID")
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_public_array_ids_exclude_parent_and_steps_but_count_unique_failed_allocations(tmp_path):
    def row(public, raw, elapsed=3600, cpus=1, tres="cpu=1", state="COMPLETED", rss=""):
        return dict(JobID=public, JobIDRaw=raw, State=state, ElapsedRaw=elapsed, AllocCPUS=cpus,
                    AllocTRES=tres, MaxRSS=rss, ReqMem="4G")

    child = row("100_0", "110", cpus=2, tres="cpu=2,gres/gpu=2,gres/gpu:l40s=2")
    rows = [row("100", "100", cpus=20), child,
            row("100_0.batch", "110.batch", cpus=2, rss="2048M"),
            row("100_0.extern", "110.extern", cpus=2), child.copy(),
            row("100_1", "111", elapsed=1800, cpus=4, tres="cpu=4,gres/gpu:a100=1", state="FAILED"),
            row("200", "200", elapsed=900, cpus=2, tres="cpu=2,gres/gpu:a100=1,gres/gpu:l40s=2")]
    summary = accounting_summary(write_accounting(tmp_path / "sacct.psv", rows))
    assert summary["verified"]
    assert summary["allocation_count"] == 3
    assert summary["cpu_hours"] == pytest.approx(2 + 2 + .5)
    assert summary["gpu_hours"] == pytest.approx(2 + .5 + .75)
    assert summary["state_counts"] == {"COMPLETED": 2, "FAILED": 1}
    assert summary["peak_step_rss_gib"] == 2


@pytest.mark.parametrize("public_ids", [False, True])
def test_raw_array_id_fallback_when_public_id_missing_or_blank(tmp_path, public_ids):
    rows = [dict(JobIDRaw=raw, State="COMPLETED", ElapsedRaw=3600, AllocCPUS=1,
                 AllocTRES="cpu=1", MaxRSS="", ReqMem="4G")
            for raw in ["100", "100_0", "100_0.batch", "100_1"]]
    if public_ids:
        for row in rows:
            row["JobID"] = ""
    summary = accounting_summary(write_accounting(tmp_path / "raw.psv", rows, public_ids=public_ids))
    assert summary["allocation_count"] == 2
    assert summary["cpu_hours"] == 2
    assert summary["gpu_hours"] == 0
