import csv
import glob
import json
import os
import statistics
from collections import defaultdict

BENCH_ROOT = os.path.join("results", "benchmarks", "jax-cpu-threads")
RAW_ROOT = os.path.join(BENCH_ROOT, "raw")
RAW_CSV = os.path.join(BENCH_ROOT, "benchmark_runs.csv")
SUMMARY_CSV = os.path.join(BENCH_ROOT, "benchmark_summary.csv")


def main() -> None:
    record_paths = sorted(glob.glob(os.path.join(RAW_ROOT, "*.json")))
    if not record_paths:
        raise FileNotFoundError(f"No benchmark records found under {RAW_ROOT}")

    rows = []
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)

    for path in record_paths:
        with open(path) as f:
            record = json.load(f)
        rows.append(record)
        if record["exit_status"] == 0:
            grouped[(record["cpus_per_task"], record["threads"])].append(record["runtime_s"])

    os.makedirs(BENCH_ROOT, exist_ok=True)

    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recorded_at_utc",
                "slurm_job_id",
                "slurm_task_id",
                "cpus_per_task",
                "thread_option",
                "threads",
                "repeat_idx",
                "runtime_s",
                "exit_status",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for (cpus, threads), runtimes in sorted(grouped.items()):
        mean_s = statistics.mean(runtimes)
        stdev_s = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
        summary_rows.append(
            {
                "cpus_per_task": cpus,
                "threads": threads,
                "n_success": len(runtimes),
                "mean_runtime_s": round(mean_s, 6),
                "stdev_runtime_s": round(stdev_s, 6),
                "throughput_rel": round(1.0 / mean_s, 9),
            }
        )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cpus_per_task",
                "threads",
                "n_success",
                "mean_runtime_s",
                "stdev_runtime_s",
                "throughput_rel",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {RAW_CSV}")
    print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
