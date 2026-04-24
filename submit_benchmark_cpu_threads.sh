#!/bin/bash
set -euo pipefail

BENCH_ROOT=./results/benchmarks/jax-cpu-threads
mkdir -p "${BENCH_ROOT}"

CPUS_VALUES=(1 2 4 8)
ARRAY_SPEC=0-11

manifest="${BENCH_ROOT}/submitted_jobs.txt"
: > "${manifest}"

for cpus in "${CPUS_VALUES[@]}"; do
    job_id=$(sbatch --parsable --cpus-per-task="${cpus}" --array="${ARRAY_SPEC}" run_benchmark_cpu_threads.sbatch)
    echo "cpus_per_task=${cpus} job_id=${job_id}" | tee -a "${manifest}"
done

echo "submitted_manifest=${manifest}"
