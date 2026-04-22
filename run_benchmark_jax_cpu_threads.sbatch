#!/bin/bash
#SBATCH --job-name=ep-jax-cpu-bench
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH -e ./results/benchmarks/jax-cpu-threads/slurm-%A_%a.err
#SBATCH -o ./results/benchmarks/jax-cpu-threads/slurm-%A_%a.out
#SBATCH --array=0-11

set -euo pipefail

BENCH_ROOT=./results/benchmarks/jax-cpu-threads
RUN_ROOT="${BENCH_ROOT}/runs"
RAW_ROOT="${BENCH_ROOT}/raw"
LOG_ROOT="${BENCH_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${RAW_ROOT}" "${LOG_ROOT}"

THREAD_OPTIONS=(1 2 4 auto)
REPEATS=3

N_THREADS=${#THREAD_OPTIONS[@]}
TOTAL=$((N_THREADS * REPEATS))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"
    exit 1
fi

thread_idx=$((TASK_ID % N_THREADS))
repeat_idx=$((TASK_ID / N_THREADS))
thread_option=${THREAD_OPTIONS[$thread_idx]}

if [[ "${thread_option}" == "auto" ]]; then
    threads=${SLURM_CPUS_PER_TASK}
else
    threads=${thread_option}
fi

if ((threads > SLURM_CPUS_PER_TASK)); then
    threads=${SLURM_CPUS_PER_TASK}
fi

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${threads}
export MKL_NUM_THREADS=${threads}
export OPENBLAS_NUM_THREADS=${threads}
export VECLIB_MAXIMUM_THREADS=${threads}
export NUMEXPR_NUM_THREADS=${threads}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${threads}"

log_path="${LOG_ROOT}/job${SLURM_JOB_ID}_task${TASK_ID}.log"
record_path="${RAW_ROOT}/job${SLURM_JOB_ID}_task${TASK_ID}.json"

echo "benchmark_start job_id=${SLURM_JOB_ID} task_id=${TASK_ID} cpus=${SLURM_CPUS_PER_TASK} thread_option=${thread_option} threads=${threads} repeat=${repeat_idx}" | tee "${log_path}"

start_ns=$(date +%s%N)
python -u train_jax.py \
    --jobid="bench_c${SLURM_CPUS_PER_TASK}_t${threads}_r${repeat_idx}" \
    --path="${RUN_ROOT}" \
    --num_nodes=15 \
    --t_max=50 \
    --cost=0.01 \
    --beta_move=100.0 \
    --eps_move=0.0 \
    --learning_rate=1.0 \
    --wm_decay=1.0 \
    --lamda_backup=1.0 \
    --num_episodes=64000 \
    --batch_size=64 \
    --hidden_size=256 \
    --lr=0.001 \
    --gamma=1.0 \
    --lamda=0.95 \
    --beta_v=0.05 \
    --beta_e=0.02 \
    --beta_e_init=0.05 \
    --beta_e_final=0.001 \
    --max_grad_norm=1.0 \
    --seed=$((100 + repeat_idx)) \
    --print_frequency=0 \
    --checkpoint_frequency=-1 \
    2>&1 | tee -a "${log_path}"
status=${PIPESTATUS[0]}
end_ns=$(date +%s%N)

runtime_s=$(python - <<'PY' "${start_ns}" "${end_ns}"
import sys
start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
print(f"{(end_ns - start_ns) / 1e9:.6f}")
PY
)

run_dir=$(grep -E '^run_dir=' "${log_path}" | tail -n 1 | cut -d'=' -f2- || true)

python - <<'PY' "${record_path}" "${SLURM_JOB_ID}" "${TASK_ID}" "${SLURM_CPUS_PER_TASK}" "${thread_option}" "${threads}" "${repeat_idx}" "${runtime_s}" "${status}" "${run_dir}"
import json
import sys
from datetime import datetime, timezone

record = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": int(sys.argv[2]),
    "slurm_task_id": int(sys.argv[3]),
    "cpus_per_task": int(sys.argv[4]),
    "thread_option": sys.argv[5],
    "threads": int(sys.argv[6]),
    "repeat_idx": int(sys.argv[7]),
    "runtime_s": float(sys.argv[8]),
    "exit_status": int(sys.argv[9]),
    "run_dir": sys.argv[10],
}

with open(sys.argv[1], "w") as f:
    json.dump(record, f, indent=2, sort_keys=True)
PY

if ((status != 0)); then
    echo "benchmark_failed job_id=${SLURM_JOB_ID} task_id=${TASK_ID} status=${status} runtime_s=${runtime_s}" | tee -a "${log_path}"
    exit "${status}"
fi

echo "benchmark_done job_id=${SLURM_JOB_ID} task_id=${TASK_ID} cpus=${SLURM_CPUS_PER_TASK} threads=${threads} repeat=${repeat_idx} runtime_s=${runtime_s} run_dir=${run_dir}" | tee -a "${log_path}"
