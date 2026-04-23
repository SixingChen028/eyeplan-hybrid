#!/bin/bash
#SBATCH --job-name=ep-jax-grid-test
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -e ./log/grid-test-%A_%a.err
#SBATCH -o ./log/grid-test-%A_%a.out
#SBATCH --array=0-8

set -euo pipefail

# Force CPU execution for JAX.
export JAX_PLATFORMS=cpu
THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
export VECLIB_MAXIMUM_THREADS=${THREADS}
export NUMEXPR_NUM_THREADS=${THREADS}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS}"

RESULT_PATH=./results/jax-grid-cpu-test
mkdir -p "${RESULT_PATH}"

# Test sweep: vary only batch size and seed (3 x 3 = 9 jobs).
BATCH_SIZES=(32 64 128)
SEEDS=(1 2 3)

N_BATCH=${#BATCH_SIZES[@]}
N_SEED=${#SEEDS[@]}
TOTAL=$((N_BATCH * N_SEED))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"
    exit 1
fi

idx=${TASK_ID}
seed_idx=$((idx % N_SEED)); idx=$((idx / N_SEED))
batch_idx=$((idx % N_BATCH))

BATCH_SIZE=${BATCH_SIZES[$batch_idx]}
SEED=${SEEDS[$seed_idx]}

echo "grid_test_task task_id=${TASK_ID} batch_size=${BATCH_SIZE} seed=${SEED}"

python -u train_jax.py \
    --jobid="test_${TASK_ID}" \
    --path="${RESULT_PATH}" \
    --resume=true \
    --num_nodes=15 \
    --t_max=30 \
    --cost=0.01 \
    --beta_move=100.0 \
    --eps_move=0.0 \
    --learning_rate=1.0 \
    --wm_decay=1.0 \
    --lamda_backup=1.0 \
    --num_episodes=1_000_000 \
    --batch_size="${BATCH_SIZE}" \
    --hidden_size=256 \
    --lr=0.001 \
    --gamma=1.0 \
    --lamda=0.95 \
    --beta_v=0.05 \
    --beta_e=0.02 \
    --beta_e_init=0.05 \
    --beta_e_final=0.0001 \
    --max_grad_norm=2.0 \
    --seed="${SEED}"
