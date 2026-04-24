#!/bin/bash
#SBATCH --job-name=hybrid-dyna
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH -e ./log/%A_%a.log
#SBATCH -o ./log/%A_%a.log
#SBATCH --array=0-71

set -euo pipefail

PYTHON_BIN=python
if [[ -x ".venv/bin/python" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source ".venv/bin/activate"
    fi
    PYTHON_BIN=".venv/bin/python"
fi

# Force CPU execution for JAX.
export JAX_PLATFORMS=cpu
THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
export VECLIB_MAXIMUM_THREADS=${THREADS}
export NUMEXPR_NUM_THREADS=${THREADS}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${THREADS}"

RESULT_PATH=./results
EXPERIMENT=hybrid-dyna
mkdir -p "${RESULT_PATH}"

# Grid values from config.
SEED_VALUES=(1 2 3)
BATCH_SIZE_VALUES=(64 128)
LR_VALUES=(0.0001 0.0005 0.001)
LAMDA_VALUES=(0.9 0.95)
BETA_E_INIT_VALUES=(0.05 0.02)

SEED_N=${#SEED_VALUES[@]}
BATCH_SIZE_N=${#BATCH_SIZE_VALUES[@]}
LR_N=${#LR_VALUES[@]}
LAMDA_N=${#LAMDA_VALUES[@]}
BETA_E_INIT_N=${#BETA_E_INIT_VALUES[@]}
TOTAL=$((SEED_N * BATCH_SIZE_N * LR_N * LAMDA_N * BETA_E_INIT_N))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"
    exit 1
fi

idx=${TASK_ID}
beta_e_init_idx=$((idx % BETA_E_INIT_N)); idx=$((idx / BETA_E_INIT_N))
lamda_idx=$((idx % LAMDA_N)); idx=$((idx / LAMDA_N))
lr_idx=$((idx % LR_N)); idx=$((idx / LR_N))
batch_size_idx=$((idx % BATCH_SIZE_N)); idx=$((idx / BATCH_SIZE_N))
seed_idx=$((idx % SEED_N)); idx=$((idx / SEED_N))

SEED_VALUE=${SEED_VALUES[$seed_idx]}
BATCH_SIZE_VALUE=${BATCH_SIZE_VALUES[$batch_size_idx]}
LR_VALUE=${LR_VALUES[$lr_idx]}
LAMDA_VALUE=${LAMDA_VALUES[$lamda_idx]}
BETA_E_INIT_VALUE=${BETA_E_INIT_VALUES[$beta_e_init_idx]}

echo "grid_task task_id=${TASK_ID} seed=${SEED_VALUE} batch_size=${BATCH_SIZE_VALUE} lr=${LR_VALUE} lamda=${LAMDA_VALUE} beta_e_init=${BETA_E_INIT_VALUE}"

"${PYTHON_BIN}" -u train_jax.py \
    --jobid="${TASK_ID}" \
    --path="${RESULT_PATH}" \
    --experiment="${EXPERIMENT}" \
    --resume=true \
    --eval_episodes=102400 \
    --print_frequency=100 \
    --num_nodes=15 \
    --t_max=30 \
    --cost=0.02 \
    --beta_move=100.0 \
    --eps_move=0.0 \
    --learning_rate=1.0 \
    --wm_decay=0.8 \
    --lamda_backup=0.0 \
    --hidden_size=256 \
    --num_episodes=16000000 \
    --gamma=1.0 \
    --beta_e_final=0.0001 \
    --beta_v=0.05 \
    --max_grad_norm=2.0 \
    --seed="${SEED_VALUE}" \
    --batch_size="${BATCH_SIZE_VALUE}" \
    --lr="${LR_VALUE}" \
    --lamda="${LAMDA_VALUE}" \
    --beta_e_init="${BETA_E_INIT_VALUE}"
