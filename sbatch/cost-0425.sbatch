#!/bin/bash
#SBATCH --job-name=cost-0425
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=1000M
#SBATCH -e ./log/%A_%a.log
#SBATCH -o ./log/%A_%a.log
#SBATCH --array=30-44

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
EXPERIMENT=cost-0425
mkdir -p "${RESULT_PATH}"

# Grid values from config.
COST_VALUES=(0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28)
SEED_VALUES=(1 2 3 4 5)

COST_N=${#COST_VALUES[@]}
SEED_N=${#SEED_VALUES[@]}
TOTAL=$((COST_N * SEED_N))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"
    exit 1
fi

idx=${TASK_ID}
seed_idx=$((idx % SEED_N)); idx=$((idx / SEED_N))
cost_idx=$((idx % COST_N)); idx=$((idx / COST_N))

COST_VALUE=${COST_VALUES[$cost_idx]}
SEED_VALUE=${SEED_VALUES[$seed_idx]}

echo "grid_task task_id=${TASK_ID} cost=${COST_VALUE} seed=${SEED_VALUE}"

"${PYTHON_BIN}" -u train.py \
    --jobid="${TASK_ID}" \
    --path="${RESULT_PATH}" \
    --experiment="${EXPERIMENT}" \
    --resume=true \
    --eval_episodes=102400 \
    --print_frequency=100 \
    --num_nodes=15 \
    --t_max=30 \
    --beta_move=100.0 \
    --eps_move=0.0 \
    --learning_rate=1.0 \
    --wm_decay=1.0 \
    --lamda_backup=1.0 \
    --hidden_size=256 \
    --num_episodes=16000000 \
    --batch_size=512 \
    --gamma=1.0 \
    --lr=0.001 \
    --lamda=0.8 \
    --beta_e_init=0.02 \
    --beta_e_final=0.01 \
    --beta_v=0.05 \
    --max_grad_norm=2.0 \
    --cost="${COST_VALUE}" \
    --seed="${SEED_VALUE}"

RUN=$(ls -td "${RESULT_PATH}/runs/${EXPERIMENT}/${TASK_ID}"_* | head -n 1)
"${PYTHON_BIN}" simulate.py "${RUN}"
"${PYTHON_BIN}" simulate.py "${RUN}" --detailed
