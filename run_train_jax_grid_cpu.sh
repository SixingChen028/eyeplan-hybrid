#!/bin/bash
#SBATCH --job-name=ep-jax-grid
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -e ./results/slurm-grid-%A_%a.err
#SBATCH -o ./results/slurm-grid-%A_%a.out
#SBATCH --array=0-242

set -euo pipefail

# Force CPU execution for JAX.
export JAX_PLATFORMS=cpu

RESULT_PATH=./results/jax-grid-cpu
mkdir -p "${RESULT_PATH}"

# Three values for each hyperparameter requested as VARY.
BATCH_SIZES=(32 64 128)
LRS=(0.0003 0.001 0.003)
LAMDAS=(0.90 0.95 1.00)
BETA_E_INITS=(0.02 0.05 0.10)
SEEDS=(1 2 3)

N_BATCH=${#BATCH_SIZES[@]}
N_LR=${#LRS[@]}
N_LAMDA=${#LAMDAS[@]}
N_BETA_E_INIT=${#BETA_E_INITS[@]}
N_SEED=${#SEEDS[@]}
TOTAL=$((N_BATCH * N_LR * N_LAMDA * N_BETA_E_INIT * N_SEED))

TASK_ID=${SLURM_ARRAY_TASK_ID}
if ((TASK_ID < 0 || TASK_ID >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected [0, $((TOTAL - 1))]"
    exit 1
fi

idx=${TASK_ID}
seed_idx=$((idx % N_SEED)); idx=$((idx / N_SEED))
beta_e_init_idx=$((idx % N_BETA_E_INIT)); idx=$((idx / N_BETA_E_INIT))
lamda_idx=$((idx % N_LAMDA)); idx=$((idx / N_LAMDA))
lr_idx=$((idx % N_LR)); idx=$((idx / N_LR))
batch_idx=$((idx % N_BATCH))

BATCH_SIZE=${BATCH_SIZES[$batch_idx]}
LR=${LRS[$lr_idx]}
LAMDA=${LAMDAS[$lamda_idx]}
BETA_E_INIT=${BETA_E_INITS[$beta_e_init_idx]}
SEED=${SEEDS[$seed_idx]}

echo "grid_task task_id=${TASK_ID} batch_size=${BATCH_SIZE} lr=${LR} lamda=${LAMDA} beta_e_init=${BETA_E_INIT} seed=${SEED}"

python -u train_jax.py \
    --jobid="${TASK_ID}" \
    --path="${RESULT_PATH}" \
    --num_nodes=15 \
    --t_max=50 \
    --cost=0.01 \
    --beta_move=100.0 \
    --eps_move=0.0 \
    --learning_rate=1.0 \
    --wm_decay=1.0 \
    --lamda_backup=1.0 \
    --num_episodes=4000000 \
    --batch_size="${BATCH_SIZE}" \
    --hidden_size=256 \
    --lr="${LR}" \
    --gamma=1.0 \
    --lamda="${LAMDA}" \
    --beta_v=0.05 \
    --beta_e=0.02 \
    --beta_e_init="${BETA_E_INIT}" \
    --beta_e_final=0.001 \
    --max_grad_norm=1.0 \
    --seed="${SEED}"
