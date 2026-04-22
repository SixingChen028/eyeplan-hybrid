#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-2

python -u simulate.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --learning_rate=${1} \
    --lamda_backup=${2} \
    --wm_decay=${3}
    
