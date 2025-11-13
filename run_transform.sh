#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-2

python -u transform.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --learning_rate=${1}
    