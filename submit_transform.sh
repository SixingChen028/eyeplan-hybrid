#!/bin/bash

for lamda_backup in 0.0 0.2 0.4 0.6 0.8 1.0
do
    for wm_decay in 0.0 0.2 0.4 0.6 0.8 1.0
    do
        sbatch run_transform.sh ${lamda_backup} ${wm_decay}
    done
done