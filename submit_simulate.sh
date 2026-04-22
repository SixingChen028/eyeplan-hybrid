#!/bin/bash

for learning_rate in 0.1 0.5 1.0
do
    for lamda_backup in 0.0 0.2 0.4 0.6 0.8 1.0
    do
        for wm_decay in 0.0 0.2 0.4 0.6 0.8 1.0
        do
            sbatch run_simulate.sh ${learning_rate} ${lamda_backup} ${wm_decay}
        done
    done
done
