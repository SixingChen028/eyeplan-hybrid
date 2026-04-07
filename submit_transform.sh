#!/bin/bash

for learning_rate in 0.1 0.5 1.0
do
    for wm_decay in 0.0 0.2 0.4 0.6 0.8 1.0
    do
        sbatch run_transform.sh ${learning_rate} ${wm_decay}
    done
done