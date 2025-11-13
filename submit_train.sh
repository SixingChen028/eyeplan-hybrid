#!/bin/bash

for learning_rate in 0.1 0.3 0.5
do
    sbatch run_train.sh ${learning_rate}
done