#!/bin/bash

export ALPHA=0.1
export SEED=90
export PARTICIPATION=1.0
export COHORTS=4
learning_rates=(0.0005 0.001 0.005 0.01 0.05 0.1)

for lr in "${learning_rates[@]}"; do
  python3 -u scripts/distill.py $PWD/data n_200_cifar10_dirichlet${ALPHA}00000_sd${SEED}_ct${COHORTS}_p${PARTICIPATION}_dfl cifar10 stl10 --epochs 100 --cohort-file cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt --cohort-participation-fraction $PARTICIPATION --public-data-dir /var/scratch/spandey/dfl-data --learning-rate $lr --momentum 0.9 --partitioner dirichlet --alpha $ALPHA --seed $SEED --weighting-scheme label --check-teachers-accuracy > output_distill_cifar10_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}_lr${lr}.log 2>&1 &
done

wait
echo "All processes are done"
