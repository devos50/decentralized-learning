#!/bin/bash

export ALPHA=0.1
export SEED=90
export PARTICIPATION=0.2
export COHORTS=4
learning_rates=(0.0005 0.001 0.005 0.01 0.05 0.1)

for lr in "${learning_rates[@]}"; do
  python3 -u scripts/distill.py $PWD/data n_1000_femnist_dirichlet${ALPHA}00000_sd${SEED}_ct${COHORTS}_p${PARTICIPATION}_dfl femnist svhn --cohort-file cohorts/cohorts_n1000_c${COHORTS}_s${SEED}_a${ALPHA}.txt --cohort-participation-fraction $PARTICIPATION --public-data-dir /var/scratch/spandey/dfl-data/svhn --private-data-dir /var/scratch/spandey/leaf/femnist --learning-rate 0.001 --momentum 0.9 --partitioner dirichlet --alpha $ALPHA --seed $SEED --weighting-scheme label --check-teachers-accuracy &
done

wait
echo "All processes are done"
