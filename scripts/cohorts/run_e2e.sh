#!/bin/bash

# Ensure the script receives exactly two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_cohorts> <seed>"
    exit 1
fi

# Extract the arguments
COHORTS=$1
SEED=$2

# Command 1
python3 -u simulations/dfl/cifar10.py --peers 200 --duration 0 --num-aggregators 1 --activity-log-interval 60 --bypass-model-transfers --seed $SEED --capability-trace data/client_device_capacity --instant-network --partitioner dirichlet --alpha 0.1 --fix-aggregator --cohort-file "cohorts/cohorts_cifar10_n200_c${COHORTS}.txt" --compute-validation-loss-global-model --train-device-name "cuda:0" --accuracy-device-name "cuda:0" > output_c${COHORTS}_s${SEED}.log 2>&1

# Command 2
python3 -u scripts/distill.py $PWD/data n_200_cifar10_dirichlet0.100000_sd${SEED}_ct${COHORTS}_dfl cifar10 stl10 --cohort-file cohorts/cohorts_cifar10_n200_c${COHORTS}.txt --public-data-dir /var/scratch/spandey/dfl-data --learning-rate 0.001 --momentum 0.9 --partitioner dirichlet --alpha 0.1 --weighting-scheme label --check-teachers-accuracy > output_distill_c${COHORTS}_s${SEED}.log 2>&1
