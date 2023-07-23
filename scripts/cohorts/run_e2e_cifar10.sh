#!/bin/bash

# Ensure the script receives exactly two arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <number_of_cohorts> <seed> <alpha>"
    exit 1
fi

# Extract the arguments
COHORTS=$1
SEED=$2
ALPHA=$3

# Create the cohort configuration
python3 scripts/cohorts/create_cohort_file.py 200 $COHORTS --seed $SEED --partitioner dirichlet --alpha $ALPHA --output "data/cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt"

python3 -u simulations/dfl/cifar10.py --peers 200 --duration 0 --num-aggregators 1 --activity-log-interval 60 --bypass-model-transfers --seed $SEED --capability-trace data/client_device_capacity --accuracy-logging-interval 0 --partitioner dirichlet --alpha $ALPHA --fix-aggregator --cohort-file "cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt" --compute-validation-loss-global-model --train-device-name "cuda:0" --accuracy-device-name "cuda:0" --log-level "ERROR" > output_c${COHORTS}_s${SEED}_a${ALPHA}.log 2>&1

python3 -u scripts/distill.py $PWD/data n_200_cifar10_dirichlet${ALPHA}00000_sd${SEED}_ct${COHORTS}_dfl cifar10 stl10 --cohort-file cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt --public-data-dir /var/scratch/spandey/dfl-data --learning-rate 0.001 --momentum 0.9 --partitioner dirichlet --alpha $ALPHA --seed $SEED --weighting-scheme label --check-teachers-accuracy > output_distill_c${COHORTS}_s${SEED}_a${ALPHA}.log 2>&1