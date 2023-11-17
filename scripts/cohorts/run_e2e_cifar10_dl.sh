#!/bin/bash

# Ensure the script receives exactly two arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <number_of_cohorts> <seed> <alpha> <participation_rate>"
    exit 1
fi

# Extract the arguments
COHORTS=$1
SEED=$2
ALPHA=$3
PARTICIPATION=$4

# Create the cohort configuration
python3 scripts/cohorts/create_cohort_file.py 200 $COHORTS --seed $SEED --partitioner dirichlet --alpha $ALPHA --output "data/cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt"

python3 -u simulations/dl/cifar10.py --peers 200 --duration 0 --activity-log-interval 60 --bypass-model-transfers --seed $SEED --capability-trace data/client_device_capacity --accuracy-logging-interval 0 --partitioner dirichlet --alpha $ALPHA --fix-aggregator --cohort-file "cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt" --compute-validation-loss-global-model > output_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}.log 2>&1

python3 -u scripts/distill.py $PWD/data n_200_cifar10_dirichlet${ALPHA}_sd${SEED}_dl_ct${COHORTS}_p${PARTICIPATION} cifar10 stl10 --cohort-file cohorts/cohorts_n200_c${COHORTS}_s${SEED}_a${ALPHA}.txt --cohort-participation-fraction $PARTICIPATION --public-data-dir /var/scratch/spandey/dfl-data --learning-rate 0.001 --momentum 0.9 --partitioner dirichlet --alpha $ALPHA --seed $SEED --weighting-scheme label --check-teachers-accuracy --model-selection last > output_distill_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}.log 2>&1