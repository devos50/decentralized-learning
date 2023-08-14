#!/bin/bash

# Ensure the script receives exactly two arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <number_of_cohorts> <seed> <alpha> <participation>"
    exit 1
fi

# Extract the arguments
COHORTS=$1
SEED=$2
ALPHA=$3
PARTICIPATION=$4

# Create the cohort configuration
python3 scripts/cohorts/create_cohort_file.py 1000 $COHORTS --seed $SEED --partitioner dirichlet --alpha $ALPHA --output "data/cohorts/cohorts_n1000_c${COHORTS}_s${SEED}_a${ALPHA}.txt" --dataset femnist --data-dir /var/scratch/spandey/leaf/femnist

python3 -u simulations/dfl/femnist.py --dataset-base-path "/var/scratch/spandey" --peers 1000 --cohort-participation-fraction $PARTICIPATION --duration 0 --num-aggregators 1 --activity-log-interval 60 --bypass-model-transfers --seed $SEED --stop-criteria-patience 200 --capability-trace data/client_device_capacity --accuracy-logging-interval 0 --partitioner dirichlet --alpha $ALPHA --fix-aggregator --cohort-file "cohorts/cohorts_n1000_c${COHORTS}_s${SEED}_a${ALPHA}.txt" --compute-validation-loss-global-model --train-device-name "cuda:0" --accuracy-device-name "cuda:0" --log-level "ERROR" --accuracy-logging-interval 10 > output_femnist_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}.log 2>&1

python3 -u scripts/distill.py $PWD/data n_1000_femnist_dirichlet${ALPHA}00000_sd${SEED}_ct${COHORTS}_p${PARTICIPATION}_dfl femnist svhn --cohort-file cohorts/cohorts_n1000_c${COHORTS}_s${SEED}_a${ALPHA}.txt --cohort-participation-fraction $PARTICIPATION --public-data-dir /var/scratch/spandey/dfl-data/svhn --private-data-dir /var/scratch/spandey/leaf/femnist --learning-rate 0.001 --momentum 0.9 --partitioner dirichlet --alpha $ALPHA --seed $SEED --weighting-scheme label --check-teachers-accuracy > output_distill_femnist_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}.log 2>&1