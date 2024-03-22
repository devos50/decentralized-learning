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
PATIENCE=100
PEERS=3550
LR=0.1
CLUSTER_METHOD="uniform"
DATASET="femnist"

python3 -u scripts/cohorts/create_cohort_file.py $PEERS $COHORTS \
--seed $SEED \
--partitioner dirichlet \
--alpha $ALPHA \
--dataset $DATASET \
--method $CLUSTER_METHOD \
--output "data/cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_a${ALPHA}_${CLUSTER_METHOD}.txt" \
--data-dir /mnt/nfs/devos/leaf/${DATASET}

python3 -u simulations/dfl/${DATASET}.py \
--dataset-base-path "/mnt/nfs/devos" \
--peers $PEERS \
--local-steps 5 \
--cohort-participation-fraction $PARTICIPATION \
--duration 0 \
--num-aggregators 1 \
--activity-log-interval 3600 \
--bypass-model-transfers \
--seed $SEED \
--stop-criteria-patience $PATIENCE \
--capability-trace data/client_device_capacity \
--accuracy-logging-interval 0 \
--partitioner dirichlet \
--alpha $ALPHA \
--fix-aggregator \
--cohort-file "cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_a${ALPHA}_${CLUSTER_METHOD}.txt" \
--compute-validation-loss-global-model \
--log-level "ERROR" \
--learning-rate $LR \
--checkpoint-interval 18000 \
--checkpoint-interval-is-in-sec > output_${DATASET}_c${COHORTS}_s${SEED}_a${ALPHA}_p${PARTICIPATION}_${CLUSTER_METHOD}.log 2>&1

python3 -u scripts/ensembles.py data n_${PEERS}_${DATASET}_dirichlet${ALPHA}_sd${SEED}_ct${COHORTS}_p${PARTICIPATION}_dfl ${DATASET} \
--cohort cohorts/cohorts_n${PEERS}_c${COHORTS}_s${SEED}_a${ALPHA}_${CLUSTER_METHOD}.txt \
--partitioner dirichlet \
--alpha $ALPHA \
--seed $SEED \
--test-interval 5 \
--data-dir /mnt/nfs/devos/leaf/${DATASET} > output_ensembles_${DATASET}_n${PEERS}_c${COHORTS}_s${SEED}_a${ALPHA}_${CLUSTER_METHOD}.txt 2>&1
