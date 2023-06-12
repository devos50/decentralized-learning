#!/bin/bash

for cohort_value in {0..9}; do
  echo "Evaluating cohort $cohort_value"
  python3 -u simulations/dfl/cifar10.py \
  --peers 100 \
  --num-aggregators 1 \
  --activity-log-interval 60 \
  --accuracy-logging-interval 5 \
  --duration 0 \
  --rounds 1000 \
  --fixed-training-time 30 \
  --bypass-model-transfers \
  --seed 24082 \
  --torch-threads 6 \
  --fix-aggregator \
  --checkpoint-interval 1800 \
  --checkpoint-interval-is-in-sec
  --cohort-file "cohorts.txt" \
  --cohort $cohort_value > "output_${cohort_value}.log" 2>&1 &
done

wait