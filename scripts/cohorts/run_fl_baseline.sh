#!/bin/bash

python3 -u simulations/dfl/cifar10.py \
--peers 100 \
--num-aggregators 1 \
--activity-log-interval 60 \
--accuracy-logging-interval 5 \
--duration 0 \
--rounds 500 \
--bypass-model-transfers \
--capability-trace data/client_device_capacity \
--seed 24082 \
--fix-aggregator \
--checkpoint-interval 5