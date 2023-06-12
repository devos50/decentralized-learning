#!/bin/bash

for timestamp in "$@"; do
  echo "Starting distillation process at time $timestamp"
  python3 -u scripts/distill.py $PWD/data n_100_cifar10_s0_a1_sf1.000000_sd24082 cifar10 cifar100 --distill-timestamp $timestamp > "distill_${timestamp}.log" 2>&1 &
done

wait