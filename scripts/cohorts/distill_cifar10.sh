#!/bin/bash

for timestamp in "$@"; do
  echo "Starting distillation process at time $timestamp"
  python3 -u scripts/distill.py $PWD/data n_200_cifar10_dirichlet0.100000_sd24082 cifar10 cifar100 --partitioner dirichlet --alpha 0.1 --distill-timestamp $timestamp > "distill_${timestamp}.log" 2>&1 &
done

wait