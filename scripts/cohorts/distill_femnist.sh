#!/bin/bash

for timestamp in "$@"; do
  echo "Starting distillation process at time $timestamp"
  python3 -u scripts/distill.py $PWD/data n_355_femnist_iid_sd24082 femnist mnist --distill-timestamp $timestamp > "distill_${timestamp}.log" 2>&1 &
done

wait