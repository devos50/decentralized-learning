#!/bin/bash

# Usage: ./script.sh <number_of_parallel_processes> <command_to_run_with_seed_placeholder>
# Example: ./script.sh 4 "python my_script.py --seed {}"

set -e

if [ "$#" -ne 2 ]; then
  echo "Error: Please provide the number of parallel processes and the command to run with a seed placeholder ({})."
  echo "Usage: $0 <number_of_parallel_processes> <command_to_run_with_seed_placeholder>"
  exit 1
fi

num_parallel_processes=$1
command_to_run=$2

# Generate a list of numbers from 1 to the number of parallel processes
seq 1 "$num_parallel_processes" | xargs -I {} -P "$num_parallel_processes" bash -c "SEED=\$RANDOM && echo \"Running command with seed: \$SEED\" && ${command_to_run/\{\}/\$SEED} > output_\$SEED.log 2>&1"
