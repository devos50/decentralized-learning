#!/bin/bash

# Usage: ./script.sh <command_to_run_with_seed_placeholder> <seed1> <seed2> ...
# Example: ./script.sh "python my_script.py --seed {}" 42 123 456 789

set -e

if [ "$#" -lt 2 ]; then
  echo "Error: Please provide the command to run with a seed placeholder ({}) and a list of seeds."
  echo "Usage: $0 <command_to_run_with_seed_placeholder> <seed1> <seed2> ..."
  exit 1
fi

command_to_run=$1
shift 1  # Remove the first argument to keep only the seeds

# Use the provided seeds and run the command with each seed in parallel
printf "%s\n" "$@" | xargs -I {} -P "$#" bash -c "SEED={} && echo \"Running command with seed: \$SEED\" && ${command_to_run/\{\}/\$SEED} > output_\$SEED.log 2>&1"
