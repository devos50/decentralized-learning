#!/bin/bash

set -e

if [ "$#" -lt 2 ]; then
  echo "Error: Please provide the command to run with a learning rate placeholder ({}) and a list of learning rates."
  echo "Usage: $0 <command_to_run_with_lr_placeholder> <lr1> <lr1> ..."
  exit 1
fi

command_to_run=$1
shift 1  # Remove the first argument to keep only the seeds

# Use the provided seeds and run the command with each seed in parallel
printf "%s\n" "$@" | xargs -I {} -P "$#" bash -c "LEARNING_RATE={} && echo \"Running command with learning rate: \$LEARNING_RATE\" && ${command_to_run/\{\}/\$LEARNING_RATE} > output_\$LEARNING_RATE.log 2>&1"
