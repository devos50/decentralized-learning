#!/bin/bash

# Usage: ./script.sh <command_to_run_with_sample_size_placeholder> <sample_size1> <sample_size2> ...
# Example: ./script.sh "python my_script.py --sample_size {}" 5 10 20

set -e

if [ "$#" -lt 2 ]; then
  echo "Error: Please provide the command to run with a sample_size placeholder ({}) and a list of sample sizes."
  echo "Usage: $0 <command_to_run_with_sample_size_placeholder> <sample_size1> <sample_size2> ..."
  exit 1
fi

command_to_run=$1
shift 1  # Remove the first argument to keep only the sample sizes

# Use the provided sample sizes and run the command with each sample size in parallel
printf "%s\n" "$@" | xargs -I {} -P "$#" bash -c "SAMPLE_SIZE={} && echo \"Running command with sample size: \$SAMPLE_SIZE\" && ${command_to_run/\{\}/\$SAMPLE_SIZE} > output_\$SAMPLE_SIZE.log 2>&1"
