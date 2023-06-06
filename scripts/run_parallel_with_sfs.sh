#!/bin/bash

# Usage: ./script.sh <command_to_run_with_sfs_placeholder> <sf1> <sf1> ...
# Example: ./script.sh "python my_script.py --success-fraction {}" 5 10 20

set -e

if [ "$#" -lt 2 ]; then
  echo "Error: Please provide the command to run with a success_fraction placeholder ({}) and a list of sample sizes."
  echo "Usage: $0 <command_to_run_with_success_fraction_placeholder> <sf1> <sf2> ..."
  exit 1
fi

command_to_run=$1
shift 1  # Remove the first argument to keep only the sample sizes

# Use the provided sample sizes and run the command with each sample size in parallel
printf "%s\n" "$@" | xargs -I {} -P "$#" bash -c "SUCCESS_FRACTION={} && echo \"Running command with sample size: \$SUCCESS_FRACTION\" && ${command_to_run/\{\}/\$SUCCESS_FRACTION} > output_\$SUCCESS_FRACTION.log 2>&1"
