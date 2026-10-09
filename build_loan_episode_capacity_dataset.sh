#!/usr/bin/env bash
# Linux equivalent of build_loan_episode_capacity_dataset.bat -- same behavior.
set -u

python3 scripts/build_loan_episode_capacity_dataset.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: build_loan_episode_capacity_dataset.py failed with exit code $status"
    exit "$status"
fi
