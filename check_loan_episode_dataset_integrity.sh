#!/usr/bin/env bash
# Linux equivalent of check_loan_episode_dataset_integrity.bat -- same behavior.
set -u

python3 scripts/check_loan_episode_dataset_integrity.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_loan_episode_dataset_integrity.py failed with exit code $status"
    exit "$status"
fi
