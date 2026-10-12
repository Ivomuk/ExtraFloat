#!/usr/bin/env bash
# Linux equivalent of characterize_capacity_candidate_k_distribution.bat -- same behavior.
set -u

python3 scripts/characterize_capacity_candidate_k_distribution.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: characterize_capacity_candidate_k_distribution.py failed with exit code $status"
    exit "$status"
fi
