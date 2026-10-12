#!/usr/bin/env bash
# Linux equivalent of combine_capacity_candidate_k_with_frozen_c3.bat -- same behavior.
set -u

python3 scripts/combine_capacity_candidate_k_with_frozen_c3.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: combine_capacity_candidate_k_with_frozen_c3.py failed with exit code $status"
    exit "$status"
fi
