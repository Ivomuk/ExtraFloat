#!/usr/bin/env bash
# Linux equivalent of characterize_capacity_times_c3_dollar_consequence.bat -- same behavior.
set -u

python3 scripts/characterize_capacity_times_c3_dollar_consequence.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: characterize_capacity_times_c3_dollar_consequence.py failed with exit code $status"
    exit "$status"
fi
