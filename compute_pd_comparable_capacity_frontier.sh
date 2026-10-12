#!/usr/bin/env bash
# Linux equivalent of compute_pd_comparable_capacity_frontier.bat -- same behavior.
set -u

python3 scripts/compute_pd_comparable_capacity_frontier.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: compute_pd_comparable_capacity_frontier.py failed with exit code $status"
    exit "$status"
fi
