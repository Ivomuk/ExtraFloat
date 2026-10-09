#!/usr/bin/env bash
# Linux equivalent of derive_capacity_frontier_from_business_state.bat -- same behavior.
set -u

python3 scripts/derive_capacity_frontier_from_business_state.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: derive_capacity_frontier_from_business_state.py failed with exit code $status"
    exit "$status"
fi
