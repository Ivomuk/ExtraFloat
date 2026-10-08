#!/usr/bin/env bash
# Linux equivalent of analyze_capacity_dimension_redundancy.bat -- same behavior.
set -u

python3 scripts/analyze_capacity_dimension_redundancy.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_capacity_dimension_redundancy.py failed with exit code $status"
    exit "$status"
fi
