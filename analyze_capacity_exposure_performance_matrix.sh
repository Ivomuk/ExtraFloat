#!/usr/bin/env bash
# Linux equivalent of analyze_capacity_exposure_performance_matrix.bat -- same behavior.
set -u

python3 scripts/analyze_capacity_exposure_performance_matrix.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_capacity_exposure_performance_matrix.py failed with exit code $status"
    exit "$status"
fi
