#!/usr/bin/env bash
# Linux equivalent of analyze_capacity_level_vs_absolute_exposure.bat -- same behavior.
set -u

python3 scripts/analyze_capacity_level_vs_absolute_exposure.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_capacity_level_vs_absolute_exposure.py failed with exit code $status"
    exit "$status"
fi
