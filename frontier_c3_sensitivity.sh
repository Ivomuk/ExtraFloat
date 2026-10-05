#!/usr/bin/env bash
# Linux equivalent of frontier_c3_sensitivity.bat -- same behavior.
set -u

python3 scripts/frontier_c3_sensitivity.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: frontier_c3_sensitivity.py failed with exit code $status"
    exit "$status"
fi
