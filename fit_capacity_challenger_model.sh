#!/usr/bin/env bash
# Linux equivalent of fit_capacity_challenger_model.bat -- same behavior.
set -u

python3 scripts/fit_capacity_challenger_model.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: fit_capacity_challenger_model.py failed with exit code $status"
    exit "$status"
fi
