#!/usr/bin/env bash
# Linux equivalent of analyze_business_state_evolution.bat -- same behavior.
set -u

python3 scripts/analyze_business_state_evolution.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_business_state_evolution.py failed with exit code $status"
    exit "$status"
fi
