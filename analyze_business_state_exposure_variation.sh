#!/usr/bin/env bash
# Linux equivalent of analyze_business_state_exposure_variation.bat -- same behavior.
set -u

python3 scripts/analyze_business_state_exposure_variation.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_business_state_exposure_variation.py failed with exit code $status"
    exit "$status"
fi
