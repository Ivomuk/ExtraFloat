#!/usr/bin/env bash
# Linux equivalent of analyze_loan_frequency_for_snapshot_cadence.bat -- same behavior.
set -u

python3 scripts/analyze_loan_frequency_for_snapshot_cadence.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_loan_frequency_for_snapshot_cadence.py failed with exit code $status"
    exit "$status"
fi
