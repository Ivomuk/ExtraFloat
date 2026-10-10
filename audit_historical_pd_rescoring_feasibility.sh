#!/usr/bin/env bash
# Linux equivalent of audit_historical_pd_rescoring_feasibility.bat -- same behavior.
set -u

python3 scripts/audit_historical_pd_rescoring_feasibility.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: audit_historical_pd_rescoring_feasibility.py failed with exit code $status"
    exit "$status"
fi
