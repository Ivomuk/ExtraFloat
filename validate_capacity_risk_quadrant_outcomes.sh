#!/usr/bin/env bash
# Linux equivalent of validate_capacity_risk_quadrant_outcomes.bat -- same behavior.
set -u

python3 scripts/validate_capacity_risk_quadrant_outcomes.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: validate_capacity_risk_quadrant_outcomes.py failed with exit code $status"
    exit "$status"
fi
