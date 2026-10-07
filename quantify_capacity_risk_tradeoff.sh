#!/usr/bin/env bash
# Linux equivalent of quantify_capacity_risk_tradeoff.bat -- same behavior.
set -u

python3 scripts/quantify_capacity_risk_tradeoff.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: quantify_capacity_risk_tradeoff.py failed with exit code $status"
    exit "$status"
fi
