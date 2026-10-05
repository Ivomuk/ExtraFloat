#!/usr/bin/env bash
# Linux equivalent of calibrate_pd_isotonic_risk_curve.bat -- same behavior.
set -u

python3 scripts/calibrate_pd_isotonic_risk_curve.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: calibrate_pd_isotonic_risk_curve.py failed with exit code $status"
    exit "$status"
fi
