#!/usr/bin/env bash
# Linux equivalent of fit_shadow_risk_calibration.bat -- same behavior.
set -u

python3 scripts/fit_shadow_risk_calibration.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: fit_shadow_risk_calibration.py failed with exit code $status"
    exit "$status"
fi
