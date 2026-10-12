#!/usr/bin/env bash
# Linux equivalent of apply_retrospective_current_isotonic_calibration.bat -- same behavior.
set -u

python3 scripts/apply_retrospective_current_isotonic_calibration.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: apply_retrospective_current_isotonic_calibration.py failed with exit code $status"
    exit "$status"
fi
