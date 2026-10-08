#!/usr/bin/env bash
# Linux equivalent of check_supported_exposure_boundary_robustness.bat -- same behavior.
set -u

python3 scripts/check_supported_exposure_boundary_robustness.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_supported_exposure_boundary_robustness.py failed with exit code $status"
    exit "$status"
fi
