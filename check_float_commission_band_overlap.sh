#!/usr/bin/env bash
# Linux equivalent of check_float_commission_band_overlap.bat -- same behavior.
set -u

python3 scripts/check_float_commission_band_overlap.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_float_commission_band_overlap.py failed with exit code $status"
    exit "$status"
fi
