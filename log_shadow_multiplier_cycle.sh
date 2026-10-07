#!/usr/bin/env bash
# Linux equivalent of log_shadow_multiplier_cycle.bat -- same behavior.
set -u

python3 scripts/log_shadow_multiplier_cycle.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: log_shadow_multiplier_cycle.py failed with exit code $status"
    exit "$status"
fi
