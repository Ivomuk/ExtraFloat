#!/usr/bin/env bash
# Linux equivalent of compare_shadow_vs_live_multiplier.bat -- same behavior.
set -u

python3 scripts/compare_shadow_vs_live_multiplier.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: compare_shadow_vs_live_multiplier.py failed with exit code $status"
    exit "$status"
fi
