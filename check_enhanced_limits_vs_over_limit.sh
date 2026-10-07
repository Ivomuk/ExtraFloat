#!/usr/bin/env bash
# Linux equivalent of check_enhanced_limits_vs_over_limit.bat -- same behavior.
set -u

python3 scripts/check_enhanced_limits_vs_over_limit.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_enhanced_limits_vs_over_limit.py failed with exit code $status"
    exit "$status"
fi
