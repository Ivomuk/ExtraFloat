#!/usr/bin/env bash
# Linux equivalent of check_live_shadow_vs_category_limit.bat -- same behavior.
set -u

python3 scripts/check_live_shadow_vs_category_limit.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_live_shadow_vs_category_limit.py failed with exit code $status"
    exit "$status"
fi
