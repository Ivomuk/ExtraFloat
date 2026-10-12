#!/usr/bin/env bash
# Linux equivalent of apply_retrospective_shadow_c3_multiplier.bat -- same behavior.
set -u

python3 scripts/apply_retrospective_shadow_c3_multiplier.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: apply_retrospective_shadow_c3_multiplier.py failed with exit code $status"
    exit "$status"
fi
