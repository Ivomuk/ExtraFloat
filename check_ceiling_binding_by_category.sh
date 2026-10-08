#!/usr/bin/env bash
# Linux equivalent of check_ceiling_binding_by_category.bat -- same behavior.
set -u

python3 scripts/check_ceiling_binding_by_category.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_ceiling_binding_by_category.py failed with exit code $status"
    exit "$status"
fi
