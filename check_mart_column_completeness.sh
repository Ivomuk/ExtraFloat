#!/usr/bin/env bash
# Linux equivalent of check_mart_column_completeness.bat.
set -u

python3 scripts/check_mart_column_completeness.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: check_mart_column_completeness.py failed with exit code $status"
    exit "$status"
fi
