#!/usr/bin/env bash
# Linux equivalent of analyze_category_limit_disbursement_frequency.bat -- same behavior.
set -u

python3 scripts/analyze_category_limit_disbursement_frequency.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_category_limit_disbursement_frequency.py failed with exit code $status"
    exit "$status"
fi
