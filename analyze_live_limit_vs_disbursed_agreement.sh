#!/usr/bin/env bash
# Linux equivalent of analyze_live_limit_vs_disbursed_agreement.bat -- same behavior.
set -u

python3 scripts/analyze_live_limit_vs_disbursed_agreement.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_live_limit_vs_disbursed_agreement.py failed with exit code $status"
    exit "$status"
fi
