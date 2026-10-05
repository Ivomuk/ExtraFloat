#!/usr/bin/env bash
# Linux equivalent of analyze_risk_tier_pd_resolution.bat -- same behavior.
set -u

python3 scripts/analyze_risk_tier_pd_resolution.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_risk_tier_pd_resolution.py failed with exit code $status"
    exit "$status"
fi
