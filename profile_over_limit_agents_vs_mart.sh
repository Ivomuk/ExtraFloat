#!/usr/bin/env bash
# Linux equivalent of profile_over_limit_agents_vs_mart.bat -- same behavior.
set -u

python3 scripts/profile_over_limit_agents_vs_mart.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: profile_over_limit_agents_vs_mart.py failed with exit code $status"
    exit "$status"
fi
