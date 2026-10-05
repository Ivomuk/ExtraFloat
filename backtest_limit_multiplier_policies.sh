#!/usr/bin/env bash
# Linux equivalent of backtest_limit_multiplier_policies.bat -- same behavior.
set -u

python3 scripts/backtest_limit_multiplier_policies.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: backtest_limit_multiplier_policies.py failed with exit code $status"
    exit "$status"
fi
