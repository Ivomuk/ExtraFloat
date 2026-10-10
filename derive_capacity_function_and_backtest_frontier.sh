#!/usr/bin/env bash
# Linux equivalent of derive_capacity_function_and_backtest_frontier.bat -- same behavior.
set -u

python3 scripts/derive_capacity_function_and_backtest_frontier.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: derive_capacity_function_and_backtest_frontier.py failed with exit code $status"
    exit "$status"
fi
