#!/usr/bin/env bash
# Linux equivalent of build_historical_pit_transaction_features.bat -- same behavior.
set -u

python3 scripts/build_historical_pit_transaction_features.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: build_historical_pit_transaction_features.py failed with exit code $status"
    exit "$status"
fi
